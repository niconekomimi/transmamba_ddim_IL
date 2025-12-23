# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError(
        "Please install mamba-ssm and causal-conv1d. "
        "e.g., `pip install mamba-ssm causal-conv1d`"
    ) from e


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)


def modulate(x, shift, scale):
    return shift + (x * (scale))


class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__(dim, mixer_cls, mlp_cls, norm_cls, fused_add_norm, residual_in_fp32)

        self.adaLN_zero = AdaLNZero(dim)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, cond=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(cond)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

            hidden_states = modulate(hidden_states, shift_msa, scale_msa)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = gate_msa * hidden_states

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))

                hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)

                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )

            hidden_states = self.mlp(hidden_states)
            hidden_states = gate_mlp * hidden_states

        return hidden_states, residual


class MambaFiLMDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        resid_pdrop: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.in_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(resid_pdrop)

        # 时间嵌入 -> FiLM 参数
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim)
        )

        # 堆叠 Mamba 层
        self.layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(n_layers)
        ])

    def _apply_film(self, x: torch.Tensor, temb: torch.Tensor | None):
        # x: (B, L, D); temb: (B, 1, D)
        if temb is None:
            return x
        gamma, beta = self.film(temb).chunk(2, dim=-1)  # (B, 1, D), (B, 1, D)
        return x * (1 + gamma) + beta

    def forward(self, x: torch.Tensor, cond1: torch.Tensor | None = None, cond2: torch.Tensor | None = None):
        """
        兼容三种调用：
        - forward(x, temb, memory)
        - forward(x, temb)
        - forward(x, memory)
        其中 x=(B, Ta, D), temb=(B, 1, D), memory=(B, S, D or 1, D)
        """
        temb, memory = None, None
        if cond2 is not None:
            # 认为 (x, temb, memory)
            temb, memory = cond1, cond2
        else:
            # 只有一个条件，判断是 temb 还是 memory（按维度启发式）
            if cond1 is not None:
                if cond1.dim() == 3 and cond1.size(-1) == self.embed_dim:
                    # cond1 可能是 (B, 1, D) 的 temb 或 (B, S, D) 的 memory，二者均为 3 维
                    # 如果长度与 x 相同通常更像 memory，不过 context_token 下 memory 也可能是 1
                    # 统一策略：
                    # - 若长度为 1 且数值范围明显像时间步嵌入：作为 temb（但这里无法判别数值）
                    # - 简化：优先当作 memory；如需仅传 temb，请在上游按 (x, temb) 严格调用
                    memory = cond1
                else:
                    # 回退：当作 temb
                    temb = cond1

        # 将 memory 作为前缀拼接
        if memory is not None:
            x = torch.cat([memory, x], dim=1)  # (B, S+Ta, D)

        h = x
        for layer in self.layers:
            y = layer(self.in_norm(h))        # Mamba 期望 (B, L, D)
            y = self._apply_film(y, temb)     # 注入时间条件
            h = h + self.drop(y)              # 残差

        return self.out_norm(h)