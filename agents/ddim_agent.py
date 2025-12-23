import logging
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class DDIM_Agent(BaseAgent):
    """
    Agent that trains with the usual DDPM forward-noising objective,
    but uses DDIM sampling at inference.
    """

    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
            cam_names=cam_names,
        )

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        # 添加投影层：将编码器输出(512)映射到模型期望的维度(256)
        # self.perceptual_projection = nn.Linear(512, latent_dim).to(device)

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

        # ---------------- DDIM 配置 ----------------
        self.ddim_steps = ddim_steps
        self.eta = eta

        # 给 diffusion model 注入 DDIM 参数（假设 model 是 Diffusion 类或兼容实现）
        if hasattr(self.model, "eta"):
            self.model.eta = float(self.eta)
        if hasattr(self.model, "ddim_steps"):
            self.model.ddim_steps = self.ddim_steps or getattr(self.model, "n_timesteps", None)

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def forward(self, obs_dict, actions=None):
        """
        训练: if_train=True → DDPM loss
        推理: if_train=False → DDIM 采样
        """
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)
        
        # 添加投影层
        # perceptual_emb = self.perceptual_projection(perceptual_emb)

        if self.training and actions is not None:
            # DDPM training: predict noise / x0
            loss = self.model(perceptual_emb, latent_goal, action=actions, if_train=True)
            return loss
        else:
            # DDIM sampling
            predicted_action = self.model(perceptual_emb, latent_goal, if_train=False)
            return predicted_action
