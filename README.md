# Real_IL 纯推理版

这个工作区已经精简为“真机控制推理”用途，只保留加载训练权重并输出动作所需的代码。

## 当前保留内容

- `agents/`：策略 agent、backbone、encoder、model 实现
- `real_robot/infer.py`：加载 checkpoint 并执行一步推理
- `real_robot/README.md`：真机推理使用说明
- `task_embeddings/`：可选的任务 embedding 文件

## 已删除内容

- 仿真环境与 rollout 评测代码
- 训练入口、trainer、批处理脚本
- wandb 缓存、论文文件、实验残留文件
- 重复的 `copy` 文件和外部环境占位目录

## 环境安装

先创建 conda 环境，并按你的 CUDA 版本安装 `torch` 和 `torchvision`。
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然后安装基础推理依赖：

```bash
pip install -r requirements.txt
```

下面两段按照上游 `ALRhub/X_IL` README 的安装写法保留。

### 安装 mamba1/2

```bash
pip install mamba-ssm[causal-conv1d] --no-build-isolation
```

说明：

- 当前工作区里的 Mamba backbone 直接依赖 `mamba_ssm`。
- 为了保证 Mamba 权重能直接加载，这一步按上游 README 保留。

### 安装 xlstm

```bash
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
pip install -e . --ignore-requires-python
```

说明：

- 当前工作区已经保留了 `agents/backbones/xlstm/` 代码。
- 按你的要求，README 仍然按照上游 README 保留 `xlstm` 的安装步骤，统一覆盖 xLSTM 权重运行环境。
- xLSTM 一般还要求 CUDA PyTorch / Triton 环境可用。

### 其他按权重类型安装

- `pip install timm`：如果 checkpoint 使用 `pretrained_resnet` 编码器
- `pip install transformers`：如果 checkpoint 使用 `siglip` 编码器
- `pip install torchsde torchdiffeq`：如果 checkpoint 是 BESO

## 推理运行

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-description "pick up the object and place it in the basket"
```

如果要做单帧离线 smoke test：

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-description "pick up the object and place it in the basket" \
  --demo-hdf5 /path/to/demo.hdf5 \
  --demo-key demo_0 \
  --step 0
```
