# 真机推理说明

这里仅用于加载训练好的 checkpoint 并输出动作。

## 必需的 checkpoint 文件

- `last_model.pth`
- `model_scaler.pkl`
- `.hydra/config.yaml`

## 环境安装要求

先安装基础依赖：

```bash
pip install -r requirements.txt
```

### 安装 mamba1/2

```bash
pip install mamba-ssm[causal-conv1d] --no-build-isolation
```

### 安装 xlstm

```bash
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
pip install -e . --ignore-requires-python
```

- `pip install timm`
- `pip install transformers`
- `pip install torchsde torchdiffeq`

## 命令行推理

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-description "pick up the object and place it in the basket"
```

如果你已经有预计算好的任务 embedding：

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-name pick_up_the_object_and_place_it_in_the_basket \
  --task-embedding-path /path/to/task_embeddings.pkl
```

## 离线单步测试

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-description "pick up the object and place it in the basket" \
  --demo-hdf5 /path/to/demo.hdf5 \
  --demo-key demo_0 \
  --step 0
```

## Python 调用方式

```python
from real_robot.infer import RealRobotPolicy

policy = RealRobotPolicy(
    checkpoint_dir="/path/to/checkpoint_dir",
    task_description="pick up the object and place it in the basket",
)

action = policy.predict_action(
    agentview_image=agentview_rgb,
    eye_in_hand_image=eye_in_hand_rgb,
    robot_states=robot_state,
)
```
