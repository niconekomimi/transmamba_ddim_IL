# Real Robot Training

这个目录用于单独放置实机训练代码和数据，不改动原有仿真训练/评测路径。

## 目录约定

- `real_robot/data/*.hdf5`: 推荐的 LIBERO 风格，多任务 HDF5 目录
- `real_robot/data/demos2.hdf5`: 单个 HDF5 数据集文件，用于最小化测试
- 或 `real_robot/data/*.npz`: 直接放在 data 根目录的 npz episode
- `real_robot/outputs/...`: Hydra 运行目录、日志、权重、scaler、checkpoint

这里和 LIBERO 一样，`real_robot/data` 里只放原始训练数据，不单独准备测试集目录。
训练/验证划分由代码内部按 `val_ratio` 自动切分。

## LIBERO 风格 HDF5 目录

现在实机训练默认支持和 LIBERO 一样的目录组织方式，也就是一个目录下放多个任务文件：

- `real_robot/data/task_a_demo.hdf5`
- `real_robot/data/task_b_demo.hdf5`
- `real_robot/data/task_c_demo.hdf5`

数据集会自动：

- 扫描目录下所有 `.hdf5` / `.h5`
- 用文件名自动推断 `task_name`
- 把每个文件下 `/data/demo_x/...` 的轨迹合并进训练集
- 如果提供 `task_embedding_path`，优先按文件名对应的任务 embedding 加载

例如：

- `pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5`
- `pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5`

会自动映射成：

- `pick_up_the_milk_and_place_it_in_the_basket`
- `pick_up_the_orange_juice_and_place_it_in_the_basket`

这样就和 LIBERO 原始训练路径的 task-conditioned 方式一致了。

## 单个 HDF5 最小化测试

如果你现在只想先拿一个文件做最小化验证，也继续支持单个 HDF5。

## 你当前这份 HDF5 已支持

支持下面这种结构：

- `/data/demo_x/actions`
- `/data/demo_x/obs/agentview_rgb`
- `/data/demo_x/obs/eye_in_hand_rgb`
- `/data/demo_x/obs/joint_states`
- `/data/demo_x/obs/gripper_states`
- `/data/demo_x/robot_states`
- `/data/demo_x/states`

其中图像会自动映射成训练代码使用的：

- `agentview_image`
- `eye_in_hand_image`

`joint_states + gripper_states` 会优先拼成 `robot_states`。

为了更接近 LIBERO 的 goal-conditioned 训练，这个数据集现在还支持两种任务目标输入方式：

- 直接在配置里提供 `task_description`，训练时走语言编码器
- 提供 `task_embedding_path`，目录模式下按每个 HDF5 文件名自动匹配任务 embedding
- 提供 `task_embedding_path` + `task_name`，单文件模式下显式指定任务 embedding

推荐优先使用第二种，这样最接近仿真里直接给 `lang_emb` 的风格。

## 如果你还是想用 `.npz`

- 直接把所有 `.npz` 放进 `real_robot/data/`
- 不需要再建 `train/` 和 `val/` 子目录
- 训练/验证划分由 `val_ratio` 自动完成

- `actions`: `[T, action_dim]`
- `agentview_image`: `[T, H, W, C]` 或 `[T, C, H, W]`
- `eye_in_hand_image`: `[T, H, W, C]` 或 `[T, C, H, W]`

可选键：

- `robot_states`: `[T, state_dim]`
- `point_cloud`: `[T, N, 3]` 或 `[T, N, 6]`
- `lang_emb`: `[goal_dim]` 或 `[1, goal_dim]`
- `task_name`: 标量字符串

## 启动训练

需要从仓库根目录启动，这样可以复用原有 `configs/agents/*`、`configs/agents/model/*` 等配置组。

```bash
python real_robot/train.py \
  agents=ddim_agent \
  agents/model=ddim/ddim_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  task_description="pick up the alphabet soup and place it in the basket"
```

如果你按 LIBERO 风格把多个 HDF5 放进 `real_robot/data/`，默认配置就能直接加载：

```bash
python real_robot/train.py \
  agents=ddim_agent \
  agents/model=ddim/ddim_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  task_embedding_path=/path/to/task_embeddings.pkl
```

如果你要显式指定目录：

```bash
python real_robot/train.py dataset_path=/path/to/real_robot_hdf5_dir
```

如果你的文件是单个 HDF5，启动时覆盖即可：

```bash
python real_robot/train.py dataset_path=/path/to/your_data.hdf5
```

如果你已经有任务 embedding 的 pkl，单文件模式可以这样跑：

```bash
python real_robot/train.py \
  dataset_path=/path/to/your_data.hdf5 \
  task_name="pick_up_the_alphabet_soup_and_place_it_in_the_basket" \
  task_embedding_path=/path/to/task_embeddings.pkl
```

## 训练输出

默认输出到当前 run 目录：

- `last_model.pth`
- `model_scaler.pkl`
- `checkpoints/epoch_xxxx.pt`（按 `checkpoint_every_n_epochs` 控制）
- `.hydra/config.yaml`
- `run.log`

这样实机训练产物和原有仿真产物完全隔离。

## 实机推理

训练完成后，实机侧最少保留这些文件：

- `last_model.pth`
- `model_scaler.pkl`
- `.hydra/config.yaml`

现在可以直接用这个脚本加载：

```bash
python real_robot/infer.py \
  --ckpt-dir real_robot/outputs/runs/xxx/xxx/xxx \
  --task-description "pick up the alphabet soup and place it in the basket"
```

如果你已经有预计算任务 embedding：

```bash
python real_robot/infer.py \
  --ckpt-dir real_robot/outputs/runs/xxx/xxx/xxx \
  --task-name "pick_up_the_alphabet_soup_and_place_it_in_the_basket" \
  --task-embedding-path /path/to/task_embeddings.pkl
```

如果只想离线 smoke test 一步，可以直接从 HDF5 里取一帧观测：

```bash
python real_robot/infer.py \
  --ckpt-dir real_robot/outputs/runs/xxx/xxx/xxx \
  --task-description "pick up the alphabet soup and place it in the basket" \
  --demo-hdf5 real_robot/data/demos2.hdf5 \
  --demo-key demo_0 \
  --step 0
```

代码里也可以直接这样用：

```python
from real_robot.infer import RealRobotPolicy

policy = RealRobotPolicy(
    checkpoint_dir="real_robot/outputs/runs/xxx/xxx/xxx",
    task_description="pick up the alphabet soup and place it in the basket",
)

action = policy.predict_action(
    agentview_image=agentview_rgb,
    eye_in_hand_image=eye_in_hand_rgb,
    robot_states=robot_state,
)
```

其中：

- `agentview_image` / `eye_in_hand_image` 接收单帧图像 `[H, W, C]` 或 `[C, H, W]`
- `robot_states` 接收单步状态 `[D]`
- 任务目标可以通过 `task_description`、`task_name` 或 `task_embedding_path` 提供