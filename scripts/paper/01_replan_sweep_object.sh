#!/usr/bin/env bash
set -euo pipefail

# 核心实验：只改推理期执行闭环频率 replan_every，其余(ckpt/steps/数据)保持不变
# 覆盖：LIBERO-Object，replan_every sweep，seed sweep

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}
K=${K:-4}
# act_seq_len=10 时，replan_every=10 等价旧版 chunk 开环执行
R=${R:-"1,2,4,10"}
DATA=${DATA:-"10"}
SEEDS=${SEEDS:-"0,1,2"}

python run.py --config-name=libero_config \
  --multirun \
  agents=ddim_agent \
  agent_name=ddim_encdec_transformer_mamba \
  group=paper_replan_sweep_object_T${T}_K${K} \
  task_suite=libero_object \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents.replan_every=${R} \
  agents/model=ddim/ddim_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  eta=0.0 \
  num_sampling_steps=${T} \
  ddim_steps=${K} \
  agents.ddim_steps=${K}
