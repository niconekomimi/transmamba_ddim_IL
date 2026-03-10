#!/usr/bin/env bash
set -euo pipefail

# 长时序（表格里“Long”通常对应 task_suite=libero_10）
# 目的：验证 replan_every 主要提升 Object，而不破坏 Long 的优势（或影响较小）。

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}
K=${K:-4}
R=${R:-"1,2,4,10"}
DATA=${DATA:-"10"}
SEEDS=${SEEDS:-"0,1,2"}

python run.py --config-name=libero_config \
  --multirun \
  agents=ddim_agent \
  agent_name=ddim_encdec_transformer_mamba \
  group=paper_replan_sweep_long_T${T}_K${K} \
  task_suite=libero_10 \
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
