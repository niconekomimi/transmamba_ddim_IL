#!/usr/bin/env bash
set -euo pipefail

# 小网格消融：replan_every × ddim_steps
# 目的：证明提升主要来自闭环执行，而不是单纯采样步数变化。

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}
# 默认做两档采样步数；可按需改成 4,8,16
K_LIST=${K_LIST:-"2,4"}
R_LIST=${R_LIST:-"1,10"}
DATA=${DATA:-"10"}
SEEDS=${SEEDS:-"0,1,2"}

python run.py --config-name=libero_config \
  --multirun \ 
  agents=ddim_agent \
  agent_name=ddim_encdec_transformer_mamba \
  group=paper_grid_replanXddim_object_T${T} \
  task_suite=libero_object \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents.replan_every=${R_LIST} \
  agents/model=ddim/ddim_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  eta=0.0 \
  num_sampling_steps=${T} \
  ddim_steps=${K_LIST} \
  agents.ddim_steps=${K_LIST}
