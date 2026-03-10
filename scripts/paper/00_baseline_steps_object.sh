#!/usr/bin/env bash
set -euo pipefail

# Baseline复现：固定(训练扩散步数T=num_sampling_steps, 推理采样步数K=ddim_steps)
# 覆盖：LIBERO-Object，traj_per_task=10/50，seed=0/1/2

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}        # num_sampling_steps / n_timesteps
K=${K:-4}        # ddim_steps
DATA=${DATA:-"10,50"}
SEEDS=${SEEDS:-"0,1,2"}

python run.py --config-name=libero_config \
  --multirun \
  agents=ddim_agent \
  agent_name=ddim_encdec_transformer_mamba \
  group=paper_baseline_object_T${T}_K${K} \
  task_suite=libero_object \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents/model=ddim/ddim_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  eta=0.0 \
  num_sampling_steps=${T} \
  ddim_steps=${K} \
  agents.ddim_steps=${K}
