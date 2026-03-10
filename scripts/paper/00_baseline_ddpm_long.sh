#!/usr/bin/env bash
set -euo pipefail

# DDPM baseline复现：LIBERO-10（论文表格里通常记为 Long）

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}
DATA=${DATA:-"10,50"}
SEEDS=${SEEDS:-"0,1,2"}

python run.py --config-name=libero_config \
  --multirun \
  agents=ddpm_agent \
  agent_name=ddpm_encdec_transformer_mamba \
  group=paper_baseline_ddpm_long_T${T} \
  task_suite=libero_10 \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents/model=ddpm/ddpm_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  num_sampling_steps=${T}
