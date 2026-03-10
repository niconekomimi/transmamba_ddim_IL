#!/usr/bin/env bash
set -euo pipefail

# 公平对比：DDPM vs DDIM（固定 steps=4）
# DDPM: num_sampling_steps=4
# DDIM: num_sampling_steps=4, ddim_steps=4, eta=0

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}
K=${K:-4}
DATA=${DATA:-"10"}
SEEDS=${SEEDS:-"0,1,2"}

# DDPM
python run.py --config-name=libero_config \
  --multirun \
  agents=ddpm_agent \
  agent_name=ddpm_encdec_transformer_mamba \
  group=paper_compare_ddpm_vs_ddim_object_T${T}_K${K} \
  task_suite=libero_object \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents/model=ddpm/ddpm_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  num_sampling_steps=${T} \
  +tag=DDPM

# DDIM
python run.py --config-name=libero_config \
  --multirun \
  agents=ddim_agent \
  agent_name=ddim_encdec_transformer_mamba \
  group=paper_compare_ddpm_vs_ddim_object_T${T}_K${K} \
  task_suite=libero_object \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents/model=ddim/ddim_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  eta=0.0 \
  num_sampling_steps=${T} \
  ddim_steps=${K} \
  agents.ddim_steps=${K} \
  +tag=DDIM
