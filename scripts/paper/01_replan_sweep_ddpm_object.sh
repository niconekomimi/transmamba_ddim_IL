#!/usr/bin/env bash
set -euo pipefail

# DDPM 上的 replan_every 消融：只改执行闭环频率，其余保持 DDPM baseline steps

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

T=${T:-4}
R=${R:-"1,2,4,10"}
DATA=${DATA:-"10"}
SEEDS=${SEEDS:-"0,1,2"}

python run.py --config-name=libero_config \
  --multirun \
  agents=ddpm_agent \
  agent_name=ddpm_encdec_transformer_mamba \
  group=paper_replan_sweep_ddpm_object_T${T} \
  task_suite=libero_object \
  traj_per_task=${DATA} \
  seed=${SEEDS} \
  agents.replan_every=${R} \
  agents/model=ddpm/ddpm_encdec_transformer_mamba \
  agents/obs_encoders=pretrained_resnet_film \
  agents.if_film_condition=True \
  num_sampling_steps=${T}
