#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd /home/guqiuyi/workspace/openpi
#   bash velocity_guider/run_train_ddp.sh 4
#   bash velocity_guider/run_train_ddp.sh 8 velocity_guider/configs/train_v1.yaml

NUM_GPUS="${1:-4}"
CONFIG="${2:-velocity_guider/configs/train_v2.yaml}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1, got ${NUM_GPUS}" >&2
  exit 1
fi

echo "Training Velocity Guider with ${NUM_GPUS} GPU(s)"
echo "Config: ${CONFIG}"

uv run torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  velocity_guider/train.py \
  --config "${CONFIG}"
