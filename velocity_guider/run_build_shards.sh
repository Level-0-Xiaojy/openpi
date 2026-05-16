#!/usr/bin/env bash
set -euo pipefail

# Launch N independent build_dataset.py shards, one process per visible GPU.
#
# Usage:
#   cd /home/guqiuyi/workspace/openpi
#   bash velocity_guider/run_build_shards.sh 4
#   bash velocity_guider/run_build_shards.sh 8 velocity_guider/configs/build_v1.yaml
#
# Notes:
# - Each process sees one GPU via CUDA_VISIBLE_DEVICES=<gpu_id>, so the YAML can keep device: cuda:0.
# - Each shard writes samples_shard_XXX.parquet. After all shards finish, the script runs --merge-shards.
# - If you want to clean old outputs, do it before launching, or set output.overwrite=true and launch shard 0
#   alone first. Concurrent deletion is intentionally avoided here.

NUM_SHARDS="${1:-4}"
CONFIG_PATH="${2:-velocity_guider/configs/build_v1.yaml}"

if ! [[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "NUM_SHARDS must be a positive integer, got: ${NUM_SHARDS}" >&2
  exit 2
fi

mkdir -p velocity_guider/logs

echo "Launching ${NUM_SHARDS} shards with config: ${CONFIG_PATH}"
pids=()
for ((shard_id=0; shard_id<NUM_SHARDS; shard_id++)); do
  log_path="velocity_guider/logs/build_shard_${shard_id}.log"
  echo "  shard ${shard_id}/${NUM_SHARDS} -> GPU ${shard_id}, log ${log_path}"
  CUDA_VISIBLE_DEVICES="${shard_id}" \
    uv run python velocity_guider/build_dataset.py \
      --config "${CONFIG_PATH}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-id "${shard_id}" \
      > "${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "${pid}"; then
    echo "shard ${i} finished"
  else
    echo "shard ${i} failed; see velocity_guider/logs/build_shard_${i}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one shard failed; skip merge." >&2
  exit 1
fi

echo "Merging shard parquet files..."
uv run python velocity_guider/build_dataset.py \
  --config "${CONFIG_PATH}" \
  --merge-shards

echo "Done."
