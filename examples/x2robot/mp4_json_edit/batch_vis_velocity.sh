#!/bin/bash
DATA_DIR=/mnt/public/datasets/standardized_v1/x2robot/fold_towel/beijing_guqiuyi_20260412_pm_rollout
OUT_DIR=$DATA_DIR/vis_velocity

mkdir -p "$OUT_DIR"

for ep_dir in "$DATA_DIR"/*/; do
    ep_name=$(basename "$ep_dir")
    [ "$ep_name" = "vis_velocity" ] && continue
    # [[ "$ep_name" != *_fail ]] && continue # 只统计fail episode
    if ls "$ep_dir"*.json >/dev/null 2>&1; then
        echo "Processing: $ep_name"
        python3 /mnt/public/guqiuyi/openpi/examples/x2robot/mp4_json_edit/vis_x2robot_velocity.py \
            --data_dir "$DATA_DIR" \
            --episode "$ep_name" \
            --auto_highlight \
            --output "$OUT_DIR/vel_${ep_name}.png"
    fi
done