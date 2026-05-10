#!/bin/bash
DATA_DIR=/mnt/public/guqiuyi/dataset/throw_sandbox/throw_sandbox_hy_0427
OUT_DIR=$DATA_DIR/vis_velocity

mkdir -p "$OUT_DIR"

for ep_dir in "$DATA_DIR"/*/; do
    ep_name=$(basename "$ep_dir")
    [ "$ep_name" = "vis_velocity" ] && continue
    if ls "$ep_dir"*.json >/dev/null 2>&1; then
        echo "Processing: $ep_name"
        python3 /mnt/public/guqiuyi/openpi/examples/x2robot/mp4_json_edit/vis_x2robot_velocity.py \
            --data_dir "$DATA_DIR" \
            --episode "$ep_name" \
            --auto_highlight \
            --output "$OUT_DIR/vel_${ep_name}.png"
    fi
done