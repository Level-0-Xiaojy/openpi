#!/bin/bash
# DATA_DIR=/mnt/public/datasets/standardized_v1/x2robot/fold_towel/beijing_guqiuyi_20260420_pm_tele
DATA_DIR=/mnt/public/guqiuyi/dataset/throw_sandbox/throw_sandbox_hy_0510
OUT_DIR=$DATA_DIR/vis_acc

mkdir -p "$OUT_DIR"

for ep_dir in "$DATA_DIR"/*/; do
    ep_name=$(basename "$ep_dir")
    [ "$ep_name" = "vis_acc" ] && continue

    if ls "$ep_dir"*.json >/dev/null 2>&1; then
        echo "Processing: $ep_name"
        python3 /mnt/public/guqiuyi/openpi/examples/x2robot/mp4_json_edit/vis_x2robot_acc.py \
            --data_dir "$DATA_DIR" \
            --episode "$ep_name" \
            --auto_highlight \
            --output "$OUT_DIR/acc_${ep_name}.png"
    fi
done