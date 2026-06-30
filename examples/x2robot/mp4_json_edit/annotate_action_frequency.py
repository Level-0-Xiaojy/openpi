#!/usr/bin/env python3
"""
为 episode JSON 中的每帧添加 action_frequency 字段。

规则:
- 默认: 20Hz
- 在 highlight 窗口内 (master left/right position z 正向最大速度时刻 t, 范围 [t-2, t+2] 秒): 40Hz

批量处理指定 data_dir 下所有 episode 目录。

Usage:
    python3 annotate_action_frequency.py \
        --data_dir /mnt/public/datasets/standardized_v1/x2robot/fold_towel/beijing_guqiuyi_20260317_pm_tele \
        --inplace --backup
"""

import argparse
import json
import os
import shutil
import numpy as np


def extract_signals(data_list: list) -> dict:
    signals = {}
    for side in ["left", "right"]:
        for prefix in ["follow", "master"]:
            for field in ["position", "rotation", "gripper"]:
                key = f"{prefix}_{side}_{field}"
                signals[key] = np.array([d[key] for d in data_list])
    return signals


def find_highlight_window(signals: dict, hz: int, window: float = 2.0) -> tuple[float, float]:
    """在 master left/right position z 的速度中找出正向最大的时刻 t,
    返回 (t - window, t + window)（单位: 秒）。"""
    dt = 1.0 / hz
    left_vz = np.gradient(signals["master_left_position"][:, 2], dt)
    right_vz = np.gradient(signals["master_right_position"][:, 2], dt)
    combined = np.concatenate([left_vz, right_vz])
    idx = int(np.argmax(combined))
    frame_idx = idx if idx < len(left_vz) else idx - len(left_vz)
    t = frame_idx / hz
    return max(0.0, t - window), t + window


def annotate_episode(
    json_path: str,
    output_path: str,
    hz: int = 20,
    default_hz: int = 20,
    fast_hz: int = 40,
    window: float = 2.0,
) -> tuple[float, float]:
    with open(json_path, "r") as f:
        raw = json.load(f)

    data = raw["data"]
    signals = extract_signals(data)
    hs, he = find_highlight_window(signals, hz, window)

    for i, frame in enumerate(data):
        t = i / hz
        frame["action_frequency"] = fast_hz if hs <= t <= he else default_hz

    with open(output_path, "w") as f:
        json.dump(raw, f)

    return hs, he


def main():
    parser = argparse.ArgumentParser(description="批量为 episode JSON 添加 action_frequency 字段")
    parser.add_argument("--data_dir", type=str, required=True, help="数据根目录")
    parser.add_argument("--hz", type=int, default=20, help="采样频率 (用于计算速度/时间)")
    parser.add_argument("--default_hz", type=int, default=20, help="默认 action_frequency")
    parser.add_argument("--fast_hz", type=int, default=40, help="highlight 窗口内的 action_frequency")
    parser.add_argument("--window", type=float, default=2.0, help="highlight 前后窗口 (秒)")
    parser.add_argument("--inplace", action="store_true",
                        help="就地覆盖原 JSON, 否则输出为 *_annotated.json")
    parser.add_argument("--backup", action="store_true",
                        help="配合 --inplace 使用时, 备份为 *.json.bak")
    args = parser.parse_args()

    episodes = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )

    for ep_name in episodes:
        ep_dir = os.path.join(args.data_dir, ep_name)
        preferred = f"{ep_name}.json"
        json_files = [f for f in os.listdir(ep_dir) if f.endswith(".json") and not f.endswith("_annotated.json")]
        if preferred in json_files:
            json_path = os.path.join(ep_dir, preferred)
        else:
            print(f"[{ep_name}] No preferred JSON file found, only find {json_files}, using first JSON file")
            json_files = [f for f in json_files if f != "trim_manifest.json"]
            json_path = os.path.join(ep_dir, sorted(json_files)[0])

        if args.inplace:
            if args.backup:
                shutil.copy2(json_path, json_path + ".bak")
            output_path = json_path
        else:
            output_path = json_path.replace(".json", "_annotated.json")

        try:
            hs, he = annotate_episode(
                json_path, output_path,
                hz=args.hz, default_hz=args.default_hz,
                fast_hz=args.fast_hz, window=args.window,
            )
            print(f"[{ep_name}] highlight=[{hs:.2f}s, {he:.2f}s] -> {output_path}")
        except Exception as e:
            print(f"[{ep_name}] FAILED: {e}")


if __name__ == "__main__":
    main()