#!/usr/bin/env python3
"""
Visualize x2robot pose data: 14 subplots (4+3+4+3 layout).
Each subplot shows follow_ vs master_ in different colors.

Usage:
    python visualize_tele.py --data_dir <path> --episode <folder_name> [--highlight_start <sec> --highlight_end <sec>] [--output <path>]

Examples:
    python visualize_tele.py --data_dir ~/xyf_projects/x2robot_data_partof/beijing_guqiuyi_20260410_pm_tele \
        --episode fold_multi_towels_gqy_0409@MASTER_SLAVE_MODE@2026_04_09_21_50_59

    python visualize_tele.py --data_dir ~/xyf_projects/x2robot_data_partof/beijing_guqiuyi_20260410_pm_tele \
        --episode fold_multi_towels_gqy_0409@MASTER_SLAVE_MODE@2026_04_09_21_50_59 \
        --highlight_start 2.0 --highlight_end 5.5
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_episode(data_dir: str, episode_name: str):
    ep_dir = os.path.join(data_dir, episode_name)
    if not os.path.isdir(ep_dir):
        available = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
        print(f"Episode '{episode_name}' not found. Available episodes:")
        for i, name in enumerate(available):
            print(f"  [{i}] {name}")
        raise SystemExit(1)

    json_files = [f for f in os.listdir(ep_dir) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {ep_dir}")

    with open(os.path.join(ep_dir, json_files[0]), "r") as f:
        return json.load(f)


def extract_signals(data_list: list):
    signals = {}
    for side in ["left", "right"]:
        for prefix in ["follow", "master"]:
            for field in ["position", "rotation", "gripper"]:
                key = f"{prefix}_{side}_{field}"
                signals[key] = np.array([d[key] for d in data_list])
    return signals


def plot_signals(signals, total_frames, episode_name, hz=20,
                 highlight_start=None, highlight_end=None,
                 view_start=None, view_end=None, output_path=None):
    time = np.arange(total_frames) / hz

    # Row 0 (4): left_pos x/y/z + left_gripper
    # Row 1 (3): left_rot roll/pitch/yaw
    # Row 2 (4): right_pos x/y/z + right_gripper
    # Row 3 (3): right_rot roll/pitch/yaw
    subplot_plan = [
        [
            ("Left Position x", "follow_left_position", "master_left_position", 0),
            ("Left Position y", "follow_left_position", "master_left_position", 1),
            ("Left Position z", "follow_left_position", "master_left_position", 2),
            ("Left Gripper",    "follow_left_gripper",  "master_left_gripper",  None),
        ],
        [
            ("Left Rotation roll",  "follow_left_rotation", "master_left_rotation", 0),
            ("Left Rotation pitch", "follow_left_rotation", "master_left_rotation", 1),
            ("Left Rotation yaw",   "follow_left_rotation", "master_left_rotation", 2),
        ],
        [
            ("Right Position x", "follow_right_position", "master_right_position", 0),
            ("Right Position y", "follow_right_position", "master_right_position", 1),
            ("Right Position z", "follow_right_position", "master_right_position", 2),
            ("Right Gripper",    "follow_right_gripper",  "master_right_gripper",  None),
        ],
        [
            ("Right Rotation roll",  "follow_right_rotation", "master_right_rotation", 0),
            ("Right Rotation pitch", "follow_right_rotation", "master_right_rotation", 1),
            ("Right Rotation yaw",   "follow_right_rotation", "master_right_rotation", 2),
        ],
    ]

    max_cols = 4
    fig = plt.figure(figsize=(28, 16))
    time_range_str = ""
    if view_start is not None or view_end is not None:
        vs = view_start if view_start is not None else 0.0
        ve = view_end if view_end is not None else total_frames / hz
        time_range_str = f"  |  View: [{vs:.1f}s, {ve:.1f}s]"
    fig.suptitle(
        f"Episode: {episode_name}\n"
        f"{total_frames} frames @ {hz}Hz = {total_frames / hz:.1f}s{time_range_str}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, max_cols, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.04, right=0.98, top=0.92, bottom=0.05)

    follow_color = "#1f77b4"
    master_color = "#ff7f0e"

    for row_idx, row_plots in enumerate(subplot_plan):
        for col_idx, (title, fkey, mkey, comp) in enumerate(row_plots):
            follow_data = signals[fkey]
            master_data = signals[mkey]

            if comp is not None:
                follow_y = follow_data[:, comp]
                master_y = master_data[:, comp]
            else:
                follow_y = follow_data
                master_y = master_data

            ax = fig.add_subplot(gs[row_idx, col_idx])

            if highlight_start is not None and highlight_end is not None:
                ax.axvspan(highlight_start, highlight_end, color="red", alpha=0.15, zorder=0)

            ax.plot(time, follow_y, color=follow_color, linewidth=0.8, label="follow", alpha=0.9)
            ax.plot(time, master_y, color=master_color, linewidth=0.8, label="master", alpha=0.9)

            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if view_start is not None or view_end is not None:
                ax.set_xlim(
                    view_start if view_start is not None else time[0],
                    view_end if view_end is not None else time[-1],
                )
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="upper right")

    if output_path is None:
        output_path = f"pose_{episode_name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize x2robot teleoperation data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--episode", type=str, required=True,
                        help="Episode folder name (e.g. fold_multi_towels_gqy_0409@MASTER_SLAVE_MODE@2026_04_09_21_50_59)")
    parser.add_argument("--hz", type=int, default=20, help="Sampling frequency (default: 20)")
    parser.add_argument("--highlight_start", type=float, default=None,
                        help="Start time (seconds) to highlight in red")
    parser.add_argument("--highlight_end", type=float, default=None,
                        help="End time (seconds) to highlight in red")
    parser.add_argument("--view_start", type=float, default=None,
                        help="Only show data from this time (seconds)")
    parser.add_argument("--view_end", type=float, default=None,
                        help="Only show data up to this time (seconds)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    raw = load_episode(args.data_dir, args.episode)
    total_frames = raw["total"]
    print(f"Episode: {args.episode}")
    print(f"Frames: {total_frames}, Duration: {total_frames / args.hz:.1f}s")

    signals = extract_signals(raw["data"])
    plot_signals(signals, total_frames, args.episode, hz=args.hz,
                 highlight_start=args.highlight_start,
                 highlight_end=args.highlight_end,
                 view_start=args.view_start,
                 view_end=args.view_end,
                 output_path=args.output)


if __name__ == "__main__":
    main()
