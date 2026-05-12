#!/usr/bin/env python3
"""
Visualize x2robot teleoperation acceleration.

Acceleration is computed as the finite difference of velocity:
    velocity = d(signal) / dt
    acceleration = d(velocity) / dt
"""

import argparse
import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


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


def derivative(data: np.ndarray, hz: int):
    dt = 1.0 / hz
    if len(data) < 2:
        return np.zeros_like(data, dtype=np.float32)
    if data.ndim == 1:
        return np.gradient(data, dt)
    return np.gradient(data, dt, axis=0)


def compute_acceleration(signals: dict, hz: int):
    acc = {}
    for key, data in signals.items():
        vel = derivative(data, hz)
        acc[key] = derivative(vel, hz)
    return acc


def find_auto_highlight(acc_signals: dict, hz: int, window: float = 2.0):
    left_norm = np.linalg.norm(acc_signals["master_left_position"], axis=1)
    right_norm = np.linalg.norm(acc_signals["master_right_position"], axis=1)
    idx = int(np.argmax(np.maximum(left_norm, right_norm)))
    t = idx / hz
    return max(0.0, t - window), t + window


def plot_acceleration(acc_signals, total_frames, episode_name, hz=20,
                      highlight_start=None, highlight_end=None, output_path=None):
    time = np.arange(total_frames) / hz

    subplot_plan = [
        [
            ("Left Acc Position x", "follow_left_position", "master_left_position", 0),
            ("Left Acc Position y", "follow_left_position", "master_left_position", 1),
            ("Left Acc Position z", "follow_left_position", "master_left_position", 2),
            ("Left Acc Gripper", "follow_left_gripper", "master_left_gripper", None),
        ],
        [
            ("Left Acc Rotation roll", "follow_left_rotation", "master_left_rotation", 0),
            ("Left Acc Rotation pitch", "follow_left_rotation", "master_left_rotation", 1),
            ("Left Acc Rotation yaw", "follow_left_rotation", "master_left_rotation", 2),
        ],
        [
            ("Right Acc Position x", "follow_right_position", "master_right_position", 0),
            ("Right Acc Position y", "follow_right_position", "master_right_position", 1),
            ("Right Acc Position z", "follow_right_position", "master_right_position", 2),
            ("Right Acc Gripper", "follow_right_gripper", "master_right_gripper", None),
        ],
        [
            ("Right Acc Rotation roll", "follow_right_rotation", "master_right_rotation", 0),
            ("Right Acc Rotation pitch", "follow_right_rotation", "master_right_rotation", 1),
            ("Right Acc Rotation yaw", "follow_right_rotation", "master_right_rotation", 2),
        ],
    ]

    fig = plt.figure(figsize=(28, 16))
    fig.suptitle(
        f"Acceleration  |  Episode: {episode_name}\n"
        f"{total_frames} frames @ {hz}Hz = {total_frames / hz:.1f}s",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        4, 4, figure=fig, hspace=0.35, wspace=0.3,
        left=0.04, right=0.98, top=0.92, bottom=0.05,
    )

    follow_color = "#1f77b4"
    master_color = "#ff7f0e"

    for row_idx, row_plots in enumerate(subplot_plan):
        for col_idx, (title, fkey, mkey, comp) in enumerate(row_plots):
            follow_data = acc_signals[fkey]
            master_data = acc_signals[mkey]

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

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="upper right")

    if output_path is None:
        output_path = f"acc_{episode_name}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize x2robot teleoperation acceleration")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--episode", type=str, required=True, help="Episode folder name")
    parser.add_argument("--hz", type=int, default=20, help="Sampling frequency (default: 20)")
    parser.add_argument("--highlight_start", type=float, default=None)
    parser.add_argument("--highlight_end", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--auto_highlight", action="store_true",
                        help="Automatically highlight around max master position acceleration")
    parser.add_argument("--auto_highlight_window", type=float, default=2.0)
    args = parser.parse_args()

    raw = load_episode(args.data_dir, args.episode)
    total_frames = raw["total"]
    print(f"Episode: {args.episode}")
    print(f"Frames: {total_frames}, Duration: {total_frames / args.hz:.1f}s")

    signals = extract_signals(raw["data"])
    acc_signals = compute_acceleration(signals, args.hz)

    highlight_start = args.highlight_start
    highlight_end = args.highlight_end
    if args.auto_highlight:
        highlight_start, highlight_end = find_auto_highlight(
            acc_signals, args.hz, window=args.auto_highlight_window
        )
        print(f"Auto highlight: [{highlight_start:.2f}s, {highlight_end:.2f}s]")

    plot_acceleration(
        acc_signals,
        total_frames,
        args.episode,
        hz=args.hz,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()