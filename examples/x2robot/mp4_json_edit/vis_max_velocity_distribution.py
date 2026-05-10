#!/usr/bin/env python3
"""
分析一批 episode 的最大速度分布，用直方图可视化。

"最大速度"的默认定义与 annotate_action_frequency.py 一致：
  master left/right position z 方向速度的正向峰值。
也可切到 --metric max_speed：双臂 position xyz 合成速度模的峰值。

Usage:
    python3 vis_max_velocity_distribution.py \
        --data_dirs \
            /mnt/public/datasets/standardized_v1/x2robot/fold_towel/beijing_guqiuyi_20260420_pm_tele \
            /mnt/public/datasets/standardized_v1/x2robot/fold_towel/beijing_guqiuyi_20260317_pm_tele \
        --bin_width 0.1 \
        --output max_vel_hist.png
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def extract_master_position(data_list: list) -> tuple[np.ndarray, np.ndarray]:
    left = np.array([d["master_left_position"] for d in data_list])
    right = np.array([d["master_right_position"] for d in data_list])
    return left, right


def max_velocity_peakz(left_pos, right_pos, hz):
    dt = 1.0 / hz
    lvz = np.gradient(left_pos[:, 2], dt)
    rvz = np.gradient(right_pos[:, 2], dt)
    return float(max(lvz.max(), rvz.max()))


def max_velocity_speed(left_pos, right_pos, hz):
    dt = 1.0 / hz
    lv = np.gradient(left_pos, dt, axis=0)
    rv = np.gradient(right_pos, dt, axis=0)
    lspeed = np.linalg.norm(lv, axis=1)
    rspeed = np.linalg.norm(rv, axis=1)
    return float(max(lspeed.max(), rspeed.max()))


METRICS = {
    "peak_vz":   max_velocity_peakz,
    "max_speed": max_velocity_speed,
}


def _match_select(ep_name: str, select: str) -> bool:
    if select == "all":
        return True
    # 大小写不敏感，按后缀匹配（容忍尾部下划线/连字符之类的无所谓，用 endswith 的小写比较）
    name_lower = ep_name.lower()
    if select == "success":
        return name_lower.endswith("success")
    if select == "fail":
        return name_lower.endswith("fail")
    return True


def collect_episodes(data_dirs: list[str], select: str = "all") -> list[tuple[str, str, str]]:
    out = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"[WARN] skip non-dir: {data_dir}")
            continue
        for ep_name in sorted(os.listdir(data_dir)):
            ep_dir = os.path.join(data_dir, ep_name)
            if not os.path.isdir(ep_dir):
                continue
            if not _match_select(ep_name, select):
                continue
            json_files = [
                f for f in os.listdir(ep_dir)
                if f.endswith(".json")
                and not f.endswith(".bak")
                and not f.endswith("_annotated.json")
            ]
            if not json_files:
                continue
            out.append((data_dir, ep_name, os.path.join(ep_dir, json_files[0])))
    return out


def analyze(data_dirs, hz, metric, select="all"):
    metric_fn = METRICS[metric]
    episodes = collect_episodes(data_dirs, select=select)
    values, names = [], []
    for src, ep, path in episodes:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            left, right = extract_master_position(raw["data"])
            values.append(metric_fn(left, right, hz))
            names.append(f"{os.path.basename(src.rstrip('/'))}/{ep}")
        except Exception as e:
            print(f"[FAIL] {path}: {e}")
    return values, names


def plot_histogram(values, bin_width, output_path, metric, select="all"):
    arr = np.array(values)
    vmax = float(np.ceil(arr.max() / bin_width) * bin_width)
    bins = np.arange(0.0, vmax + bin_width, bin_width)
    counts, _ = np.histogram(arr, bins=bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    centers = (bins[:-1] + bins[1:]) / 2.0
    bars = ax.bar(centers, counts, width=bin_width * 0.9,
                  color="#4c78a8", edgecolor="black", alpha=0.85)

    for b, c in zip(bars, counts):
        if c > 0:
            ax.text(b.get_x() + b.get_width() / 2.0,
                    c + max(counts) * 0.01,
                    str(int(c)), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel(f"Max velocity ({metric}) [m/s]")
    ax.set_ylabel("# episodes")
    ax.set_title(
        f"Max velocity distribution "
        f"(n={len(arr)}, bin={bin_width}, metric={metric}, select={select})\n"
        f"min={arr.min():.3f}  median={np.median(arr):.3f}  "
        f"mean={arr.mean():.3f}  max={arr.max():.3f}"
    )
    ax.set_xticks(bins)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {output_path}")


def save_csv(values, names, csv_path):
    order = np.argsort(-np.array(values))
    with open(csv_path, "w") as f:
        f.write("episode,max_velocity\n")
        for i in order:
            f.write(f"{names[i]},{values[i]:.6f}\n")
    print(f"Saved per-episode values to: {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Batch analyze max-velocity distribution across episodes.")
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="One or more dataset root dirs; each contains episode subfolders.")
    p.add_argument("--hz", type=float, default=20.0, help="Recording frequency (default: 20)")
    p.add_argument("--metric", choices=list(METRICS.keys()), default="peak_vz",
                   help="'peak_vz' (与 annotate_action_frequency 一致) 或 'max_speed' (xyz 合成速度模)")
    p.add_argument("--bin_width", type=float, default=0.1, help="Histogram bin width")
    p.add_argument("--output", type=str, default="max_vel_hist.png", help="Output PNG path")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV dump of per-episode values")
    p.add_argument("--select", choices=["all", "success", "fail"], default="all",
                   help="只统计以 success/fail 结尾的 episode；默认 all 不过滤")
    args = p.parse_args()

    values, names = analyze(args.data_dirs, args.hz, args.metric, select=args.select)
    print(f"Collected {len(values)} episodes (select={args.select}).")
    if not values:
        print("No valid episodes found.")
        return

    plot_histogram(values, args.bin_width, args.output, args.metric, select=args.select)
    csv_path = args.csv or os.path.splitext(args.output)[0] + ".csv"
    # save_csv(values, names, csv_path)


if __name__ == "__main__":
    main()