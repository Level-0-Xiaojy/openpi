#!/usr/bin/env python3
"""Visualize velocity comparison: SFT (v_mode=3 demo) vs Rollout (v_mode=2->3 conversion).

Generates two separate PNG figures:
1. SFT figure:     original position velocity + burst highlight
2. Rollout figure:  two rows of subplots (original trajectory velocity, converted
   trajectory velocity), each with burst highlight and summary stats.

The conversion simulates factor=2 execution of rollout chunks during burst:
  - At each burst frame t, take chunk = master[t:t+20]
  - Interpolate factor=2 -> 39 controller substeps at 60Hz
  - Downsample x3 -> ~13 frames at 20Hz (0.65s of motion)
  - Advance t by 13 and repeat until burst ends

Usage:
    cd /home/guqiuyi/workspace/openpi
    uv run python velocity_guider/visualize_vmode_conversion.py \
        --lerobot-root /mnt/public/guqiuyi/huggingface/lerobot \
        --sft-repo fold_towel_gqy_0420 \
        --rollout-repo fold_towel_gqy_0412 \
        --sft-episode 0 --rollout-episode 0 \
        --output-dir velocity_guider/pics/vmode_conversion
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from velocity_guider.data.lerobot_loader import LeRobotDatasetInfo
from velocity_guider.data.resample import resample_master_chunk

logger = logging.getLogger("velocity_guider.vis_vmode")

CHUNK_SIZE = 20


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def position_velocity(master: np.ndarray, hz: int) -> dict[str, np.ndarray]:
    dt = 1.0 / hz
    return {
        "left": np.gradient(master[:, 0:3], dt, axis=0),
        "right": np.gradient(master[:, 7:10], dt, axis=0),
    }


def combined_speed(vel: dict[str, np.ndarray]) -> np.ndarray:
    left_speed = np.linalg.norm(vel["left"], axis=1)
    right_speed = np.linalg.norm(vel["right"], axis=1)
    return np.maximum(left_speed, right_speed)


def detect_burst(
    speed: np.ndarray,
    *,
    threshold: float = 0.4,
    dilate_frames: int = 10,
) -> np.ndarray:
    is_burst = speed > threshold
    if dilate_frames > 0:
        kernel = np.ones(2 * dilate_frames + 1, dtype=float)
        is_burst = np.convolve(is_burst.astype(float), kernel, mode="same") > 0
    return is_burst


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            seg_start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((seg_start, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(mask)))
    return segments


def convert_vmode2_to_vmode3(
    master_full: np.ndarray,
    is_burst: np.ndarray,
) -> np.ndarray:
    """Build a new (shorter) trajectory simulating factor=2 execution during burst.

    Non-burst frames are copied as-is. For burst frames the algorithm steps
    forward in 20-frame chunks:
      1) chunk = master[t : t+20]
      2) factor=2 interpolation -> 2*(20-1)+1 = 39 points (60Hz controller path)
      3) downsample x3 -> 13 points (20Hz recording of factor=2 execution)
      4) append these 13 frames to the new trajectory
      5) advance t += 13 (0.65s elapsed)
      6) repeat while t is still in burst and t+20 <= T
    """
    T = master_full.shape[0]
    new_frames: list[np.ndarray] = []
    t = 0

    while t < T:
        if not is_burst[t]:
            new_frames.append(master_full[t:t + 1])
            t += 1
            continue

        if t + CHUNK_SIZE > T:
            new_frames.append(master_full[t:t + 1])
            t += 1
            continue

        chunk = master_full[t:t + CHUNK_SIZE]
        interpolated = resample_master_chunk(chunk, target_len=2 * (CHUNK_SIZE - 1) + 1)
        downsampled = interpolated[::3]
        new_frames.append(downsampled)
        advance = downsampled.shape[0]
        t += CHUNK_SIZE

        while t < T and is_burst[min(t, T - 1)] and t + CHUNK_SIZE <= T:
            chunk = master_full[t:t + CHUNK_SIZE]
            interpolated = resample_master_chunk(chunk, target_len=2 * (CHUNK_SIZE - 1) + 1)
            downsampled = interpolated[::3]
            new_frames.append(downsampled)
            t += CHUNK_SIZE

        if t < T and is_burst[min(t, T - 1)] and t + CHUNK_SIZE > T:
            remaining = T - t
            new_frames.append(master_full[t:T])
            t = T

    return np.concatenate(new_frames, axis=0).astype(np.float32)


def _add_burst_background(ax: plt.Axes, time: np.ndarray, is_burst: np.ndarray) -> None:
    for start, end in _contiguous_segments(is_burst):
        t_start = time[start]
        t_end = time[min(end, len(time) - 1)]
        ax.axvspan(t_start, t_end, color="red", alpha=0.10, zorder=0)


def plot_sft_episode(
    *,
    output_path: Path,
    repo: str,
    ep_idx: int,
    master: np.ndarray,
    hz: int,
    burst_threshold: float,
) -> None:
    vel = position_velocity(master, hz)
    speed = combined_speed(vel)
    is_burst = detect_burst(speed, threshold=burst_threshold)
    total = master.shape[0]
    time = np.arange(total) / hz

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"SFT Velocity (v_mode=3 demo) | {repo} ep {ep_idx}\n"
        f"{total} frames @ {hz}Hz = {total / hz:.1f}s  |  burst threshold = {burst_threshold} m/s",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.25,
                           left=0.05, right=0.98, top=0.91, bottom=0.05,
                           height_ratios=[0.7, 1, 1])

    ax_speed = fig.add_subplot(gs[0, :])
    _add_burst_background(ax_speed, time, is_burst)
    ax_speed.plot(time, speed, color="black", linewidth=0.8)
    ax_speed.axhline(burst_threshold, color="red", linewidth=0.8, linestyle="--",
                     label=f"burst threshold ({burst_threshold} m/s)")
    ax_speed.set_title("Combined position speed (max of L/R)", fontsize=10, fontweight="bold")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("m/s")
    ax_speed.legend(fontsize=8)
    ax_speed.grid(True, alpha=0.3)

    plan = [
        (1, 0, "Left vx", "left", 0), (1, 1, "Left vy", "left", 1), (1, 2, "Left vz", "left", 2),
        (2, 0, "Right vx", "right", 0), (2, 1, "Right vy", "right", 1), (2, 2, "Right vz", "right", 2),
    ]
    for row, col, title, side, comp in plan:
        ax = fig.add_subplot(gs[row, col])
        _add_burst_background(ax, time, is_burst)
        ax.plot(time, vel[side][:, comp], color="#1f77b4", linewidth=0.8, label="SFT (v_mode=3)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("m/s", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if row == 1 and col == 0:
            ax.legend(fontsize=8)

    burst_n = int(is_burst.sum())
    peak = float(speed[is_burst].max()) if burst_n > 0 else 0.0
    fig.text(0.5, 0.005,
             f"Burst frames: {burst_n} | Peak speed: {peak:.4f} m/s",
             ha="center", fontsize=10, style="italic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SFT figure: %s", output_path)


def plot_rollout_episode(
    *,
    output_path: Path,
    repo: str,
    ep_idx: int,
    master: np.ndarray,
    hz: int,
    burst_threshold: float,
) -> None:
    vel_orig = position_velocity(master, hz)
    speed_orig = combined_speed(vel_orig)
    is_burst_orig = detect_burst(speed_orig, threshold=burst_threshold)
    T_orig = master.shape[0]
    time_orig = np.arange(T_orig) / hz

    master_conv = convert_vmode2_to_vmode3(master, is_burst_orig)
    T_conv = master_conv.shape[0]
    time_conv = np.arange(T_conv) / hz
    vel_conv = position_velocity(master_conv, hz)
    speed_conv = combined_speed(vel_conv)
    is_burst_conv = detect_burst(speed_conv, threshold=burst_threshold)

    fig = plt.figure(figsize=(22, 22))
    fig.suptitle(
        f"Rollout v_mode=2->3 Conversion | {repo} ep {ep_idx}\n"
        f"Original: {T_orig} frames ({T_orig / hz:.1f}s)  |  "
        f"Converted: {T_conv} frames ({T_conv / hz:.1f}s)  |  "
        f"burst threshold = {burst_threshold} m/s",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.25,
                           left=0.05, right=0.98, top=0.94, bottom=0.04,
                           height_ratios=[0.6, 1, 1, 0.6, 1, 1])

    ax_sp_orig = fig.add_subplot(gs[0, :])
    _add_burst_background(ax_sp_orig, time_orig, is_burst_orig)
    ax_sp_orig.plot(time_orig, speed_orig, color="red", linewidth=0.8)
    ax_sp_orig.axhline(burst_threshold, color="gray", linewidth=0.8, linestyle="--")
    ax_sp_orig.set_title("ORIGINAL rollout combined speed", fontsize=10, fontweight="bold")
    ax_sp_orig.set_xlabel("Time (s)")
    ax_sp_orig.set_ylabel("m/s")
    ax_sp_orig.grid(True, alpha=0.3)

    plan = [
        (1, 0, "Orig Left vx", "left", 0), (1, 1, "Orig Left vy", "left", 1), (1, 2, "Orig Left vz", "left", 2),
        (2, 0, "Orig Right vx", "right", 0), (2, 1, "Orig Right vy", "right", 1), (2, 2, "Orig Right vz", "right", 2),
    ]
    for row, col, title, side, comp in plan:
        ax = fig.add_subplot(gs[row, col])
        _add_burst_background(ax, time_orig, is_burst_orig)
        ax.plot(time_orig, vel_orig[side][:, comp], color="red", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("m/s", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    ax_sp_conv = fig.add_subplot(gs[3, :])
    _add_burst_background(ax_sp_conv, time_conv, is_burst_conv)
    ax_sp_conv.plot(time_conv, speed_conv, color="green", linewidth=0.8)
    ax_sp_conv.axhline(burst_threshold, color="gray", linewidth=0.8, linestyle="--")
    ax_sp_conv.set_title("CONVERTED (v_mode=2->3) combined speed", fontsize=10, fontweight="bold")
    ax_sp_conv.set_xlabel("Time (s)")
    ax_sp_conv.set_ylabel("m/s")
    ax_sp_conv.grid(True, alpha=0.3)

    plan_conv = [
        (4, 0, "Conv Left vx", "left", 0), (4, 1, "Conv Left vy", "left", 1), (4, 2, "Conv Left vz", "left", 2),
        (5, 0, "Conv Right vx", "right", 0), (5, 1, "Conv Right vy", "right", 1), (5, 2, "Conv Right vz", "right", 2),
    ]
    for row, col, title, side, comp in plan_conv:
        ax = fig.add_subplot(gs[row, col])
        _add_burst_background(ax, time_conv, is_burst_conv)
        ax.plot(time_conv, vel_conv[side][:, comp], color="green", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("m/s", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    orig_burst_n = int(is_burst_orig.sum())
    orig_peak = float(speed_orig[is_burst_orig].max()) if orig_burst_n > 0 else 0.0
    conv_burst_n = int(is_burst_conv.sum())
    conv_peak = float(speed_conv[is_burst_conv].max()) if conv_burst_n > 0 else 0.0
    ratio = conv_peak / orig_peak if orig_peak > 1e-8 else 0.0
    fig.text(
        0.5, 0.005,
        f"Orig burst: {orig_burst_n} frames ({orig_burst_n / hz:.2f}s), peak {orig_peak:.4f} m/s  |  "
        f"Conv burst: {conv_burst_n} frames ({conv_burst_n / hz:.2f}s), peak {conv_peak:.4f} m/s  |  "
        f"Peak ratio: {ratio:.2f}x  |  "
        f"Time saved: {(T_orig - T_conv) / hz:.2f}s",
        ha="center", fontsize=10, style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rollout figure: %s", output_path)
    logger.info(
        "  orig: %d frames, burst=%d, peak=%.4f | conv: %d frames, burst=%d, peak=%.4f | ratio=%.2fx",
        T_orig, orig_burst_n, orig_peak, T_conv, conv_burst_n, conv_peak, ratio,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize v_mode conversion: SFT vs Rollout velocity.")
    p.add_argument("--lerobot-root", type=str, default="/mnt/public/guqiuyi/huggingface/lerobot")
    p.add_argument("--sft-repo", type=str, default="fold_towel_gqy_0420")
    p.add_argument("--rollout-repo", type=str, default="fold_towel_gqy_0412")
    p.add_argument("--sft-episode", type=int, default=None)
    p.add_argument("--rollout-episode", type=int, default=None)
    p.add_argument("--all-sft", action="store_true")
    p.add_argument("--all-rollout", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="velocity_guider/pics/vmode_conversion")
    p.add_argument("--burst-threshold", type=float, default=0.4,
                   help="Absolute speed threshold (m/s) to define burst phase")
    p.add_argument("--burst-dilate-frames", type=int, default=10)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.sft_episode is not None or args.all_sft:
        sft_info = LeRobotDatasetInfo(args.sft_repo, args.lerobot_root)
        sft_eps = sft_info.list_episodes() if args.all_sft else [int(args.sft_episode)]
        if args.max_episodes:
            sft_eps = sft_eps[:args.max_episodes]
        for ep in sft_eps:
            master = sft_info.get_episode_master_actions(ep)
            plot_sft_episode(
                output_path=output_dir / f"sft_{args.sft_repo}_ep_{ep:06d}.png",
                repo=args.sft_repo, ep_idx=ep, master=master, hz=sft_info.fps,
                burst_threshold=args.burst_threshold,
            )

    if args.rollout_episode is not None or args.all_rollout:
        ro_info = LeRobotDatasetInfo(args.rollout_repo, args.lerobot_root)
        ro_eps = ro_info.list_episodes() if args.all_rollout else [int(args.rollout_episode)]
        if args.max_episodes:
            ro_eps = ro_eps[:args.max_episodes]
        for ep in ro_eps:
            master = ro_info.get_episode_master_actions(ep)
            plot_rollout_episode(
                output_path=output_dir / f"rollout_{args.rollout_repo}_ep_{ep:06d}.png",
                repo=args.rollout_repo, ep_idx=ep, master=master, hz=ro_info.fps,
                burst_threshold=args.burst_threshold,
            )


if __name__ == "__main__":
    main()
