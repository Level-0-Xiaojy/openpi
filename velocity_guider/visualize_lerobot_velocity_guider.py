#!/usr/bin/env python3
"""Visualize Velocity Guider predictions on a LeRobot dataset.

This script runs the pi0 vision tower online to extract ``obs_feat`` for each
frame, feeds the demo master action chunk ``master[t:t+20]`` into a trained
Velocity Guider, and plots:

- predicted v_mode per frame
- class probabilities for v_mode=3/2/1
- left/right position x/y/z velocities

Example:
    cd /home/guqiuyi/workspace/openpi
    uv run python velocity_guider/visualize_lerobot_velocity_guider.py \
        --lerobot-root /mnt/public/guqiuyi/huggingface/lerobot \
        --repo fold_towel_gqy_0412 \
        --checkpoint /mnt/public/guqiuyi/checkpoints/velocity_guider/v1/best.pt \
        --pi0-checkpoint-path /mnt/public/models/pytorch_models/pi0_base_pytorch \
        --episode 0 \
        --output-dir /mnt/public/guqiuyi/velocity_guider_vis/fold_towel_gqy_0412
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from infer import VelocityGuiderInfer
from velocity_guider.data.lerobot_loader import LeRobotDatasetInfo, iter_episode_image_batches
from velocity_guider.data.vision_encoder import VisionEncoder

logger = logging.getLogger("velocity_guider.visualize")


LABEL_NAMES = {
    0: "label 0 / v_mode=3 (smooth)",
    1: "label 1 / v_mode=2",
    2: "label 2 / v_mode=1 (fast)",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def compute_obs_feats(
    encoder: VisionEncoder,
    info: LeRobotDatasetInfo,
    ep_idx: int,
    *,
    batch_size: int,
    expected_num_frames: int,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for _indices, imgs in tqdm(
        iter_episode_image_batches(info, ep_idx, batch_size=batch_size, expected_num_frames=expected_num_frames),
        desc=f"encode images ep {ep_idx}",
        leave=False,
    ):
        feats.append(encoder.encode(imgs))
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


def build_demo_chunks(master_actions: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build v_mode inference chunks from raw demo master actions.

    Returns:
        ``(frame_idx [N], chunks [N, K, 14])`` where ``chunks[i]`` is
        ``master_actions[t:t+K]``. The last ``K-1`` frames are skipped because
        they do not have a full action chunk.
    """

    total_frames = master_actions.shape[0]
    num_valid = total_frames - chunk_size + 1
    if num_valid <= 0:
        raise ValueError(f"Episode too short: {total_frames} frames < chunk_size={chunk_size}")
    frame_idx = np.arange(num_valid, dtype=np.int64)
    chunks = np.stack([master_actions[t:t + chunk_size] for t in frame_idx], axis=0)
    return frame_idx, chunks.astype(np.float32, copy=False)


@torch.no_grad()
def predict_episode(
    guider: VelocityGuiderInfer,
    obs_feats: np.ndarray,
    chunks: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    v_modes: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for start in tqdm(range(0, len(chunks), batch_size), desc="guider inference", leave=False):
        end = min(start + batch_size, len(chunks))
        out = guider.predict(obs_feats[start:end], chunks[start:end])
        labels.append(out["label"])
        v_modes.append(out["v_mode"])
        probs.append(out["prob"])

    return (
        np.concatenate(labels, axis=0),
        np.concatenate(v_modes, axis=0),
        np.concatenate(probs, axis=0),
    )


def compute_position_velocity(master_actions: np.ndarray, hz: int) -> dict[str, np.ndarray]:
    dt = 1.0 / hz
    return {
        "left_position": np.gradient(master_actions[:, 0:3], dt, axis=0),
        "right_position": np.gradient(master_actions[:, 7:10], dt, axis=0),
    }


def fill_full_length_predictions(
    total_frames: int,
    frame_idx: np.ndarray,
    labels: np.ndarray,
    v_modes: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_labels = np.full((total_frames,), -1, dtype=np.int32)
    full_v_modes = np.full((total_frames,), np.nan, dtype=np.float32)
    full_probs = np.full((total_frames, probs.shape[1]), np.nan, dtype=np.float32)

    full_labels[frame_idx] = labels.astype(np.int32)
    full_v_modes[frame_idx] = v_modes.astype(np.float32)
    full_probs[frame_idx] = probs.astype(np.float32)
    return full_labels, full_v_modes, full_probs


def add_vmode_background(ax: plt.Axes, time: np.ndarray, v_modes: np.ndarray) -> None:
    """Lightly shade contiguous predicted v_mode segments."""

    colors = {3: "#1f77b4", 2: "#ff7f0e", 1: "#d62728"}
    valid = np.isfinite(v_modes)
    if not valid.any():
        return

    i = int(np.argmax(valid))
    while i < len(v_modes):
        if not np.isfinite(v_modes[i]):
            i += 1
            continue
        mode = int(v_modes[i])
        j = i + 1
        while j < len(v_modes) and np.isfinite(v_modes[j]) and int(v_modes[j]) == mode:
            j += 1
        ax.axvspan(time[i], time[min(j, len(time) - 1)], color=colors.get(mode, "gray"), alpha=0.08, zorder=0)
        i = j


def plot_episode(
    *,
    output_path: Path,
    repo: str,
    episode_idx: int,
    hz: int,
    v_modes: np.ndarray,
    probs: np.ndarray,
    pos_vel: dict[str, np.ndarray],
    highlight_start: float | None = None,
    highlight_end: float | None = None,
) -> None:
    total_frames = len(v_modes)
    time = np.arange(total_frames) / hz

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"Velocity Guider Predictions | {repo} ep {episode_idx}\n"
        f"{total_frames} frames @ {hz}Hz = {total_frames / hz:.1f}s",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.38,
        wspace=0.25,
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.05,
        height_ratios=[0.8, 0.9, 1.0, 1.0],
    )

    def decorate_axis(ax: plt.Axes) -> None:
        if highlight_start is not None and highlight_end is not None:
            ax.axvspan(highlight_start, highlight_end, color="red", alpha=0.12, zorder=0)
        add_vmode_background(ax, time, v_modes)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    ax_mode = fig.add_subplot(gs[0, :])
    decorate_axis(ax_mode)
    ax_mode.step(time, v_modes, where="post", color="black", linewidth=1.5, label="pred v_mode")
    ax_mode.set_yticks([1, 2, 3])
    ax_mode.set_ylim(0.5, 3.5)
    ax_mode.set_title("Predicted v_mode per frame", fontsize=11, fontweight="bold")
    ax_mode.set_xlabel("Time (s)")
    ax_mode.legend(loc="upper right", fontsize=8)

    ax_prob = fig.add_subplot(gs[1, :])
    decorate_axis(ax_prob)
    ax_prob.plot(time, probs[:, 0], label=LABEL_NAMES[0], linewidth=1.0, color="#1f77b4")
    ax_prob.plot(time, probs[:, 1], label=LABEL_NAMES[1], linewidth=1.0, color="#ff7f0e")
    ax_prob.plot(time, probs[:, 2], label=LABEL_NAMES[2], linewidth=1.0, color="#d62728")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_title("Velocity Guider probabilities", fontsize=11, fontweight="bold")
    ax_prob.set_xlabel("Time (s)")
    ax_prob.legend(loc="upper right", fontsize=8)

    subplot_plan = [
        (2, 0, "Left position vx", "left_position", 0),
        (2, 1, "Left position vy", "left_position", 1),
        (2, 2, "Left position vz", "left_position", 2),
        (3, 0, "Right position vx", "right_position", 0),
        (3, 1, "Right position vy", "right_position", 1),
        (3, 2, "Right position vz", "right_position", 2),
    ]
    for row, col, title, key, comp in subplot_plan:
        ax = fig.add_subplot(gs[row, col])
        decorate_axis(ax)
        ax.plot(time, pos_vel[key][:, comp], color="#2ca02c", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("m/s or unit/s", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved visualization: %s", output_path)


def save_predictions_npz(
    output_path: Path,
    *,
    labels: np.ndarray,
    v_modes: np.ndarray,
    probs: np.ndarray,
    pos_vel: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        label=labels,
        v_mode=v_modes,
        prob=probs,
        left_position_velocity=pos_vel["left_position"].astype(np.float32),
        right_position_velocity=pos_vel["right_position"].astype(np.float32),
    )
    logger.info("Saved prediction arrays: %s", output_path)


def process_episode(
    *,
    info: LeRobotDatasetInfo,
    encoder: VisionEncoder,
    guider: VelocityGuiderInfer,
    ep_idx: int,
    output_dir: Path,
    image_batch_size: int,
    infer_batch_size: int,
    highlight_start: float | None,
    highlight_end: float | None,
    save_npz: bool,
) -> None:
    logger.info("Processing %s episode %d", info.repo_name, ep_idx)
    master = info.get_episode_master_actions(ep_idx)
    total_frames = master.shape[0]
    obs_feats = compute_obs_feats(
        encoder,
        info,
        ep_idx,
        batch_size=image_batch_size,
        expected_num_frames=total_frames,
    )
    if obs_feats.shape[0] != total_frames:
        aligned = min(obs_feats.shape[0], total_frames)
        logger.warning(
            "Frame mismatch for ep %d: actions=%d, obs_feats=%d; truncating to %d",
            ep_idx,
            total_frames,
            obs_feats.shape[0],
            aligned,
        )
        master = master[:aligned]
        obs_feats = obs_feats[:aligned]
        total_frames = aligned

    frame_idx, chunks = build_demo_chunks(master, guider.cfg.chunk_size)
    labels, v_modes, probs = predict_episode(
        guider,
        obs_feats[frame_idx],
        chunks,
        batch_size=infer_batch_size,
    )
    full_labels, full_v_modes, full_probs = fill_full_length_predictions(
        total_frames,
        frame_idx,
        labels,
        v_modes,
        probs,
    )
    pos_vel = compute_position_velocity(master, info.fps)

    output_png = output_dir / f"velocity_guider_{info.repo_name}_ep_{ep_idx:06d}.png"
    plot_episode(
        output_path=output_png,
        repo=info.repo_name,
        episode_idx=ep_idx,
        hz=info.fps,
        v_modes=full_v_modes,
        probs=full_probs,
        pos_vel=pos_vel,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
    )
    if save_npz:
        save_predictions_npz(
            output_dir / f"velocity_guider_{info.repo_name}_ep_{ep_idx:06d}.npz",
            labels=full_labels,
            v_modes=full_v_modes,
            probs=full_probs,
            pos_vel=pos_vel,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Velocity Guider predictions on LeRobot episodes.")
    parser.add_argument("--lerobot-root", type=str, default="/mnt/public/guqiuyi/huggingface/lerobot")
    parser.add_argument("--repo", type=str, default="fold_towel_gqy_0412")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Velocity Guider best.pt")
    parser.add_argument(
        "--pi0-checkpoint-path",
        type=str,
        default="/mnt/public/models/pytorch_models/pi0_base_pytorch",
        help="pi0 PyTorch checkpoint path used by the vision tower",
    )
    parser.add_argument("--episode", type=int, default=None, help="Single episode index to visualize")
    parser.add_argument("--all", action="store_true", help="Visualize all episodes in the repo")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap when using --all")
    parser.add_argument("--output-dir", type=str, default="/mnt/public/guqiuyi/velocity_guider_vis")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vision-dtype", type=str, default="float32")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    parser.add_argument("--highlight-start", type=float, default=None)
    parser.add_argument("--highlight-end", type=float, default=None)
    parser.add_argument("--no-save-npz", action="store_true", help="Only save png, not prediction arrays")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.episode is None and not args.all:
        raise ValueError("Please specify --episode <idx> or --all.")

    info = LeRobotDatasetInfo(args.repo, args.lerobot_root)
    episodes = info.list_episodes() if args.all else [int(args.episode)]
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    output_dir = Path(args.output_dir) / args.repo
    logger.info("Loading vision encoder from %s", args.pi0_checkpoint_path)
    encoder = VisionEncoder(
        checkpoint_path=args.pi0_checkpoint_path,
        action_horizon=20,
        device=args.device,
        dtype=args.vision_dtype,
    )
    logger.info("Loading Velocity Guider from %s", args.checkpoint)
    guider = VelocityGuiderInfer(args.checkpoint, device=args.device)

    for ep_idx in episodes:
        process_episode(
            info=info,
            encoder=encoder,
            guider=guider,
            ep_idx=ep_idx,
            output_dir=output_dir,
            image_batch_size=args.image_batch_size,
            infer_batch_size=args.infer_batch_size,
            highlight_start=args.highlight_start,
            highlight_end=args.highlight_end,
            save_npz=not args.no_save_npz,
        )


if __name__ == "__main__":
    main()
