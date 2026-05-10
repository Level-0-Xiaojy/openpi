"""Offline evaluation for X2Robot with pose + velocity visualization.

与 ``x2robot_offline_test.py`` 的主要差异：
1. 每个 episode 输出两张 4x4 布局的对比图（GT vs Pred）
   - ``<ep>_pose.jpg``：14 个分量的 pose 曲线
   - ``<ep>_vel.jpg`` ：14 个分量的速度曲线（对时间求梯度）
2. 自动在每张子图上以红色条带标注 GT 速度的最大峰值区间
   （与 ``annotate_action_frequency.py`` 的逻辑一致：取 GT master
    left/right position z 速度的正向峰值，窗口 ``[t-w, t+w]``）。

布局参考 ``examples/x2robot/mp4_json_edit/vis_x2robot_pose.py`` 和
``vis_x2robot_velocity.py``。
"""

import os
import dataclasses
import glob
import json
import logging
from pathlib import Path

import tyro
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    dataset_dir: str = "datasets/x2robot/throw_0113"  # Support multiple datasets separated by comma
    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: str | None = None  # Auto-detect from policy_dir if None (s2s, s2m, sm2m, sm2sm)
    state_history_size: int | None = None  # Auto-load from config if None
    state_future_size: int | None = None
    state_step: int | None = None  # Auto-load from config if None; step size for sampling future states
    move_steps: int = 15
    num_episodes: int | None = None
    split: str | None = None  # "train", "val", or None
    val_ratio: float = 0.1
    split_seed: int = 42
    output_dir: str = "offline_test_results_vel"

    # 可视化相关
    hz: float = 20.0
    auto_highlight: bool = True
    auto_highlight_window: float = 2.0  # 红色标记窗口半宽（秒）


# 14-D master 动作在数组中的索引（与 vis_x2robot_pose.py 中 master_* 字段一致）
SUBPLOT_PLAN: list[list[tuple[str, int]]] = [
    [
        ("Left Position x", 0),
        ("Left Position y", 1),
        ("Left Position z", 2),
        ("Left Gripper",    6),
    ],
    [
        ("Left Rotation roll",  3),
        ("Left Rotation pitch", 4),
        ("Left Rotation yaw",   5),
    ],
    [
        ("Right Position x", 7),
        ("Right Position y", 8),
        ("Right Position z", 9),
        ("Right Gripper",    13),
    ],
    [
        ("Right Rotation roll",  10),
        ("Right Rotation pitch", 11),
        ("Right Rotation yaw",   12),
    ],
]


def build_state(frame_data: dict, policy_mode: str) -> np.ndarray:
    """Build state from frame data."""
    slave = np.concatenate([
        frame_data['follow_left_position'], frame_data['follow_left_rotation'],
        [frame_data['follow_left_gripper']], frame_data['follow_right_position'],
        frame_data['follow_right_rotation'], [frame_data['follow_right_gripper']]
    ]).astype(np.float32)

    if policy_mode in ["s2s", "s2m"]:
        return slave

    master = np.concatenate([
        frame_data['master_left_position'], frame_data['master_left_rotation'],
        [frame_data['master_left_gripper']], frame_data['master_right_position'],
        frame_data['master_right_rotation'], [frame_data['master_right_gripper']]
    ]).astype(np.float32)

    return np.concatenate([slave, master])


def run_inference(policy, episode_path: str, args: Args):
    """Run inference on episode and return predictions and GT."""
    episode_name = os.path.basename(episode_path)
    with open(os.path.join(episode_path, f"{episode_name}.json"), 'r') as f:
        episode_data = json.load(f)

    frames = episode_data['data']
    total_frames = len(frames)
    videos = {k: cv2.VideoCapture(os.path.join(episode_path, f"{k}Img.mp4"))
              for k in ['left', 'face', 'right']}

    all_preds, all_gts = [], []
    current_idx = 0

    while current_idx < total_frames:
        images = {}
        for k, cap in videos.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            images[f"{k}_wrist_view" if k != "face" else "face_view"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        indices = [max(0, min(current_idx + i * args.state_step, total_frames - 1))
                   for i in range(-args.state_history_size, args.state_future_size + 1)]
        state_seq = np.array([build_state(frames[i], args.policy_mode) for i in indices], dtype=np.float32)

        obs = {'images': images, 'prompt': '', 'state': state_seq}
        action_pred = policy.infer(obs)['actions']

        if args.policy_mode == "sm2sm":
            action_pred = action_pred[:, 14:28]  # Extract master part
        else:
            action_pred = action_pred[:, :14]

        action_pred = action_pred[:args.move_steps]

        gt_length = min(args.move_steps, total_frames - current_idx - 1)
        if gt_length > 0:
            gt_actions = []
            for i in range(gt_length):
                idx = min(current_idx + 1 + i, total_frames - 1)
                frame = frames[idx]
                gt = np.concatenate([
                    frame['master_left_position'], frame['master_left_rotation'],
                    [frame['master_left_gripper']], frame['master_right_position'],
                    frame['master_right_rotation'], [frame['master_right_gripper']]
                ]).astype(np.float32)
                gt_actions.append(gt)

            all_preds.append(action_pred[:gt_length])
            all_gts.append(np.array(gt_actions))

        current_idx += args.move_steps

    for cap in videos.values():
        cap.release()

    return (np.concatenate(all_preds) if all_preds else np.zeros((0, 14)),
            np.concatenate(all_gts) if all_gts else np.zeros((0, 14)))


def compute_velocity_14d(actions: np.ndarray, hz: float) -> np.ndarray:
    """(N, 14) master actions -> (N, 14) velocity via time gradient."""
    if len(actions) < 2:
        return np.zeros_like(actions)
    dt = 1.0 / hz
    return np.gradient(actions, dt, axis=0)


def find_max_vel_window(gt: np.ndarray, hz: float,
                        window: float = 2.0) -> tuple[float, float] | None:
    """在 GT master left/right position z 的速度上找正向峰值，返回 [t-w, t+w]。

    与 ``examples/x2robot/mp4_json_edit/annotate_action_frequency.py`` 一致：
    以双臂 z 方向速度正向峰值作为"快速动作"锚点。
    """
    n = len(gt)
    if n < 2:
        return None
    dt = 1.0 / hz
    lvz = np.gradient(gt[:, 2], dt)   # left  pos z
    rvz = np.gradient(gt[:, 9], dt)   # right pos z (7+2)
    combined = np.concatenate([lvz, rvz])
    idx = int(np.argmax(combined))
    frame_idx = idx if idx < len(lvz) else idx - len(lvz)
    t = frame_idx / hz
    return max(0.0, t - window), t + window


def plot_panel(pred: np.ndarray, gt: np.ndarray, hz: float,
               episode_name: str, output_path: str, kind: str,
               highlight: tuple[float, float] | None = None) -> None:
    """4x4 layout: 每个分量一张子图，画 GT(橙虚) vs Pred(蓝实)；可选红色标注区间。

    Args:
        pred, gt: shape (N, 14) —— 若 kind="Velocity" 则传入速度。
        kind: "Pose" 或 "Velocity"，仅用于标题。
    """
    n = min(len(pred), len(gt))
    if n == 0:
        return
    pred, gt = pred[:n], gt[:n]
    time = np.arange(n) / hz

    gt_color = "#ff7f0e"
    pred_color = "#1f77b4"

    fig = plt.figure(figsize=(28, 16))
    title = (f"{kind}  |  Episode: {episode_name}\n"
             f"{n} frames @ {hz:.0f}Hz = {n / hz:.1f}s")
    if highlight is not None:
        title += f"  |  highlight=[{highlight[0]:.2f}s, {highlight[1]:.2f}s]"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.04, right=0.98, top=0.92, bottom=0.05)

    first = True
    for row_idx, row_plots in enumerate(SUBPLOT_PLAN):
        for col_idx, (subtitle, idx) in enumerate(row_plots):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if highlight is not None:
                ax.axvspan(highlight[0], highlight[1],
                           color="red", alpha=0.15, zorder=0)
            ax.plot(time, gt[:, idx],   color=gt_color,   linewidth=0.8,
                    linestyle="--", alpha=0.9, label="GT")
            ax.plot(time, pred[:, idx], color=pred_color, linewidth=0.8,
                    alpha=0.9, label="Pred")
            ax.set_title(f"{kind} {subtitle}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if first:
                ax.legend(fontsize=8, loc="upper right")
                first = False

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(args: Args):
    if args.policy_mode is None:
        for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
            if mode in args.policy_dir.lower():
                args.policy_mode = mode
                logging.info(f"Auto-detected policy_mode from path: {args.policy_mode}")
                break
        if args.policy_mode is None:
            raise ValueError(f"Could not detect policy_mode from path: {args.policy_dir}. Please specify --policy-mode")

    cfg = _config.get_config(args.policy_config)
    if args.state_history_size is None:
        args.state_history_size = getattr(cfg.data, 'state_history_size', 0)
        logging.info(f"Using state_history_size from config: {args.state_history_size}")
    if args.state_future_size is None:
        args.state_future_size = getattr(cfg.data, 'state_future_size', 0)
        logging.info(f"Using state_future_size from config: {args.state_future_size}")
    if args.state_step is None:
        args.state_step = getattr(cfg.data, 'state_step', 1)
        logging.info(f"Using state_step from config: {args.state_step}")

    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)

    dataset_dirs = [d.strip() for d in args.dataset_dir.split(',')]
    all_episodes = []

    for dataset_dir in dataset_dirs:
        dataset_episodes = sorted([p for p in glob.glob(f'{dataset_dir}/*')
                                   if os.path.isdir(p) and glob.glob(f'{p}/*.mp4')])

        if args.split:
            rng = np.random.RandomState(args.split_seed)
            indices = np.arange(len(dataset_episodes))
            rng.shuffle(indices)
            val_size = int(len(dataset_episodes) * args.val_ratio)
            indices = indices[:val_size] if args.split == "val" else indices[val_size:]
            dataset_episodes = [dataset_episodes[i] for i in sorted(indices)]
            logging.info(f"Dataset {dataset_dir}: {len(dataset_episodes)} episodes in {args.split} split")
        else:
            logging.info(f"Dataset {dataset_dir}: {len(dataset_episodes)} episodes")

        all_episodes.extend(dataset_episodes)

    episodes = all_episodes
    logging.info(f"Total: {len(episodes)} episodes across {len(dataset_dirs)} dataset(s)")

    if args.num_episodes:
        episodes = episodes[:args.num_episodes]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_path in tqdm(episodes, desc="Processing"):
        ep_name = os.path.basename(ep_path)
        pred, gt = run_inference(policy, ep_path, args)
        if len(pred) == 0:
            continue

        highlight: tuple[float, float] | None = None
        if args.auto_highlight:
            highlight = find_max_vel_window(gt, hz=args.hz,
                                            window=args.auto_highlight_window)
            if highlight is not None:
                logging.info(f"{ep_name} highlight=[{highlight[0]:.2f}s, {highlight[1]:.2f}s]")

        pose_path = str(output_dir / f"{ep_name}_pose.jpg")
        plot_panel(pred, gt, args.hz, ep_name, pose_path,
                   kind="Pose", highlight=highlight)

        pred_vel = compute_velocity_14d(pred, args.hz)
        gt_vel = compute_velocity_14d(gt, args.hz)
        vel_path = str(output_dir / f"{ep_name}_vel.jpg")
        plot_panel(pred_vel, gt_vel, args.hz, ep_name, vel_path,
                   kind="Velocity", highlight=highlight)

        logging.info(f"Saved {ep_name} -> pose.jpg + vel.jpg")

    logging.info(f"Done! Results in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
