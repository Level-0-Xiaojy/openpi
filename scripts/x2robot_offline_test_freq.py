"""Offline evaluation for X2Robot (v6/29-D action with action_frequency).

Based on x2robot_offline_test.py. Expects policies trained on data produced by
examples/x2robot/convert_x2robot_data_to_lerobot_v6.py, where actions are 29-D:
    [master_left(7) + master_right(7)] for sm2sm uses indices [14:28];
    the last dim [28] is action_frequency (Hz).

Differences vs the original script:
- Splits policy output into (master_14d, freq_1d) and plots both.
- Reads per-frame ground-truth `action_frequency` from JSON (default 20Hz).
- Prints per-episode MAE for master and freq.
- Sanity-checks video vs. JSON frame counts.
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
from tqdm import tqdm

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


DEFAULT_ACTION_FREQUENCY = 20.0
FREQ_ACTION_INDEX = 28  # index of action_frequency in 29-D action vector


@dataclasses.dataclass
class Args:
    dataset_dir: str = "datasets/x2robot/fold_towel_gqy_0420"  # comma-separated list allowed
    policy_config: str = "fold_towel_sm2sm_freq"
    policy_dir: str = "checkpoints/fold_towel_sm2sm_freq/fold_towel_gqy031703180420_pi0base_sm2sm_freq_h3f2_a20_dm10dh50df50po20/29999"
    policy_mode: str | None = None  # Auto-detect from policy_dir if None (s2s, s2m, sm2m, sm2sm)
    state_history_size: int | None = None  # Auto-load from config if None
    state_future_size: int | None = None
    state_step: int | None = None  # Auto-load from config if None
    move_steps: int = 15
    num_episodes: int | None = None
    split: str | None = None  # "train", "val", or None
    val_ratio: float = 0.1
    split_seed: int = 42
    output_dir: str = "offline_test_results_freq"


def build_state(frame_data: dict, policy_mode: str) -> np.ndarray:
    """Build 14-D (s*) or 28-D (sm*) state from a frame. action_frequency is NOT in state."""
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


def build_gt_master(frame: dict) -> np.ndarray:
    return np.concatenate([
        frame['master_left_position'], frame['master_left_rotation'],
        [frame['master_left_gripper']], frame['master_right_position'],
        frame['master_right_rotation'], [frame['master_right_gripper']]
    ]).astype(np.float32)


def run_inference(policy, episode_path: str, args: Args):
    """Run inference on an episode.

    Returns:
        pred_master: (N, 14) predicted master actions
        gt_master:   (N, 14) ground-truth master actions
        pred_freq:   (N,) predicted action_frequency, or None if policy output <29-D
        gt_freq:     (N,) ground-truth action_frequency from JSON
    """
    episode_name = os.path.basename(episode_path)
    with open(os.path.join(episode_path, f"{episode_name}.json"), 'r') as f:
        episode_data = json.load(f)

    frames = episode_data['data']
    total_frames = len(frames)
    videos = {k: cv2.VideoCapture(os.path.join(episode_path, f"{k}Img.mp4"))
              for k in ['left', 'face', 'right']}

    for k, cap in videos.items():
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nf != total_frames:
            logging.warning(f"{episode_name} {k}: video frames={nf} vs json frames={total_frames}")

    pred_master_all, gt_master_all = [], []
    pred_freq_all, gt_freq_all = [], []
    current_idx = 0

    while current_idx < total_frames:
        images = {}
        for k, cap in videos.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"{episode_name}: failed to read {k} frame {current_idx}")
                break
            images[f"{k}_wrist_view" if k != "face" else "face_view"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if len(images) != 3:
            break

        indices = [max(0, min(current_idx + i * args.state_step, total_frames - 1))
                   for i in range(-args.state_history_size, args.state_future_size + 1)]
        state_seq = np.array([build_state(frames[i], args.policy_mode) for i in indices], dtype=np.float32)

        obs = {'images': images, 'prompt': '', 'state': state_seq}
        action_pred = policy.infer(obs)['actions']  # (action_horizon, action_dim)

        has_freq = action_pred.shape[-1] > FREQ_ACTION_INDEX
        if args.policy_mode == "sm2sm":
            pred_master = action_pred[:, 14:28]
        else:
            pred_master = action_pred[:, :14]
        pred_freq = action_pred[:, FREQ_ACTION_INDEX] if has_freq else None

        pred_master = pred_master[:args.move_steps]
        if pred_freq is not None:
            pred_freq = pred_freq[:args.move_steps]

        gt_length = min(args.move_steps, total_frames - current_idx - 1)
        if gt_length > 0:
            gt_master_chunk = []
            gt_freq_chunk = []
            for i in range(gt_length):
                idx = min(current_idx + 1 + i, total_frames - 1)
                f = frames[idx]
                gt_master_chunk.append(build_gt_master(f))
                gt_freq_chunk.append(float(f.get('action_frequency', DEFAULT_ACTION_FREQUENCY)))

            pred_master_all.append(pred_master[:gt_length])
            gt_master_all.append(np.array(gt_master_chunk, dtype=np.float32))
            if pred_freq is not None:
                pred_freq_all.append(pred_freq[:gt_length])
                gt_freq_all.append(np.array(gt_freq_chunk, dtype=np.float32))

        current_idx += args.move_steps

    for cap in videos.values():
        cap.release()

    pred_master_arr = np.concatenate(pred_master_all) if pred_master_all else np.zeros((0, 14), dtype=np.float32)
    gt_master_arr = np.concatenate(gt_master_all) if gt_master_all else np.zeros((0, 14), dtype=np.float32)
    if pred_freq_all:
        pred_freq_arr = np.concatenate(pred_freq_all)
        gt_freq_arr = np.concatenate(gt_freq_all)
    else:
        pred_freq_arr, gt_freq_arr = None, None

    return pred_master_arr, gt_master_arr, pred_freq_arr, gt_freq_arr


def plot_results(pred: np.ndarray, gt: np.ndarray,
                 pred_freq: np.ndarray | None, gt_freq: np.ndarray | None,
                 name: str, output_path: str):
    """Plot master (6 subplots) + optional action_frequency (1 subplot)."""
    has_freq = pred_freq is not None and gt_freq is not None
    n_rows = 3 if has_freq else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    fig.suptitle(f'Episode: {name}', fontsize=16)

    configs = [
        (0, 0, [0, 1, 2], 'Left Arm XYZ', 'Position'),
        (0, 1, [3, 4, 5], 'Left Arm RPY', 'Rotation'),
        (0, 2, [6], 'Left Gripper', 'Gripper'),
        (1, 0, [7, 8, 9], 'Right Arm XYZ', 'Position'),
        (1, 1, [10, 11, 12], 'Right Arm RPY', 'Rotation'),
        (1, 2, [13], 'Right Gripper', 'Gripper'),
    ]
    labels = {3: ['X', 'Y', 'Z'], 1: ['']}

    for row, col, indices, title, ylabel in configs:
        ax = axes[row, col]
        if title == 'Left Arm RPY' or title == 'Right Arm RPY':
            lbls = ['R', 'P', 'Y']
        else:
            lbls = labels.get(len(indices), ['X', 'Y', 'Z'])
        for i, idx in enumerate(indices):
            ax.plot(gt[:, idx], '--', alpha=0.7, label=f'GT {lbls[i]}')
            ax.plot(pred[:, idx], alpha=0.7, label=f'Pred {lbls[i]}')
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    if has_freq:
        ax = axes[2, 0]
        ax.plot(gt_freq, '--', alpha=0.8, label='GT freq')
        ax.plot(pred_freq, alpha=0.8, label='Pred freq')
        ax.set_title('Action Frequency')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Hz')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        err = pred_freq - gt_freq
        ax.plot(err, alpha=0.8, color='tab:red')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'Freq error (MAE={np.abs(err).mean():.3f} Hz)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Pred - GT (Hz)')
        ax.grid(True, alpha=0.3)

        axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    agg_master_mae, agg_freq_mae = [], []

    for ep_path in tqdm(episodes, desc="Processing"):
        ep_name = os.path.basename(ep_path)
        pred, gt, pred_freq, gt_freq = run_inference(policy, ep_path, args)
        if len(pred) == 0:
            logging.warning(f"{ep_name}: no predictions produced, skipping.")
            continue

        master_mae = float(np.abs(pred - gt).mean())
        agg_master_mae.append(master_mae)

        if pred_freq is not None:
            freq_mae = float(np.abs(pred_freq - gt_freq).mean())
            agg_freq_mae.append(freq_mae)
            logging.info(f"{ep_name}  master MAE={master_mae:.4f}  freq MAE={freq_mae:.3f} Hz  "
                         f"(pred freq mean={pred_freq.mean():.2f}, gt freq mean={gt_freq.mean():.2f})")
        else:
            logging.info(f"{ep_name}  master MAE={master_mae:.4f}  (no freq output)")

        plot_results(pred, gt, pred_freq, gt_freq, ep_name, str(output_dir / f"{ep_name}.jpg"))

    if agg_master_mae:
        logging.info(f"Overall master MAE: {np.mean(agg_master_mae):.4f}  "
                     f"(over {len(agg_master_mae)} episodes)")
    if agg_freq_mae:
        logging.info(f"Overall freq   MAE: {np.mean(agg_freq_mae):.3f} Hz  "
                     f"(over {len(agg_freq_mae)} episodes)")

    logging.info(f"Done! Results in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
