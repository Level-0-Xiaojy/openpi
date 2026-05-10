"""Offline evaluation for X2Robot: max-velocity distribution across episodes.

组合 ``x2robot_offline_test.py`` 的推理流程和
``examples/x2robot/mp4_json_edit/vis_max_velocity_distribution.py`` 的直方图统计。

对 ``dataset-dir`` 下所有满足 split/select 条件的 episode：
1. 用训练好的 policy 做一遍 offline 推理，得到逐帧 (N, 14) 的 pred / gt master
   动作序列。
2. 按 ``--metric`` 计算每个 episode 的最大速度（peak_vz 或 max_speed），
   对 pred 和 gt 各得到一个值。
3. 输出：
   - ``max_vel_hist_gt.png``   GT 最大速度分布
   - ``max_vel_hist_pred.png`` Pred 最大速度分布
   - ``max_vel_hist.png``      GT vs Pred 叠加对比图
   - ``max_vel_hist.csv``      每个 episode 的 pred/gt 最大速度
"""

import os
import csv
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


# ---------- metric 定义，和 vis_max_velocity_distribution.py 保持一致 ----------

def _max_velocity_peakz(actions: np.ndarray, hz: float) -> float:
    """master left/right position z 方向速度的正向峰值。"""
    if len(actions) < 2:
        return 0.0
    dt = 1.0 / hz
    lvz = np.gradient(actions[:, 2], dt)   # left pos z
    rvz = np.gradient(actions[:, 9], dt)   # right pos z (7+2)
    return float(max(lvz.max(), rvz.max()))


def _max_velocity_speed(actions: np.ndarray, hz: float) -> float:
    """双臂 position xyz 合成速度模的峰值。"""
    if len(actions) < 2:
        return 0.0
    dt = 1.0 / hz
    lv = np.gradient(actions[:, 0:3], dt, axis=0)
    rv = np.gradient(actions[:, 7:10], dt, axis=0)
    lspeed = np.linalg.norm(lv, axis=1)
    rspeed = np.linalg.norm(rv, axis=1)
    return float(max(lspeed.max(), rspeed.max()))


METRICS = {
    "peak_vz":   _max_velocity_peakz,
    "max_speed": _max_velocity_speed,
}


# ---------- Args ----------

@dataclasses.dataclass
class Args:
    dataset_dir: str = "datasets/x2robot/throw_0113"  # 逗号分隔可多个
    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: str | None = None
    state_history_size: int | None = None
    state_future_size: int | None = None
    state_step: int | None = None
    move_steps: int = 15
    num_episodes: int | None = None
    split: str | None = None
    val_ratio: float = 0.1
    split_seed: int = 42
    output_dir: str = "offline_test_results_vel_dist"

    # 统计参数
    hz: float = 20.0
    metric: str = "peak_vz"   # "peak_vz" 或 "max_speed"
    bin_width: float = 0.1
    # 只统计名字以 success/fail 结尾的 episode；all 不过滤
    select: str = "all"


# ---------- 推理相关：和 x2robot_offline_test.py 对齐 ----------

def build_state(frame_data: dict, policy_mode: str) -> np.ndarray:
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
            action_pred = action_pred[:, 14:28]
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


# ---------- select 过滤 ----------

def _match_select(ep_name: str, select: str) -> bool:
    if select == "all":
        return True
    name_lower = ep_name.lower()
    if select == "success":
        return name_lower.endswith("success")
    if select == "fail":
        return name_lower.endswith("fail")
    return True


# ---------- 直方图绘制 ----------

def plot_single_histogram(values: list[float], bin_width: float, output_path: str,
                          metric: str, label: str, select: str) -> None:
    arr = np.array(values)
    vmax = float(np.ceil(max(arr.max(), bin_width) / bin_width) * bin_width)
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
        f"{label} max velocity distribution "
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
    logging.info(f"Saved {label} histogram -> {output_path}")


def plot_compare_histogram(pred_vals: list[float], gt_vals: list[float],
                           bin_width: float, output_path: str,
                           metric: str, select: str) -> None:
    pred_arr = np.array(pred_vals)
    gt_arr = np.array(gt_vals)
    vmax = float(np.ceil(max(pred_arr.max(), gt_arr.max(), bin_width) / bin_width) * bin_width)
    bins = np.arange(0.0, vmax + bin_width, bin_width)
    centers = (bins[:-1] + bins[1:]) / 2.0
    gt_counts,   _ = np.histogram(gt_arr,   bins=bins)
    pred_counts, _ = np.histogram(pred_arr, bins=bins)

    fig, ax = plt.subplots(figsize=(13, 6))
    w = bin_width * 0.42
    ax.bar(centers - w / 2, gt_counts,   width=w, color="#f28e2b",
           edgecolor="black", alpha=0.85, label=f"GT (mean={gt_arr.mean():.3f})")
    ax.bar(centers + w / 2, pred_counts, width=w, color="#4c78a8",
           edgecolor="black", alpha=0.85, label=f"Pred (mean={pred_arr.mean():.3f})")

    ax.set_xlabel(f"Max velocity ({metric}) [m/s]")
    ax.set_ylabel("# episodes")
    ax.set_title(
        f"Max velocity distribution: GT vs Pred "
        f"(n={len(gt_arr)}, bin={bin_width}, metric={metric}, select={select})"
    )
    ax.set_xticks(bins)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved compare histogram -> {output_path}")


def save_csv(names: list[str], pred_vals: list[float], gt_vals: list[float],
             csv_path: str) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "gt_max_vel", "pred_max_vel", "diff(pred-gt)"])
        order = np.argsort(-np.array(gt_vals))
        for i in order:
            writer.writerow([names[i], f"{gt_vals[i]:.6f}",
                             f"{pred_vals[i]:.6f}",
                             f"{pred_vals[i] - gt_vals[i]:.6f}"])
    logging.info(f"Saved per-episode CSV -> {csv_path}")


# ---------- main ----------

def main(args: Args):
    if args.metric not in METRICS:
        raise ValueError(f"--metric must be one of {list(METRICS)}, got {args.metric}")
    metric_fn = METRICS[args.metric]

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
    if args.state_future_size is None:
        args.state_future_size = getattr(cfg.data, 'state_future_size', 0)
    if args.state_step is None:
        args.state_step = getattr(cfg.data, 'state_step', 1)
    logging.info(
        f"state_history_size={args.state_history_size}, "
        f"state_future_size={args.state_future_size}, state_step={args.state_step}"
    )

    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)

    dataset_dirs = [d.strip() for d in args.dataset_dir.split(',')]
    all_episodes: list[str] = []
    for dataset_dir in dataset_dirs:
        dataset_episodes = sorted([p for p in glob.glob(f'{dataset_dir}/*')
                                   if os.path.isdir(p) and glob.glob(f'{p}/*.mp4')])
        dataset_episodes = [p for p in dataset_episodes
                            if _match_select(os.path.basename(p), args.select)]

        if args.split:
            rng = np.random.RandomState(args.split_seed)
            indices = np.arange(len(dataset_episodes))
            rng.shuffle(indices)
            val_size = int(len(dataset_episodes) * args.val_ratio)
            indices = indices[:val_size] if args.split == "val" else indices[val_size:]
            dataset_episodes = [dataset_episodes[i] for i in sorted(indices)]
            logging.info(f"Dataset {dataset_dir}: {len(dataset_episodes)} episodes in {args.split} split (select={args.select})")
        else:
            logging.info(f"Dataset {dataset_dir}: {len(dataset_episodes)} episodes (select={args.select})")
        all_episodes.extend(dataset_episodes)

    if args.num_episodes:
        all_episodes = all_episodes[:args.num_episodes]
    logging.info(f"Total: {len(all_episodes)} episodes to process")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    pred_vals: list[float] = []
    gt_vals: list[float] = []

    for ep_path in tqdm(all_episodes, desc="Inference"):
        ep_name = os.path.basename(ep_path)
        try:
            pred, gt = run_inference(policy, ep_path, args)
        except Exception as e:
            logging.warning(f"[FAIL] {ep_name}: {e}")
            continue
        if len(pred) < 2 or len(gt) < 2:
            continue
        pv = metric_fn(pred, args.hz)
        gv = metric_fn(gt, args.hz)
        names.append(ep_name)
        pred_vals.append(pv)
        gt_vals.append(gv)
        logging.info(f"{ep_name}: gt_max_vel={gv:.3f}, pred_max_vel={pv:.3f}")

    if not gt_vals:
        logging.warning("No valid episodes. Nothing to plot.")
        return

    tag = f"{args.metric}_{args.select}"
    plot_single_histogram(gt_vals, args.bin_width,
                          str(output_dir / f"max_vel_hist_gt_{tag}.png"),
                          args.metric, "GT", args.select)
    plot_single_histogram(pred_vals, args.bin_width,
                          str(output_dir / f"max_vel_hist_pred_{tag}.png"),
                          args.metric, "Pred", args.select)
    plot_compare_histogram(pred_vals, gt_vals, args.bin_width,
                           str(output_dir / f"max_vel_hist_{tag}.png"),
                           args.metric, args.select)
    save_csv(names, pred_vals, gt_vals,
             str(output_dir / f"max_vel_hist_{tag}.csv"))

    logging.info(f"Done! Results in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
