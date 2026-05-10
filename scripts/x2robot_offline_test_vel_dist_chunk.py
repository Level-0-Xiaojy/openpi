"""Offline max-velocity distribution, computed **per action chunk**.

与 ``x2robot_offline_test_vel_dist.py`` 的关键区别在"速度是怎么算出来的"：

rollout 时每次只根据当前观测产生 size=``chunk_size`` 的 chunk 并执行完，
然后再根据新观测做下一次推理。也就是说：
- 同一个 chunk 内的 action 是连续的、由同一次推理产生；
- 跨 chunk 之间，pred 并不保证连续（每次观测都是独立推理的），
  因此把整条轨迹拼起来再算 ``np.gradient`` 会在 chunk 边界制造假峰值。

所以这里的做法是：
1. 按 ``current_idx += chunk_size`` 推理（默认 chunk_size=20），每次拿到
   一整段 pred chunk 及对应的 gt chunk。
2. 对 **每个 chunk 独立** 调用 metric_fn 得到一个 max-velocity。
3. 一个 episode 取所有 chunk 的最大值作为该 episode 的 max velocity
   （也可选 ``--agg mean`` 看"平均一个 chunk 的峰值"）。
4. 跨 episode 画直方图。

运行示例：

    CUDA_VISIBLE_DEVICES=7 uv run scripts/x2robot_offline_test_vel_dist_chunk.py \
        --dataset-dir /mnt/.../beijing_guqiuyi_20260420_pm_tele \
        --policy-config fold_towel_sm2sm \
        --policy-dir checkpoints/.../29999 \
        --metric peak_vz --chunk-size 20 --hz 20 --bin-width 0.1 \
        --output-dir offline_test_results_vel_dist_chunk
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


# ---------- metric：和 vis_max_velocity_distribution.py 一致的口径 ----------

def _max_velocity_peakz(actions: np.ndarray, hz: float) -> float:
    if len(actions) < 2:
        return 0.0
    dt = 1.0 / hz
    lvz = np.gradient(actions[:, 2], dt)
    rvz = np.gradient(actions[:, 9], dt)
    return float(max(lvz.max(), rvz.max()))


def _max_velocity_speed(actions: np.ndarray, hz: float) -> float:
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
    dataset_dir: str = "datasets/x2robot/throw_0113"
    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: str | None = None
    state_history_size: int | None = None
    state_future_size: int | None = None
    state_step: int | None = None

    # rollout 一次推理执行的动作步数；每 chunk_size 帧重新推理一次
    chunk_size: int = 20

    num_episodes: int | None = None
    split: str | None = None
    val_ratio: float = 0.1
    split_seed: int = 42
    output_dir: str = "offline_test_results_vel_dist_chunk"

    hz: float = 20.0
    metric: str = "peak_vz"     # "peak_vz" 或 "max_speed"
    # 多个 chunk 聚合到 episode 的方式：max / mean / p95
    agg: str = "max"
    bin_width: float = 0.1
    select: str = "all"         # all / success / fail


# ---------- 推理 ----------

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


def _read_gt_chunk(frames: list, start_idx: int, length: int) -> np.ndarray:
    total = len(frames)
    out = []
    for i in range(length):
        idx = min(start_idx + 1 + i, total - 1)
        f = frames[idx]
        out.append(np.concatenate([
            f['master_left_position'], f['master_left_rotation'],
            [f['master_left_gripper']], f['master_right_position'],
            f['master_right_rotation'], [f['master_right_gripper']],
        ]).astype(np.float32))
    return np.array(out) if out else np.zeros((0, 14), dtype=np.float32)


def run_inference_chunked(policy, episode_path: str, args: Args):
    """Return list of (pred_chunk, gt_chunk) arrays, one per rollout step."""
    episode_name = os.path.basename(episode_path)
    with open(os.path.join(episode_path, f"{episode_name}.json"), 'r') as f:
        episode_data = json.load(f)

    frames = episode_data['data']
    total_frames = len(frames)
    videos = {k: cv2.VideoCapture(os.path.join(episode_path, f"{k}Img.mp4"))
              for k in ['left', 'face', 'right']}

    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    current_idx = 0
    while current_idx < total_frames:
        images = {}
        for k, cap in videos.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            images[f"{k}_wrist_view" if k != "face" else "face_view"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        indices = [max(0, min(current_idx + i * args.state_step, total_frames - 1))
                   for i in range(-args.state_history_size, args.state_future_size + 1)]
        state_seq = np.array([build_state(frames[i], args.policy_mode) for i in indices],
                             dtype=np.float32)

        obs = {'images': images, 'prompt': '', 'state': state_seq}
        action_pred = policy.infer(obs)['actions']

        if args.policy_mode == "sm2sm":
            action_pred = action_pred[:, 14:28]
        else:
            action_pred = action_pred[:, :14]

        pred_chunk = action_pred[:args.chunk_size]

        gt_length = min(args.chunk_size, total_frames - current_idx - 1)
        if gt_length > 0:
            gt_chunk = _read_gt_chunk(frames, current_idx, gt_length)
            chunks.append((pred_chunk[:gt_length], gt_chunk))

        current_idx += args.chunk_size

    for cap in videos.values():
        cap.release()
    return chunks


# ---------- 聚合 ----------

def _aggregate(values: list[float], agg: str) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    if agg == "max":
        return float(arr.max())
    if agg == "mean":
        return float(arr.mean())
    if agg == "p95":
        return float(np.percentile(arr, 95))
    raise ValueError(f"unknown agg: {agg}")


# ---------- select ----------

def _match_select(ep_name: str, select: str) -> bool:
    if select == "all":
        return True
    name_lower = ep_name.lower()
    if select == "success":
        return name_lower.endswith("success")
    if select == "fail":
        return name_lower.endswith("fail")
    return True


# ---------- 绘图 ----------

def _plot_single(values: list[float], bin_width: float, output_path: str,
                 metric: str, label: str, select: str, agg: str) -> None:
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

    ax.set_xlabel(f"Max velocity per episode ({metric}, agg={agg}) [m/s]")
    ax.set_ylabel("# episodes")
    ax.set_title(
        f"{label} max velocity distribution (per-chunk) "
        f"(n={len(arr)}, bin={bin_width}, metric={metric}, agg={agg}, select={select})\n"
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


def _plot_compare(pred_vals: list[float], gt_vals: list[float],
                  bin_width: float, output_path: str,
                  metric: str, select: str, agg: str) -> None:
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

    ax.set_xlabel(f"Max velocity per episode ({metric}, agg={agg}) [m/s]")
    ax.set_ylabel("# episodes")
    ax.set_title(
        f"Max velocity distribution: GT vs Pred (per-chunk) "
        f"(n={len(gt_arr)}, bin={bin_width}, metric={metric}, agg={agg}, select={select})"
    )
    ax.set_xticks(bins)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved compare histogram -> {output_path}")


def _save_csv(rows: list[dict], csv_path: str) -> None:
    if not rows:
        return
    fieldnames = ["episode", "n_chunks",
                  "gt_vel_episode", "pred_vel_episode",
                  "diff(pred-gt)",
                  "gt_vel_chunks", "pred_vel_chunks"]
    rows_sorted = sorted(rows, key=lambda r: -r["gt_vel_episode"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({
                "episode": r["episode"],
                "n_chunks": r["n_chunks"],
                "gt_vel_episode":   f"{r['gt_vel_episode']:.6f}",
                "pred_vel_episode": f"{r['pred_vel_episode']:.6f}",
                "diff(pred-gt)":    f"{r['pred_vel_episode'] - r['gt_vel_episode']:.6f}",
                "gt_vel_chunks":    ";".join(f"{v:.4f}" for v in r["gt_vel_chunks"]),
                "pred_vel_chunks":  ";".join(f"{v:.4f}" for v in r["pred_vel_chunks"]),
            })
    logging.info(f"Saved per-episode CSV -> {csv_path}")


# ---------- main ----------

def main(args: Args):
    if args.metric not in METRICS:
        raise ValueError(f"--metric must be one of {list(METRICS)}, got {args.metric}")
    if args.agg not in ("max", "mean", "p95"):
        raise ValueError(f"--agg must be one of max/mean/p95, got {args.agg}")
    metric_fn = METRICS[args.metric]

    if args.policy_mode is None:
        for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
            if mode in args.policy_dir.lower():
                args.policy_mode = mode
                logging.info(f"Auto-detected policy_mode from path: {args.policy_mode}")
                break
        if args.policy_mode is None:
            raise ValueError(f"Could not detect policy_mode from path: {args.policy_dir}")

    cfg = _config.get_config(args.policy_config)
    if args.state_history_size is None:
        args.state_history_size = getattr(cfg.data, 'state_history_size', 0)
    if args.state_future_size is None:
        args.state_future_size = getattr(cfg.data, 'state_future_size', 0)
    if args.state_step is None:
        args.state_step = getattr(cfg.data, 'state_step', 1)
    logging.info(
        f"state_history_size={args.state_history_size}, "
        f"state_future_size={args.state_future_size}, "
        f"state_step={args.state_step}, chunk_size={args.chunk_size}"
    )

    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)

    dataset_dirs = [d.strip() for d in args.dataset_dir.split(',')]
    all_episodes: list[str] = []
    for dataset_dir in dataset_dirs:
        eps = sorted([p for p in glob.glob(f'{dataset_dir}/*')
                      if os.path.isdir(p) and glob.glob(f'{p}/*.mp4')])
        eps = [p for p in eps if _match_select(os.path.basename(p), args.select)]
        if args.split:
            rng = np.random.RandomState(args.split_seed)
            indices = np.arange(len(eps))
            rng.shuffle(indices)
            val_size = int(len(eps) * args.val_ratio)
            indices = indices[:val_size] if args.split == "val" else indices[val_size:]
            eps = [eps[i] for i in sorted(indices)]
            logging.info(f"Dataset {dataset_dir}: {len(eps)} episodes in {args.split} split (select={args.select})")
        else:
            logging.info(f"Dataset {dataset_dir}: {len(eps)} episodes (select={args.select})")
        all_episodes.extend(eps)

    if args.num_episodes:
        all_episodes = all_episodes[:args.num_episodes]
    logging.info(f"Total: {len(all_episodes)} episodes to process")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ep_path in tqdm(all_episodes, desc="Inference"):
        ep_name = os.path.basename(ep_path)
        try:
            chunks = run_inference_chunked(policy, ep_path, args)
        except Exception as e:
            logging.warning(f"[FAIL] {ep_name}: {e}")
            continue
        if not chunks:
            continue

        gt_chunk_vals = [metric_fn(g, args.hz) for (_, g) in chunks]
        pred_chunk_vals = [metric_fn(p, args.hz) for (p, _) in chunks]

        ep_gt = _aggregate(gt_chunk_vals, args.agg)
        ep_pred = _aggregate(pred_chunk_vals, args.agg)

        rows.append({
            "episode": ep_name,
            "n_chunks": len(chunks),
            "gt_vel_episode": ep_gt,
            "pred_vel_episode": ep_pred,
            "gt_vel_chunks": gt_chunk_vals,
            "pred_vel_chunks": pred_chunk_vals,
        })
        logging.info(
            f"{ep_name}: n_chunks={len(chunks)}, "
            f"gt_{args.agg}={ep_gt:.3f}, pred_{args.agg}={ep_pred:.3f}"
        )

    if not rows:
        logging.warning("No valid episodes. Nothing to plot.")
        return

    gt_vals = [r["gt_vel_episode"] for r in rows]
    pred_vals = [r["pred_vel_episode"] for r in rows]

    tag = f"{args.metric}_{args.agg}_{args.select}_chunk{args.chunk_size}"
    _plot_single(gt_vals, args.bin_width,
                 str(output_dir / f"max_vel_hist_gt_{tag}.png"),
                 args.metric, "GT", args.select, args.agg)
    _plot_single(pred_vals, args.bin_width,
                 str(output_dir / f"max_vel_hist_pred_{tag}.png"),
                 args.metric, "Pred", args.select, args.agg)
    _plot_compare(pred_vals, gt_vals, args.bin_width,
                  str(output_dir / f"max_vel_hist_{tag}.png"),
                  args.metric, args.select, args.agg)
    _save_csv(rows, str(output_dir / f"max_vel_hist_{tag}.csv"))

    logging.info(f"Done! Results in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
