"""构造 Velocity Guider 训练数据集的主入口。

用法：
    cd /home/guqiuyi/workspace/openpi
    uv run python velocity_guider/build_dataset.py \
        --config velocity_guider/configs/build_v2.yaml

输出布局（写到 ``output.root`` 下，例如 ``/mnt/public/guqiuyi/dataset/velocity_guider_data/v2/``）：

    ├── train/ep_<repo>_<ep_idx>.npz   # 每个 train episode 一个 npz
    ├── val/ep_<repo>_<ep_idx>.npz
    ├── samples.parquet                 # 单进程或 merge 后的全局样本索引
    ├── samples_shard_000.parquet       # 多 shard 模式下每个进程单独写
    ├── action_stats.json               # 14 维 master action 的归一化统计
    └── build_config.yaml               # 配置快照（便于复现）

每个 npz 内容：
    obs_feat:     [num_t, 3*width]      float32  — 每个起点 t 一个 obs 特征
    chunks:       [num_t, 3, 20, 14]    float32  — axis=1 顺序: v_mode=3, 2, 1
    v_modes:      [num_t, 3]            int8     — [[3,2,1], ...]
    frame_idx:    [num_t]               int32    — 在原 episode 中的起点帧索引
    burst_mask:   [num_t]               bool     — 该帧是否处于爆发阶段

数据构造原理（详细见 README.md）：
    v_mode=3 chunk = demo[t:t+20]（所有帧）
    v_mode=2 chunk = resample(demo[t:t+14], target_len=20)（仅 burst 帧）
    v_mode=1 chunk = resample(demo[t:t+ 7], target_len=20)（仅 burst 帧）
    非 burst 帧的 v_mode=2/1 slot 复制 v_mode=3 的值（不参与训练采样）

burst 检测支持两种模式（在 YAML 的 burst: 块配置）：
    - "threshold": 左右臂 position speed max > speed_threshold
    - "peak_window": position z 正向速度全局最大帧 ± window 秒
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from velocity_guider.data.action_stats import compute_master_action_stats, write_action_stats
from velocity_guider.data.lerobot_loader import (
    CAMERA_KEYS,
    LeRobotDatasetInfo,
    iter_episode_image_batches,
)
from velocity_guider.data.resample import V_MODE_SOURCE_LEN, build_three_v_mode_chunks
from velocity_guider.data.vision_encoder import VisionEncoder

logger = logging.getLogger("velocity_guider.build_dataset")

V_MODES_ORDER: tuple[int, int, int] = (3, 2, 1)  # axis=1 of chunks 数组


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class BuildConfig:
    # data
    lerobot_root: str = "/mnt/public/guqiuyi/huggingface/lerobot"
    repos: list[str] = field(default_factory=lambda: [
        "fold_towel_gqy_0317",
        "fold_towel_gqy_0318",
        "fold_towel_gqy_0410",
        "fold_towel_gqy_0420",
    ])
    val_ratio: float = 0.1
    val_split_seed: int = 0
    episode_limit: int | None = None  # debug 用：每个 repo 最多取多少 episode

    # chunks
    chunk_size: int = 20
    stride: int = 5

    # vision encoder
    pi0_checkpoint_path: str = "/mnt/public/models/pytorch_models/pi0_base_pytorch"
    pi0_action_horizon: int = 20
    encoder_device: str = "cuda:0"
    # 当前环境未安装 openpi 的 transformers_replace patch；float32 最稳。
    # 若后续修好 patch，可在 yaml 中改回 bfloat16 提速。
    encoder_dtype: str = "float32"
    encoder_batch_size: int = 32

    # burst detection
    burst_mode: str = "threshold"        # "threshold" | "peak_window"
    burst_speed_threshold: float = 0.6   # burst_mode="threshold": absolute speed (m/s)
    burst_peak_window: float = 2.0       # burst_mode="peak_window": ±window (seconds)
    burst_dilate_frames: int = 5         # dilate burst mask by ±N frames
    burst_hz: int = 20                   # frequency for velocity computation

    # output
    output_root: str = "/mnt/public/guqiuyi/dataset/velocity_guider_data/v2"
    overwrite: bool = False

    @staticmethod
    def from_yaml(path: Path) -> "BuildConfig":
        with path.open("r") as f:
            raw = yaml.safe_load(f)

        cfg = BuildConfig()
        data_cfg = raw.get("data", {})
        cfg.lerobot_root = data_cfg.get("lerobot_root", cfg.lerobot_root)
        cfg.repos = data_cfg.get("repos", cfg.repos)
        cfg.val_ratio = data_cfg.get("val_ratio", cfg.val_ratio)
        cfg.val_split_seed = data_cfg.get("val_split_seed", cfg.val_split_seed)
        cfg.episode_limit = data_cfg.get("episode_limit", cfg.episode_limit)

        chunk_cfg = raw.get("chunks", {})
        cfg.chunk_size = chunk_cfg.get("size", cfg.chunk_size)
        cfg.stride = chunk_cfg.get("stride", cfg.stride)

        burst_cfg = raw.get("burst", {})
        cfg.burst_mode = burst_cfg.get("mode", cfg.burst_mode)
        cfg.burst_speed_threshold = burst_cfg.get("speed_threshold", cfg.burst_speed_threshold)
        cfg.burst_peak_window = burst_cfg.get("peak_window", cfg.burst_peak_window)
        cfg.burst_dilate_frames = burst_cfg.get("dilate_frames", cfg.burst_dilate_frames)
        cfg.burst_hz = burst_cfg.get("hz", cfg.burst_hz)

        enc_cfg = raw.get("vision_encoder", {})
        cfg.pi0_checkpoint_path = enc_cfg.get("pi0_checkpoint_path", cfg.pi0_checkpoint_path)
        cfg.pi0_action_horizon = enc_cfg.get("pi0_action_horizon", cfg.pi0_action_horizon)
        cfg.encoder_device = enc_cfg.get("device", cfg.encoder_device)
        cfg.encoder_dtype = enc_cfg.get("dtype", cfg.encoder_dtype)
        cfg.encoder_batch_size = enc_cfg.get("batch_size", cfg.encoder_batch_size)

        out_cfg = raw.get("output", {})
        cfg.output_root = out_cfg.get("root", cfg.output_root)
        cfg.overwrite = out_cfg.get("overwrite", cfg.overwrite)

        return cfg


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def split_train_val(
    info: LeRobotDatasetInfo,
    val_ratio: float,
    seed: int,
    episode_limit: int | None = None,
) -> tuple[list[int], list[int]]:
    """按 episode 划分 train / val（每个 repo 独立、确定性）。"""
    eps = info.list_episodes()
    if episode_limit is not None:
        eps = eps[:episode_limit]
    stable_repo_hash = int(hashlib.md5(info.repo_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_repo_hash % 10_000)
    shuffled = eps.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def _position_speed(master: np.ndarray, hz: int) -> np.ndarray:
    """Compute per-frame max(left, right) position speed from 14-dim master actions."""
    dt = 1.0 / hz
    left_vel = np.gradient(master[:, 0:3], dt, axis=0)
    right_vel = np.gradient(master[:, 7:10], dt, axis=0)
    left_speed = np.linalg.norm(left_vel, axis=1)
    right_speed = np.linalg.norm(right_vel, axis=1)
    return np.maximum(left_speed, right_speed)


def _dilate_mask(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask
    kernel = np.ones(2 * n + 1, dtype=float)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


def detect_burst_mask(
    master: np.ndarray,
    cfg: BuildConfig,
) -> np.ndarray:
    """Detect burst frames.  Returns bool array of shape ``[T]``.

    Two modes:
    - ``"threshold"``: frame is burst if max(left, right) position speed
      exceeds ``cfg.burst_speed_threshold``.
    - ``"peak_window"``: find the frame with max positive left/right
      position-z velocity, mark ``[t_peak - window, t_peak + window]``.
    """
    hz = cfg.burst_hz
    T = master.shape[0]

    if cfg.burst_mode == "threshold":
        speed = _position_speed(master, hz)
        is_burst = speed > cfg.burst_speed_threshold

    elif cfg.burst_mode == "peak_window":
        dt = 1.0 / hz
        left_vz = np.gradient(master[:, 2], dt)
        right_vz = np.gradient(master[:, 9], dt)
        combined = np.concatenate([left_vz, right_vz])
        idx = int(np.argmax(combined))
        frame_idx = idx if idx < T else idx - T
        window_frames = int(round(cfg.burst_peak_window * hz))
        start = max(0, frame_idx - window_frames)
        end = min(T, frame_idx + window_frames + 1)
        is_burst = np.zeros(T, dtype=bool)
        is_burst[start:end] = True

    else:
        raise ValueError(f"Unknown burst_mode: {cfg.burst_mode!r}")

    is_burst = _dilate_mask(is_burst, cfg.burst_dilate_frames)
    return is_burst


def shard_tasks(
    tasks: list[tuple[LeRobotDatasetInfo, str, int]],
    num_shards: int,
    shard_id: int,
) -> list[tuple[LeRobotDatasetInfo, str, int]]:
    """Deterministically assign flattened episode tasks to this shard."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    return tasks[shard_id::num_shards]


def merge_shard_outputs(out_root: Path) -> int:
    """Merge ``samples_shard_*.parquet`` into ``samples.parquet``."""
    shard_paths = sorted(out_root.glob("samples_shard_*.parquet"))
    if not shard_paths:
        logger.error("No shard parquet files found under %s", out_root)
        return 2

    dfs = []
    for p in shard_paths:
        df = pd.read_parquet(p)
        dfs.append(df)
        logger.info("Loaded %s: %d rows", p.name, len(df))

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(
        ["split", "source_repo", "episode_idx", "frame_idx", "shard_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    merged_path = out_root / "samples.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info("Wrote merged %s: %d rows", merged_path, len(merged))
    if "split" in merged:
        logger.info("  split rows: %s", merged.groupby("split").size().to_dict())

    # Also write a small manifest so downstream code can see what was merged.
    manifest = {
        "num_shards": len(shard_paths),
        "shard_files": [p.name for p in shard_paths],
        "num_rows": int(len(merged)),
        "splits": {str(k): int(v) for k, v in merged.groupby("split").size().to_dict().items()},
    }
    with (out_root / "merge_manifest.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.info("Wrote merge_manifest.yaml")
    return 0


def build_episode_obs_feats(
    encoder: VisionEncoder,
    info: LeRobotDatasetInfo,
    ep_idx: int,
    batch_size: int,
    expected_num_frames: int | None = None,
) -> tuple[np.ndarray, int]:
    """对一个 episode 的全部帧批量提特征。

    Returns:
        ``(obs_feats [T, 3*width] float32, T_decoded)``
    """
    all_feats: list[np.ndarray] = []
    for _indices, imgs in iter_episode_image_batches(
        info, ep_idx, batch_size=batch_size, expected_num_frames=expected_num_frames
    ):
        feat = encoder.encode(imgs)  # [B, 3*width]
        all_feats.append(feat)
    feats = np.concatenate(all_feats, axis=0)
    return feats, feats.shape[0]


def build_chunks_for_episode(
    master_actions: np.ndarray,
    chunk_size: int,
    stride: int,
    is_burst: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build chunks with burst-aware v_mode augmentation.

    Returns ``(frame_idx [N], chunks [N, 3, K, 14], v_modes [N, 3], burst_mask [N])``.

    When ``is_burst`` is provided, v_mode=2/1 slots are only populated for
    burst frames.  Non-burst frames get v_mode=3 data copied into all three
    slots (the duplicates are filtered out by the Dataset at training time).
    """
    T = master_actions.shape[0]
    max_src_len = max(V_MODE_SOURCE_LEN.values())
    last_t = T - max_src_len
    if last_t < 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3, chunk_size, 14), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int8),
            np.zeros((0,), dtype=bool),
        )

    starts = list(range(0, last_t + 1, stride))
    frame_idx_arr = np.array(starts, dtype=np.int32)
    chunks_arr = np.zeros((len(starts), 3, chunk_size, 14), dtype=np.float32)
    v_modes_arr = np.zeros((len(starts), 3), dtype=np.int8)
    burst_arr = np.zeros(len(starts), dtype=bool)

    for i, t in enumerate(starts):
        frame_is_burst = is_burst[t] if is_burst is not None else True
        burst_arr[i] = frame_is_burst

        three = build_three_v_mode_chunks(master_actions, t, chunk_size=chunk_size)
        if three is None:
            raise RuntimeError(f"unexpected None at t={t} (T={T})")

        vm3_chunk = three[3]
        for j, vm in enumerate(V_MODES_ORDER):
            if vm == 3 or frame_is_burst:
                chunks_arr[i, j] = three[vm]
            else:
                chunks_arr[i, j] = vm3_chunk
            v_modes_arr[i, j] = vm

    return frame_idx_arr, chunks_arr, v_modes_arr, burst_arr


def process_episode(
    encoder: VisionEncoder,
    info: LeRobotDatasetInfo,
    ep_idx: int,
    cfg: BuildConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """对一个 episode 提特征 + 构造 chunk + 落盘 npz。返回该 episode 的样本元数据。"""
    t0 = time.perf_counter()
    master = info.get_episode_master_actions(ep_idx)  # [T, 14]
    T = master.shape[0]

    obs_feats, T_dec = build_episode_obs_feats(
        encoder, info, ep_idx,
        batch_size=cfg.encoder_batch_size,
        expected_num_frames=T,
    )
    if T_dec != T:
        logger.warning(
            "ep %d (%s): action T=%d but decoded video frames=%d; aligning to min",
            ep_idx, info.repo_name, T, T_dec,
        )
        T = min(T, T_dec)
        master = master[:T]
        obs_feats = obs_feats[:T]

    is_burst_full = detect_burst_mask(master, cfg)

    frame_idx, chunks, v_modes, burst_mask = build_chunks_for_episode(
        master, chunk_size=cfg.chunk_size, stride=cfg.stride,
        is_burst=is_burst_full,
    )
    n_t = len(frame_idx)
    if n_t == 0:
        logger.warning("ep %d (%s) too short (T=%d) — skipped", ep_idx, info.repo_name, T)
        return {"n_t": 0}

    n_burst = int(burst_mask.sum())
    sel_obs = obs_feats[frame_idx]  # [n_t, 3*width]

    out_path = out_dir / f"ep_{info.repo_name}_{ep_idx:06d}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs_feat=sel_obs.astype(np.float32),
        chunks=chunks.astype(np.float32),
        v_modes=v_modes.astype(np.int8),
        frame_idx=frame_idx.astype(np.int32),
        episode_idx=np.int32(ep_idx),
        burst_mask=burst_mask,
    )

    dt = time.perf_counter() - t0
    logger.info(
        "%s ep %4d: T=%4d, n_t=%4d, burst=%4d/%4d, dt=%.1fs (%.1f frames/s) -> %s",
        info.repo_name, ep_idx, T, n_t, n_burst, n_t, dt, T / max(dt, 1e-6), out_path.name,
    )
    return {
        "n_t": n_t,
        "out_path": str(out_path),
        "T": T,
        "frame_idx": frame_idx,
        "burst_mask": burst_mask,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Velocity Guider training dataset.")
    parser.add_argument("--config", type=str, required=True, help="path to build YAML config")
    parser.add_argument("--repo", type=str, default=None, help="only process this repo (debug)")
    parser.add_argument("--episode-limit", type=int, default=None,
                        help="override episode_limit per repo (debug)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="number of independent episode shards to split work into")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="this process shard id in [0, num_shards)")
    parser.add_argument("--merge-shards", action="store_true",
                        help="only merge samples_shard_*.parquet into samples.parquet and exit")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = BuildConfig.from_yaml(Path(args.config))
    if args.episode_limit is not None:
        cfg.episode_limit = args.episode_limit
    if args.repo is not None:
        cfg.repos = [args.repo]

    out_root = Path(cfg.output_root)
    if args.merge_shards:
        return merge_shard_outputs(out_root)

    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
        )

    if out_root.exists() and cfg.overwrite:
        if args.num_shards > 1 and args.shard_id != 0:
            logger.warning(
                "output.overwrite=true ignored on shard %d. "
                "Only shard 0 removes the output directory.",
                args.shard_id,
            )
        elif args.num_shards > 1 and args.shard_id == 0:
            logger.warning(
                "output.overwrite=true with multiple shards will remove %s in shard 0. "
                "Start shard 0 first or clean the directory manually before launching all shards.",
                out_root,
            )
            logger.warning("Removing existing output dir: %s", out_root)
            shutil.rmtree(out_root)
        else:
            logger.warning("Removing existing output dir: %s", out_root)
            shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train").mkdir(exist_ok=True)
    (out_root / "val").mkdir(exist_ok=True)

    # --- 加载所有 dataset 元信息，划分 train/val ---
    dataset_infos: list[LeRobotDatasetInfo] = []
    train_eps_per_repo: dict[str, list[int]] = {}
    val_eps_per_repo: dict[str, list[int]] = {}
    for repo in cfg.repos:
        info = LeRobotDatasetInfo(repo, cfg.lerobot_root)
        dataset_infos.append(info)
        train_eps, val_eps = split_train_val(
            info, val_ratio=cfg.val_ratio, seed=cfg.val_split_seed,
            episode_limit=cfg.episode_limit,
        )
        train_eps_per_repo[repo] = train_eps
        val_eps_per_repo[repo] = val_eps
        logger.info(
            "%s: total_ep=%d, fps=%d -> train=%d, val=%d",
            repo, info.total_episodes, info.fps, len(train_eps), len(val_eps),
        )

    # --- 1) 计算 action stats（只用 train） ---
    action_stats_path = out_root / "action_stats.json"
    if args.num_shards == 1 or args.shard_id == 0:
        stats = compute_master_action_stats(dataset_infos, train_eps_per_repo)
        write_action_stats(stats, action_stats_path)
    else:
        logger.info(
            "Shard %d/%d skips action stats; shard 0 writes %s",
            args.shard_id,
            args.num_shards,
            action_stats_path,
        )

    # --- 2) 加载 vision encoder ---
    encoder = VisionEncoder(
        checkpoint_path=cfg.pi0_checkpoint_path,
        action_horizon=cfg.pi0_action_horizon,
        device=cfg.encoder_device,
        dtype=cfg.encoder_dtype,
    )
    logger.info(
        "VisionEncoder ready: paligemma=%s, obs_feat_dim=%d",
        encoder.paligemma_variant, encoder.obs_feat_dim,
    )

    # --- 3) 逐 episode 处理 ---
    all_tasks: list[tuple[LeRobotDatasetInfo, str, int]] = []
    for info in dataset_infos:
        for split, ep_list in [
            ("train", train_eps_per_repo[info.repo_name]),
            ("val", val_eps_per_repo[info.repo_name]),
        ]:
            for ep_idx in ep_list:
                all_tasks.append((info, split, ep_idx))

    this_tasks = shard_tasks(all_tasks, args.num_shards, args.shard_id)
    logger.info(
        "Shard %d/%d processing %d / %d episode tasks",
        args.shard_id,
        args.num_shards,
        len(this_tasks),
        len(all_tasks),
    )

    sample_rows: list[dict] = []
    for info, split, ep_idx in tqdm(
        this_tasks,
        desc=f"shard {args.shard_id}/{args.num_shards}",
        leave=False,
    ):
        split_dir = out_root / split
        try:
            meta = process_episode(encoder, info, ep_idx, cfg, split_dir)
        except Exception as e:
            logger.exception("Failed on %s ep %d: %s", info.repo_name, ep_idx, e)
            continue
        if meta["n_t"] == 0:
            continue
        burst_flags = meta["burst_mask"]
        for sid, t in enumerate(meta["frame_idx"].tolist()):
            sample_rows.append({
                "split": split,
                "source_repo": info.repo_name,
                "episode_idx": ep_idx,
                "frame_idx": int(t),
                "sample_id_in_episode": sid,
                "is_burst": bool(burst_flags[sid]),
                "out_path": meta["out_path"],
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
            })

    # --- 4) 写 parquet + 配置快照 ---
    if sample_rows:
        df = pd.DataFrame(sample_rows)
        samples_name = (
            "samples.parquet"
            if args.num_shards == 1
            else f"samples_shard_{args.shard_id:03d}.parquet"
        )
        df.to_parquet(out_root / samples_name, index=False)
        logger.info("Wrote %s: %d rows", samples_name, len(df))
        logger.info("  train rows: %d", int((df["split"] == "train").sum()))
        logger.info("  val   rows: %d", int((df["split"] == "val").sum()))
    else:
        if args.num_shards > 1:
            samples_name = f"samples_shard_{args.shard_id:03d}.parquet"
            empty_df = pd.DataFrame(
                columns=[
                    "split",
                    "source_repo",
                    "episode_idx",
                    "frame_idx",
                    "sample_id_in_episode",
                    "is_burst",
                    "out_path",
                    "shard_id",
                    "num_shards",
                ]
            )
            empty_df.to_parquet(out_root / samples_name, index=False)
            logger.warning(
                "Shard %d/%d produced no samples; wrote empty %s",
                args.shard_id,
                args.num_shards,
                samples_name,
            )
        else:
            logger.error("No samples were produced — check inputs.")
            return 2

    # 配置快照
    snapshot = {
        "lerobot_root": cfg.lerobot_root,
        "repos": cfg.repos,
        "val_ratio": cfg.val_ratio,
        "val_split_seed": cfg.val_split_seed,
        "episode_limit": cfg.episode_limit,
        "chunk_size": cfg.chunk_size,
        "stride": cfg.stride,
        "burst_mode": cfg.burst_mode,
        "burst_speed_threshold": cfg.burst_speed_threshold,
        "burst_peak_window": cfg.burst_peak_window,
        "burst_dilate_frames": cfg.burst_dilate_frames,
        "burst_hz": cfg.burst_hz,
        "v_mode_source_len": dict(V_MODE_SOURCE_LEN),
        "v_modes_order_in_chunks_axis1": list(V_MODES_ORDER),
        "pi0_checkpoint_path": cfg.pi0_checkpoint_path,
        "pi0_action_horizon": cfg.pi0_action_horizon,
        "paligemma_variant": encoder.paligemma_variant,
        "obs_feat_dim": encoder.obs_feat_dim,
        "encoder_dtype": cfg.encoder_dtype,
        "encoder_device": cfg.encoder_device,
        "encoder_batch_size": cfg.encoder_batch_size,
        "output_root": str(out_root),
        "cameras": list(CAMERA_KEYS),
        "lerobot_to_openpi_camera_map": {
            "face_view": "base_0_rgb",
            "left_wrist_view": "left_wrist_0_rgb",
            "right_wrist_view": "right_wrist_0_rgb",
        },
        "train_episodes_per_repo": train_eps_per_repo,
        "val_episodes_per_repo": val_eps_per_repo,
        "action_dim_master": 14,
        "action_slice_in_lerobot_actions": [14, 28],
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_all_episode_tasks": len(all_tasks),
        "num_this_shard_episode_tasks": len(this_tasks),
    }
    build_config_name = (
        "build_config.yaml"
        if args.num_shards == 1
        else f"build_config_shard_{args.shard_id:03d}.yaml"
    )
    with (out_root / build_config_name).open("w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    logger.info("Wrote %s snapshot", build_config_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
