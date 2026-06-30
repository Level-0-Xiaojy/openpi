"""
Convert X2Robot raw data (mp4 + json) to RECAP-compatible LeRobot datasets.

Produces TWO independent LeRobot datasets from the raw data directories:
  1. SFT dataset  -- all episodes treated as successful (no is_success column needed)
  2. Rollout dataset -- mixed success/fail episodes, with per-frame is_success column

The is_success flag is inferred as follows:
  - rollout datasets: from episode directory suffix (*_success / *_fail)
  - dagger datasets:  all episodes treated as successful
  - other datasets:   from episode suffix, default True if no suffix

Based on convert_x2robot_data_to_lerobot_v5.py; adds:
  - is_success as a boolean feature in parquet data (for rollout datasets)
  - is_success field in episodes.jsonl metadata
  - Builds sft and rollout datasets in one invocation

Usage (fold_towel example):

    DATA=/mnt/public/datasets/standardized_v1/x2robot/fold_towel
    OUT=/mnt/public/guqiuyi/RLinf_active/datasets

    python convert_x2robot_data_to_lerobot_recap.py \\
        --sft-raw-paths \\
            $DATA/beijing_guqiuyi_20260317_pm_tele \\
            $DATA/beijing_guqiuyi_20260318_pm_tele \\
            $DATA/beijing_guqiuyi_20260420_pm_tele \\
        --rollout-raw-paths \\
            $DATA/beijing_guqiuyi_20260325_pm_rollout \\
            $DATA/beijing_guqiuyi_20260330_pm_rollout \\
            $DATA/beijing_guqiuyi_20260407_pm_rollout \\
            $DATA/beijing_guqiuyi_20260423_pm_rollout \\
            $DATA/beijing_guqiuyi_20260331_pm_dagger_filter_static \\
        --sft-repo-name fold_towel_sft \\
        --rollout-repo-name fold_towel_rollout \\
        --task "Fold the towel." \\
        --output-dir $OUT
"""

import dataclasses
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

os.environ["SVT_LOG"] = "0"


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

@dataclass
class ConvertArgs:
    sft_raw_paths: list[str] = dataclasses.field(default_factory=list)
    rollout_raw_paths: list[str] = dataclasses.field(default_factory=list)
    sft_repo_name: str = "fold_towel_sft"
    rollout_repo_name: str = "fold_towel_rollout"
    task: str = "Fold the towel."
    push_to_hub: bool = False
    debug: bool = False
    debug_episodes: int = 3
    low_resolution: bool = True
    num_workers: int = 10
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Camera / state / action definitions
# ---------------------------------------------------------------------------

FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}

STATE_KEYS = [
    "follow_left_position",
    "follow_left_rotation",
    "follow_left_gripper",
    "follow_right_position",
    "follow_right_rotation",
    "follow_right_gripper",
    "master_left_position",
    "master_left_rotation",
    "master_left_gripper",
    "master_right_position",
    "master_right_rotation",
    "master_right_gripper",
]

ACTION_KEYS = [
    "follow_left_position",
    "follow_left_rotation",
    "follow_left_gripper",
    "follow_right_position",
    "follow_right_rotation",
    "follow_right_gripper",
    "master_left_position",
    "master_left_rotation",
    "master_left_gripper",
    "master_right_position",
    "master_right_rotation",
    "master_right_gripper",
]


def get_dim_from_keys(keys: list[str]) -> int:
    dim = 0
    for key in keys:
        if "gripper" in key:
            dim += 1
        elif "position" in key:
            dim += 3
        elif "rotation" in key:
            dim += 3
        elif "joint" in key:
            dim += 7
        else:
            raise ValueError(f"Unknown key type: {key}")
    return dim


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def infer_is_success(episode_dir_name: str) -> bool:
    """Infer success/failure from the episode directory name suffix."""
    name = episode_dir_name.rstrip("/").lower()
    if name.endswith("_fail"):
        return False
    if name.endswith("_success"):
        return True
    return True


def infer_is_success_from_dataset(raw_path: str, episode_dir_name: str) -> bool:
    """Infer is_success based on dataset directory name and episode name.

    - Dataset name contains 'dagger' -> always True (human-corrected data)
    - Dataset name contains 'rollout' -> infer from episode suffix (_success/_fail)
    - Otherwise -> fall back to episode suffix inference
    """
    dataset_name = os.path.basename(raw_path.rstrip("/")).lower()
    if "dagger" in dataset_name:
        return True
    if "rollout" in dataset_name:
        return infer_is_success(episode_dir_name)
    return infer_is_success(episode_dir_name)


def find_episodes(raw_paths: list[str]) -> list[tuple[str, bool]]:
    """Find episode directories and their success status.

    Returns list of (episode_path, is_success).
    """
    episodes = []
    for raw_path in raw_paths:
        for dir_path in sorted(glob.glob(f"{raw_path}/*")):
            if not os.path.isdir(dir_path):
                continue
            mp4_files = glob.glob(f"{dir_path}/*.mp4")
            if len(mp4_files) > 0:
                is_success = infer_is_success_from_dataset(
                    raw_path, os.path.basename(dir_path)
                )
                episodes.append((dir_path, is_success))
    return episodes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_data(episode_path: str) -> tuple[np.ndarray, np.ndarray]:
    episode_name = os.path.basename(episode_path)
    json_path = os.path.join(episode_path, f"{episode_name}.json")
    with open(json_path, "r") as f:
        data = json.load(f)["data"]

    all_keys = set(STATE_KEYS) | set(ACTION_KEYS)
    trajectories = {key: [] for key in all_keys}
    for frame_data in data:
        for key in all_keys:
            trajectories[key].append(frame_data[key])

    arrays = {}
    for key, vals in trajectories.items():
        arr = np.array(vals, dtype=np.float32)
        if "gripper" in key:
            arr = arr.reshape(-1, 1)
        arrays[key] = arr

    state_array = np.concatenate([arrays[key] for key in STATE_KEYS], axis=1)
    action_array = np.concatenate([arrays[key] for key in ACTION_KEYS], axis=1)
    return state_array, action_array


# ---------------------------------------------------------------------------
# Video transcoding
# ---------------------------------------------------------------------------

def transcode_video_ffmpeg(
    input_path: str,
    output_path: Path,
    target_size: tuple[int, int] = (320, 240),
    fps: int = 20,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 23,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale={target_size[0]}:{target_size[1]}",
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        "-g", str(g),
        "-crf", str(crf),
        str(output_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**os.environ, "SVT_LOG": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed for {input_path}: {result.stderr}"
        )

    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(output_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    return int(probe_result.stdout.strip())


def transcode_single_video(
    episode_path: str,
    episode_index: int,
    camera_name: str,
    video_filename: str,
    output_root: Path,
    target_size: tuple[int, int],
    fps: int = 20,
) -> tuple[int, str, int]:
    video_path = os.path.join(episode_path, video_filename)
    output_path = (
        output_root / "videos" / "chunk-000" / camera_name
        / f"episode_{episode_index:06d}.mp4"
    )
    num_frames = transcode_video_ffmpeg(
        video_path, output_path, target_size, fps,
    )
    return episode_index, camera_name, num_frames


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def get_dummy_video_stats(num_frames: int) -> dict:
    return {
        "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
        "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "mean": np.array([[[0.4]], [[0.4]], [[0.4]]]),
        "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),
        "count": np.array([num_frames]),
    }


def compute_episode_stats_with_dummy_video(
    episode_buffer: dict, features: dict, video_frame_count: int
) -> dict:
    from lerobot.common.datasets.compute_stats import get_feature_stats

    ep_stats = {}
    for key, data in episode_buffer.items():
        if key not in features:
            continue
        dtype = features[key]["dtype"]
        if dtype == "string":
            continue
        elif dtype in ("image", "video"):
            ep_stats[key] = get_dummy_video_stats(video_frame_count)
        elif dtype == "bool":
            continue
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1
            ep_stats[key] = get_feature_stats(
                ep_ft_array, axis=axes_to_reduce, keepdims=keepdims
            )
    return ep_stats


# ---------------------------------------------------------------------------
# Custom LeRobotDataset that skips video I/O and writes is_success
# ---------------------------------------------------------------------------

class NoVideoIOLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that skips all video/image I/O in save_episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_all_media = False
        self._video_frame_count = 0
        self._episode_is_success = True

    def _save_image(self, image, fpath: Path) -> None:
        if self._skip_all_media:
            return
        super()._save_image(image, fpath)

    def save_episode(self, episode_data: dict | None = None) -> None:
        """Override to skip video I/O, use dummy stats, and write is_success."""
        if not self._skip_all_media:
            super().save_episode(episode_data)
            return

        if not episode_data:
            episode_buffer = self.episode_buffer

        from lerobot.common.datasets.lerobot_dataset import (
            aggregate_stats,
            validate_episode_buffer,
            write_episode,
            write_episode_stats,
            write_info,
        )

        validate_episode_buffer(
            episode_buffer, self.meta.total_episodes, self.features
        )

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full(
            (episode_length,), episode_index
        )

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self.features.items():
            if key in ("index", "episode_index", "task_index"):
                continue
            if ft["dtype"] in ("image", "video"):
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        ep_stats = compute_episode_stats_with_dummy_video(
            episode_buffer, self.features, self._video_frame_count
        )

        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length

        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1

        self.meta.info["splits"] = {
            "train": f"0:{self.meta.info['total_episodes']}"
        }
        self.meta.info["total_videos"] += len(self.meta.video_keys)

        write_info(self.meta.info, self.meta.root)

        # Write episode metadata including is_success
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            "is_success": self._episode_is_success,
        }
        self.meta.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.meta.root)

        self.meta.episodes_stats[episode_index] = ep_stats
        self.meta.stats = (
            aggregate_stats([self.meta.stats, ep_stats])
            if self.meta.stats
            else ep_stats
        )
        write_episode_stats(episode_index, ep_stats, self.meta.root)

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    def finalize_video_info(self) -> None:
        self.meta.update_video_info()
        from lerobot.common.datasets.lerobot_dataset import write_info
        write_info(self.meta.info, self.meta.root)

    @classmethod
    def create(cls, **kwargs) -> "NoVideoIOLeRobotDataset":
        parent_obj = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent_obj.__dict__)
        obj._skip_all_media = False
        obj._video_frame_count = 0
        obj._episode_is_success = True
        return obj


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    episodes: list[tuple[str, bool]],
    repo_name: str,
    output_dir: Path,
    task: str,
    include_is_success: bool,
    target_size: tuple[int, int],
    num_workers: int,
) -> Path:
    """Build a single LeRobot dataset from a list of (episode_path, is_success).

    Args:
        include_is_success: If True, add is_success as a boolean column
                            in the parquet data (required for rollout datasets).
    """
    output_path = output_dir / repo_name
    if output_path.exists():
        print(f"  Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    shape = (target_size[1], target_size[0], 3)

    features: dict = {
        "face_view": {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        },
        "left_wrist_view": {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        },
        "right_wrist_view": {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (get_dim_from_keys(STATE_KEYS),),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (get_dim_from_keys(ACTION_KEYS),),
            "names": ["actions"],
        },
    }

    if include_is_success:
        features["is_success"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_success"],
        }

    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="ARX",
        fps=20,
        features=features,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    dataset._skip_all_media = True

    # --- PHASE 1: Parallel video transcoding ---
    print(
        f"  Transcoding {len(episodes)} episodes x "
        f"{len(FILE_CAMERA_MAPPING)} cameras..."
    )
    t0 = time.time()

    transcode_tasks = []
    for ep_idx, (ep_path, _) in enumerate(episodes):
        for cam_name, vid_file in FILE_CAMERA_MAPPING.items():
            transcode_tasks.append(
                (ep_path, ep_idx, cam_name, vid_file,
                 output_path, target_size, 20)
            )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(transcode_single_video, *t): t[:3]
            for t in transcode_tasks
        }
        with tqdm.tqdm(
            total=len(transcode_tasks), desc="  Transcoding"
        ) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    ep_path, ep_idx, cam = futures[future]
                    print(
                        f"  Error transcoding ep {ep_idx} cam {cam}: {e}"
                    )
                    raise
                pbar.update(1)

    t1 = time.time()
    print(f"  Transcoding done in {t1 - t0:.1f}s")

    # --- PHASE 2: Build dataset metadata + parquet ---
    print("  Building parquet data...")
    dummy_image = np.zeros(
        (shape[0], shape[1], shape[2]), dtype=np.uint8
    )

    import datasets as _hf_datasets
    _hf_datasets.disable_progress_bars()

    success_count = 0
    fail_count = 0

    for ep_idx, (ep_path, is_success) in tqdm.tqdm(
        enumerate(episodes), total=len(episodes), desc="  Building"
    ):
        state_array, action_array = load_json_data(ep_path)
        num_frames = len(state_array)

        dataset._video_frame_count = num_frames - 1
        dataset._episode_is_success = is_success

        if is_success:
            success_count += 1
        else:
            fail_count += 1

        for i in range(num_frames - 1):
            frame_data: dict = {
                "face_view": dummy_image,
                "left_wrist_view": dummy_image,
                "right_wrist_view": dummy_image,
                "state": state_array[i],
                "actions": action_array[i + 1],
                "task": task,
            }
            if include_is_success:
                frame_data["is_success"] = np.array([is_success], dtype=bool)
            dataset.add_frame(frame_data)

        dataset.save_episode()

    t2 = time.time()

    # --- PHASE 3: Finalize ---
    dataset.finalize_video_info()

    img_dir = output_path / "images"
    if img_dir.is_dir():
        shutil.rmtree(img_dir)

    print(
        f"  Done: {len(episodes)} episodes "
        f"({success_count} success, {fail_count} fail)"
    )
    print(f"  Transcode: {t1 - t0:.1f}s, Build: {t2 - t1:.1f}s")
    print(f"  Saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: ConvertArgs):
    print("=" * 60)
    print("X2Robot -> LeRobot RECAP Dataset Converter")
    print("=" * 60)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = HF_LEROBOT_HOME
    print(f"Output directory: {output_dir}")

    target_size = (320, 240) if args.low_resolution else (640, 480)

    # ==========================================
    # Build SFT dataset (all success)
    # ==========================================
    if args.sft_raw_paths:
        print(f"\n{'=' * 60}")
        print(f"Building SFT dataset: {args.sft_repo_name}")
        print(f"  Sources: {args.sft_raw_paths}")
        print(f"{'=' * 60}")

        sft_episodes = find_episodes(args.sft_raw_paths)
        if args.debug:
            sft_episodes = sft_episodes[: args.debug_episodes]
        print(f"  Found {len(sft_episodes)} SFT episodes")

        sft_success = sum(1 for _, s in sft_episodes if s)
        sft_fail = sum(1 for _, s in sft_episodes if not s)
        print(f"  (success={sft_success}, fail={sft_fail})")
        if sft_fail > 0:
            print(
                "  WARNING: SFT data contains episodes with _fail suffix. "
                "They will still be included; RECAP Step 1 treats all as "
                "success when type=sft."
            )

        build_dataset(
            episodes=sft_episodes,
            repo_name=args.sft_repo_name,
            output_dir=output_dir,
            task=args.task,
            include_is_success=False,
            target_size=target_size,
            num_workers=args.num_workers,
        )
    else:
        print("\nNo SFT raw paths provided, skipping SFT dataset.")

    # ==========================================
    # Build Rollout dataset (mixed, with is_success)
    # ==========================================
    if args.rollout_raw_paths:
        print(f"\n{'=' * 60}")
        print(f"Building Rollout dataset: {args.rollout_repo_name}")
        print(f"  Sources: {args.rollout_raw_paths}")
        print(f"{'=' * 60}")

        rollout_episodes = find_episodes(args.rollout_raw_paths)
        if args.debug:
            rollout_episodes = rollout_episodes[: args.debug_episodes]
        print(f"  Found {len(rollout_episodes)} Rollout episodes")

        rollout_success = sum(1 for _, s in rollout_episodes if s)
        rollout_fail = sum(1 for _, s in rollout_episodes if not s)
        print(f"  (success={rollout_success}, fail={rollout_fail})")

        build_dataset(
            episodes=rollout_episodes,
            repo_name=args.rollout_repo_name,
            output_dir=output_dir,
            task=args.task,
            include_is_success=True,
            target_size=target_size,
            num_workers=args.num_workers,
        )
    else:
        print("\nNo Rollout raw paths provided, skipping Rollout dataset.")

    # ==========================================
    # Summary
    # ==========================================
    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")
    if args.sft_raw_paths:
        print(f"  SFT dataset:     {output_dir / args.sft_repo_name}")
    if args.rollout_raw_paths:
        print(f"  Rollout dataset: {output_dir / args.rollout_repo_name}")
    print()
    print("Next steps:")
    print("  1. Run RECAP Step 1 (compute_returns) on both datasets")
    print("  2. Run RECAP Step 2 (value_sft) to train the value model")
    print("  3. Run RECAP Step 3 (compute_advantages)")
    print("  4. Run RECAP Step 4 (cfg_sft) for CFG training")


if __name__ == "__main__":
    args = tyro.cli(ConvertArgs)
    main(args)
