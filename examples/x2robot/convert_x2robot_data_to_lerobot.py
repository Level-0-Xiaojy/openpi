"""Convert X2Robot (ARX) raw episode data to LeRobot dataset format.

Usage:
    uv run examples/x2robot/convert_x2robot_data_to_lerobot.py \
        --raw-dirs /path/to/raw/data \
        --repo-name place_goods \
        --cache-dir /path/to/lerobot_datasets \
        --task "Put the goods on your left into the bag on your right."
"""

import dataclasses
import glob
import os
import shutil

import einops
import numpy as np
import tqdm
import tyro

from openpi.shared.x2robot_tools import ACTION_KEY_MAPPING_INV, decode_video_torchvision, process_action

FILE_CAM_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}

STATE_KEYS = [
    "follow_left_ee_cartesian_pos",
    "follow_left_ee_rotation",
    "follow_left_gripper",
    "follow_right_ee_cartesian_pos",
    "follow_right_ee_rotation",
    "follow_right_gripper",
]

ACTION_KEYS = STATE_KEYS


@dataclasses.dataclass
class Config:
    raw_dirs: list[str]
    """Directories containing episode subdirs (each with mp4s + json)."""

    repo_name: str
    """Output dataset name (also used as HuggingFace repo id)."""

    cache_dir: str = ""
    """HF_LEROBOT_HOME override. Defaults to lerobot's own default if empty."""

    task: str = ""
    """Task description string to store with each frame."""

    fps: int = 20
    robot_type: str = "ARX"

    state_keys: list[str] = dataclasses.field(default_factory=lambda: list(STATE_KEYS))
    action_keys: list[str] = dataclasses.field(default_factory=lambda: list(ACTION_KEYS))

    push_to_hub: bool = False
    max_episodes: int = -1
    """Stop after this many episodes (-1 = no limit)."""


def _discover_episodes(raw_dirs: list[str]) -> list[str]:
    episodes = []
    for d in raw_dirs:
        for entry in sorted(glob.glob(f"{d}/*")):
            if os.path.isdir(entry) and glob.glob(f"{entry}/*.mp4"):
                episodes.append(entry)
    return episodes


def _compose_vector(pose_dicts: dict, keys: list[str]) -> np.ndarray:
    parts = []
    for key in keys:
        arr = pose_dicts[key]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def _detect_video_shape(episode_path: str) -> tuple[int, int, int]:
    for cam_file in FILE_CAM_MAPPING.values():
        path = os.path.join(episode_path, cam_file)
        if os.path.exists(path):
            frames = decode_video_torchvision(path)
            _, c, h, w = frames.shape
            return (h, w, c)
    raise FileNotFoundError(f"No video files found in {episode_path}")


def main(cfg: Config):
    if cfg.cache_dir:
        os.environ["HF_LEROBOT_HOME"] = cfg.cache_dir

    # Delayed import so HF_LEROBOT_HOME is set before lerobot reads it.
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    episodes = _discover_episodes(cfg.raw_dirs)
    if not episodes:
        print(f"No episodes found in {cfg.raw_dirs}")
        return

    video_shape = _detect_video_shape(episodes[0])

    sample_poses = process_action(episodes[0], action_key_mapping=ACTION_KEY_MAPPING_INV)
    state_dim = sum(np.atleast_1d(sample_poses[k][0]).size for k in cfg.state_keys)
    action_dim = sum(np.atleast_1d(sample_poses[k][0]).size for k in cfg.action_keys)

    output_path = HF_LEROBOT_HOME / cfg.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {
        cam_name: {
            "dtype": "video",
            "shape": video_shape,
            "names": ["height", "width", "channel"],
        }
        for cam_name in FILE_CAM_MAPPING
    }
    features["state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["actions"],
    }

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_name,
        robot_type=cfg.robot_type,
        fps=cfg.fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    episode_count = 0
    for episode_path in tqdm.tqdm(episodes, desc="episodes"):
        json_path = os.path.join(episode_path, f"{os.path.basename(episode_path)}.json")
        if not os.path.exists(json_path):
            print(f"Skipping {episode_path}: no JSON found")
            continue

        pose_dicts = process_action(episode_path, action_key_mapping=ACTION_KEY_MAPPING_INV)
        states = _compose_vector(pose_dicts, cfg.state_keys)
        actions = _compose_vector(pose_dicts, cfg.action_keys)

        video_frames = {}
        for cam_name in FILE_CAM_MAPPING:
            video_path = os.path.join(episode_path, FILE_CAM_MAPPING[cam_name])
            frames = decode_video_torchvision(video_path)
            video_frames[cam_name] = einops.rearrange(frames, "t c h w -> t h w c")

        n_frames = len(states) - 1
        for i in range(n_frames):
            frame = {"state": states[i], "actions": actions[i + 1], "task": cfg.task}
            for cam_name in FILE_CAM_MAPPING:
                frame[cam_name] = video_frames[cam_name][i]
            dataset.add_frame(frame)

        dataset.save_episode()
        episode_count += 1
        if cfg.max_episodes > 0 and episode_count >= cfg.max_episodes:
            break

    print(f"Saved {episode_count} episodes to {HF_LEROBOT_HOME}/{cfg.repo_name}")


if __name__ == "__main__":
    tyro.cli(main)
