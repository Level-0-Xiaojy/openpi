"""Utilities for processing X2Robot (ARX) raw episode data."""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision


ACTION_KEY_MAPPING = {
    "follow_right_arm_joint_pos": "follow_right_joint_pos",
    "follow_right_arm_joint_dev": "follow_right_joint_dev",
    "follow_right_arm_joint_cur": "follow_right_joint_cur",
    "follow_right_ee_cartesian_pos": "follow_right_position",
    "follow_right_ee_rotation": "follow_right_rotation",
    "follow_right_gripper": "follow_right_gripper",
    "master_right_arm_joint_pos": "master_right_joint_pos",
    "master_right_arm_joint_dev": "master_right_joint_dev",
    "master_right_arm_joint_cur": "master_right_joint_cur",
    "master_right_ee_cartesian_pos": "master_right_position",
    "master_right_ee_rotation": "master_right_rotation",
    "master_right_gripper": "master_right_gripper",
    "follow_left_arm_joint_pos": "follow_left_joint_pos",
    "follow_left_arm_joint_dev": "follow_left_joint_dev",
    "follow_left_arm_joint_cur": "follow_left_joint_cur",
    "follow_left_ee_cartesian_pos": "follow_left_position",
    "follow_left_ee_rotation": "follow_left_rotation",
    "follow_left_gripper": "follow_left_gripper",
    "master_left_arm_joint_pos": "master_left_joint_pos",
    "master_left_arm_joint_dev": "master_left_joint_dev",
    "master_left_arm_joint_cur": "master_left_joint_cur",
    "master_left_ee_cartesian_pos": "master_left_position",
    "master_left_ee_rotation": "master_left_rotation",
    "master_left_gripper": "master_left_gripper",
    "master_left_joint_pos": "master_left_joint_pos",
    "master_right_joint_pos": "master_right_joint_pos",
    "base_movement": "base_movement",
    "car_pose": "car_pose",
    "head_actions": "head_rotation",
    "height": "lifting_mechanism_position",
}

ACTION_KEY_MAPPING_INV = {v: k for k, v in ACTION_KEY_MAPPING.items()}


def decode_video_torchvision(file_name: str, keyframes_only: bool = True, backend: str = "pyav") -> np.ndarray:
    """Decode video using torchvision.io.VideoReader. Returns (T, C, H, W) uint8 array."""
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(file_name, "video")
    reader.seek(0, keyframes_only=keyframes_only)

    loaded_frames = []
    for frame in reader:
        loaded_frames.append(frame["data"])

    reader.container.close()
    return torch.stack(loaded_frames).numpy()


def process_action(
    file_path: str,
    action_key_mapping: dict[str, str] = ACTION_KEY_MAPPING,
    filter_angle_outliers: bool = True,
) -> dict[str, np.ndarray]:
    """Load action JSON from an episode directory and return mapped trajectory arrays."""
    file_name = os.path.basename(file_path)
    action_path = os.path.join(file_path, f"{file_name}.json")

    trajectories: dict[str, list] = defaultdict(list)

    with open(action_path) as file:
        actions = json.load(file)
        for action in actions["data"]:
            for key, val in action.items():
                new_key = action_key_mapping.get(key)
                if new_key is not None:
                    trajectories[new_key].append(val)

    result = {k: np.array(v, dtype=np.float32) for k, v in trajectories.items()}

    if filter_angle_outliers:
        result = smooth_action(result)

    return result


def smooth_action(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply EMA smoothing to rotation trajectories to filter angle outliers."""

    def _filter(traj: np.ndarray, threshold: float = 3, alpha: float = 0.05, window: int = 10) -> np.ndarray:
        orig_dtype = traj.dtype
        data = pd.Series(traj)
        derivatives = np.diff(data)

        spike_indices = np.where(abs(derivatives) > threshold)[0]
        if len(spike_indices) > 0:
            ema = data.ewm(alpha=alpha, adjust=True).mean()
            start_idx = max(0, spike_indices[0] - window)
            end_idx = min(len(data), spike_indices[-1] + window + 1)
            modified_seg = ema.iloc[start_idx:end_idx]
            if len(modified_seg) > 0:
                data.iloc[start_idx:end_idx] = modified_seg.values.astype(orig_dtype)

        return data.to_numpy().astype(orig_dtype)

    for key in ["follow_right_ee_rotation", "follow_left_ee_rotation"]:
        if key in action:
            orig_dtype = action[key].dtype
            filtered_traj = np.stack([_filter(action[key][:, i]) for i in range(3)], axis=1)
            if not np.isnan(filtered_traj).any():
                action[key] = filtered_traj.astype(orig_dtype)

    return action
