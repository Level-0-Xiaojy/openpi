"""LeRobot 数据集加载工具。

只暴露 velocity guider 数据构造需要的接口：
- 列出 episode、读取 master action 14 维、提供视频路径
- 用 PyAV 顺序解码视频帧，返回 ``uint8 [T, H, W, C]`` numpy 数组
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# action[:, 14:28] 是双臂 master action（参考 examples/x2robot/convert_x2robot_data_to_lerobot_v5.py）
MASTER_ACTION_START: int = 14
MASTER_ACTION_END: int = 28

# 三个相机的存储 key（lerobot dataset 内部），与 v5 转换脚本一致
CAMERA_KEYS: tuple[str, ...] = ("face_view", "left_wrist_view", "right_wrist_view")


class LeRobotDatasetInfo:
    """轻量化的 lerobot dataset 元信息读取。

    跳过 ``LeRobotDataset`` 重型加载（它会拉取 hf assets、做 video stats 等），
    直接读 ``meta/info.json`` 和 episode parquet。
    """

    def __init__(self, repo_name: str, root: Path | str):
        self.repo_name = repo_name
        self.root = Path(root) / repo_name
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root}")

        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with info_path.open("r") as f:
            self.info: dict = json.load(f)

        self.fps: int = int(self.info.get("fps", 20))
        self.total_episodes: int = int(self.info["total_episodes"])
        self.total_frames: int = int(self.info["total_frames"])
        self.data_path_template: str = self.info["data_path"]
        self.video_path_template: str = self.info["video_path"]

        for cam in CAMERA_KEYS:
            if cam not in self.info["features"]:
                raise ValueError(
                    f"Camera key '{cam}' not in dataset features. "
                    f"Available: {list(self.info['features'].keys())}"
                )
        cam0 = self.info["features"][CAMERA_KEYS[0]]
        self.video_shape: tuple[int, int, int] = tuple(cam0["shape"])  # (H, W, C)

    def episode_parquet_path(self, ep_idx: int) -> Path:
        chunk = ep_idx // int(self.info.get("chunks_size", 1000))
        return self.root / self.data_path_template.format(
            episode_chunk=chunk, episode_index=ep_idx
        )

    def episode_video_path(self, ep_idx: int, video_key: str) -> Path:
        chunk = ep_idx // int(self.info.get("chunks_size", 1000))
        return self.root / self.video_path_template.format(
            episode_chunk=chunk, episode_index=ep_idx, video_key=video_key
        )

    def list_episodes(self) -> list[int]:
        return list(range(self.total_episodes))

    def get_episode_actions(self, ep_idx: int) -> np.ndarray:
        """返回 ``[T, 28]`` float32 完整 action。"""
        p = self.episode_parquet_path(ep_idx)
        if not p.exists():
            raise FileNotFoundError(f"Episode parquet not found: {p}")
        table = pq.read_table(p, columns=["actions"])
        arr = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 28:
            raise ValueError(
                f"Unexpected action shape for ep {ep_idx}: {arr.shape}; expected [T, 28]"
            )
        return arr

    def get_episode_master_actions(self, ep_idx: int) -> np.ndarray:
        """返回 ``[T, 14]`` float32 双臂 master action。"""
        return self.get_episode_actions(ep_idx)[:, MASTER_ACTION_START:MASTER_ACTION_END]

    def get_episode_length(self, ep_idx: int) -> int:
        p = self.episode_parquet_path(ep_idx)
        return pq.read_metadata(p).num_rows


def decode_video_to_array(video_path: Path) -> np.ndarray:
    """用 PyAV 顺序解码整段视频，返回 ``[T, H, W, C] uint8``。"""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # 让解码器尽可能并发
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8
            frames.append(img)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def iter_episode_image_batches(
    info: LeRobotDatasetInfo,
    ep_idx: int,
    batch_size: int,
    expected_num_frames: int | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """按 batch 读取一个 episode 的三视图。

    为了避免一次性把所有视频帧驻留内存，先把每个相机的整段解码到 ``np.ndarray``（仍要进
    一次内存，但只持续到该 episode 处理完），再按 ``batch_size`` 切片 yield。

    Yields:
        ``(frame_indices [B], images_dict {camera: [B, H, W, C] uint8})``
    """
    frames_per_cam: dict[str, np.ndarray] = {}
    num_frames_each: list[int] = []
    for cam in CAMERA_KEYS:
        vp = info.episode_video_path(ep_idx, cam)
        arr = decode_video_to_array(vp)
        frames_per_cam[cam] = arr
        num_frames_each.append(arr.shape[0])

    # 三个相机的帧数应该一致（采集时同步），不一致就警告并取最短
    n_min = min(num_frames_each)
    n_max = max(num_frames_each)
    if n_max != n_min:
        logger.warning(
            f"Episode {ep_idx} cameras have mismatched frame counts: {num_frames_each}; "
            f"truncating to {n_min}."
        )
    if expected_num_frames is not None and n_min != expected_num_frames:
        logger.warning(
            f"Episode {ep_idx} decoded {n_min} frames but parquet has {expected_num_frames}; "
            f"using min({n_min}, {expected_num_frames})."
        )
        n_min = min(n_min, expected_num_frames)

    for cam in CAMERA_KEYS:
        frames_per_cam[cam] = frames_per_cam[cam][:n_min]

    for start in range(0, n_min, batch_size):
        end = min(start + batch_size, n_min)
        indices = np.arange(start, end, dtype=np.int64)
        batch_imgs = {cam: frames_per_cam[cam][start:end] for cam in CAMERA_KEYS}
        yield indices, batch_imgs
