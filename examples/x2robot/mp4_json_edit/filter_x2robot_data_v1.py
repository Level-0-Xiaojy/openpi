"""
X2Robot Data Pre-processor (Filter stationary & pre-intervention frames)

Workflow:
0. Input: X2Robot data (MP4+JSON).
1. Detect Intervention: Find a continuous pause >= 4 seconds. Drop everything before the end of this pause.
2. Filter Stationary: For the remaining frames, drop those where arm/gripper barely moves.
3. Export: Save the filtered data in the exact same MP4+JSON structure for downstream processing.
"""
import os
import glob
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


# --- Configuration ---
# 判定机械臂运动的阈值 (单位通常是米)
POS_THRESHOLD = 0.002 
ROT_THRESHOLD = 0.01
# 判定夹爪运动的阈值
GRIPPER_THRESHOLD = 0.01

# --- Constants based on convert_x2robot_data_to_lerobot_v5.py ---
FPS = 20
PAUSE_SECONDS = 4.0
PAUSE_FRAMES = int(FPS * PAUSE_SECONDS)

VIDEO_FILES =["faceImg.mp4", "leftImg.mp4", "rightImg.mp4"]

def get_state_arrays(data: list) -> dict:
    """提取每一帧的左右臂位置、旋转和夹爪状态用于计算 delta"""
    keys =[
        'follow_left_position', 'follow_left_rotation', 'follow_left_gripper',
        'follow_right_position', 'follow_right_rotation', 'follow_right_gripper'
    ]
    arrays = {k:[] for k in keys}
    for frame in data:
        for k in keys:
            arrays[k].append(frame[k])
    
    for k in keys:
        arrays[k] = np.array(arrays[k], dtype=np.float32)
    return arrays


def detect_intervention_start(arrays: dict) -> int:
    """
    功能 2: 寻找模型Rollout后的静止期，返回人类接管的起始索引。
    如果在数据中找到连续 >= PAUSE_FRAMES (默认80帧) 的静止窗口，
    则认为静止窗口结束的那一帧是人类接管的开始。
    """
    n_frames = len(arrays['follow_left_position'])
    is_moving = np.zeros(n_frames, dtype=bool)

    # 简单计算帧间变化 (Frame-to-frame delta)
    for i in range(1, n_frames):
        l_pos_diff = np.linalg.norm(arrays['follow_left_position'][i] - arrays['follow_left_position'][i-1])
        r_pos_diff = np.linalg.norm(arrays['follow_right_position'][i] - arrays['follow_right_position'][i-1])
        
        # 只要有一只手臂在动，就认为在运动
        if l_pos_diff > POS_THRESHOLD or r_pos_diff > POS_THRESHOLD:
            is_moving[i] = True

    # 寻找最长的静止区间 (连续 is_moving == False 的区间)
    max_pause_len = 0
    current_pause_len = 0
    best_pause_end_idx = 0
    
    for i in range(n_frames):
        if not is_moving[i]:
            current_pause_len += 1
            if current_pause_len > max_pause_len:
                max_pause_len = current_pause_len
                best_pause_end_idx = i
        else:
            current_pause_len = 0

    # 如果找到的最长静止期满足要求
    if max_pause_len >= PAUSE_FRAMES:
        # 接管点是静止期结束后的下一帧
        takeover_idx = min(best_pause_end_idx + 1, n_frames - 1)
        return takeover_idx
    
    # 如果没有找到满足4s的静止期，默认从第0帧开始（即全人工遥操数据）
    return 0


def filter_stationary_frames(arrays: dict, start_idx: int) -> list[int]:
    """
    功能 1: 从 start_idx 开始，剔除不需要的静止帧。
    返回需要保留的帧索引列表。
    """
    n_frames = len(arrays['follow_left_position'])
    if start_idx >= n_frames:
        return[]

    keep_indices = [start_idx]
    
    for i in range(start_idx + 1, n_frames):
        last_kept_idx = keep_indices[-1]
        
        # 计算当前帧与【上一个保留帧】的差异
        l_pos_diff = np.linalg.norm(arrays['follow_left_position'][i] - arrays['follow_left_position'][last_kept_idx])
        r_pos_diff = np.linalg.norm(arrays['follow_right_position'][i] - arrays['follow_right_position'][last_kept_idx])
        
        l_rot_diff = np.linalg.norm(arrays['follow_left_rotation'][i] - arrays['follow_left_rotation'][last_kept_idx])
        r_rot_diff = np.linalg.norm(arrays['follow_right_rotation'][i] - arrays['follow_right_rotation'][last_kept_idx])
        
        l_grip_diff = abs(arrays['follow_left_gripper'][i] - arrays['follow_left_gripper'][last_kept_idx])
        r_grip_diff = abs(arrays['follow_right_gripper'][i] - arrays['follow_right_gripper'][last_kept_idx])

        # 如果位置、旋转或夹爪有任何一个发生了显著变化，保留该帧
        is_pos_moving = (l_pos_diff > POS_THRESHOLD) or (r_pos_diff > POS_THRESHOLD)
        is_rot_moving = (l_rot_diff > ROT_THRESHOLD) or (r_rot_diff > ROT_THRESHOLD)
        is_grip_moving = (l_grip_diff > GRIPPER_THRESHOLD) or (r_grip_diff > GRIPPER_THRESHOLD)
        
        if is_pos_moving or is_rot_moving or is_grip_moving:
            keep_indices.append(i)
            
    # 强制保留最后一帧
    if keep_indices[-1] != n_frames - 1:
        keep_indices.append(n_frames - 1)
        
    return keep_indices


def process_video(input_mp4: str, output_mp4: str, keep_indices: list[int]):
    """根据保留索引抽取视频帧，重写新的MP4"""
    if not os.path.exists(input_mp4):
        print(f"      [Warn] Video not found: {input_mp4}")
        return

    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        print(f"      [Error] Cannot open video: {input_mp4}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps):
        orig_fps = FPS
        
    # 使用 mp4v 编码，这足以作为中间格式，后续脚本A的ffmpeg会重新压成 AV1/x264
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, orig_fps, (width, height))

    current_frame_idx = 0
    keep_set = set(keep_indices)  # 用 set 提高查找速度
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame_idx in keep_set:
            out.write(frame)
            
        current_frame_idx += 1

    cap.release()
    out.release()


def process_episode(episode_dir: str, output_dir: str, enable_takeover_detect: bool, enable_stationary_filter: bool):
    ep_name = os.path.basename(episode_dir)
    json_path = os.path.join(episode_dir, f"{ep_name}.json")
    
    if not os.path.exists(json_path):
        return

    # 1. Load JSON
    with open(json_path, 'r') as f:
        raw_json = json.load(f)
    
    raw_data = raw_json['data']
    total_frames = len(raw_data)
    if total_frames < 2:
        return
        
    arrays = get_state_arrays(raw_data)

    # 2. Detect Takeover (Drop model rollout)
    start_idx = 0
    if enable_takeover_detect:
        start_idx = detect_intervention_start(arrays)
        
    # 3. Filter Stationary
    if enable_stationary_filter:
        keep_indices = filter_stationary_frames(arrays, start_idx)
    else:
        keep_indices = list(range(start_idx, total_frames))
        
    if not keep_indices:
        print(f"  [{ep_name}] All frames filtered out. Skipping.")
        return

    # 4. Save Filtered Data
    # 建立输出目录
    ep_out_dir = os.path.join(output_dir, ep_name)
    os.makedirs(ep_out_dir, exist_ok=True)
    
    # 保存 JSON
    filtered_data =[raw_data[i] for i in keep_indices]
    raw_json['data'] = filtered_data
    out_json_path = os.path.join(ep_out_dir, f"{ep_name}.json")
    with open(out_json_path, 'w') as f:
        json.dump(raw_json, f, indent=4)
        
    # 处理并保存三个视频
    for vid_name in VIDEO_FILES:
        in_vid = os.path.join(episode_dir, vid_name)
        out_vid = os.path.join(ep_out_dir, vid_name)
        process_video(in_vid, out_vid, keep_indices)
        
    # 日志报告
    drop_rollout = start_idx
    drop_stationary = (total_frames - start_idx) - len(keep_indices)
    print(f"  [{ep_name}] Total: {total_frames} -> Kept: {len(keep_indices)} "
          f"(Drop Rollout: {drop_rollout}, Drop Stat: {drop_stationary})")


def main():
    parser = argparse.ArgumentParser(description="Pre-process raw MP4+JSON data before LeRobot conversion.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw episode folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save filtered episode folders.")
    parser.add_argument("--no_takeover_detect", action="store_true", help="Disable the 4s pause takeover detection.")
    parser.add_argument("--no_stationary_filter", action="store_true", help="Disable stationary frame filtering.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    # Find all subdirectories that look like episode folders
    episode_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(episode_dirs)} episodes. Processing...")
    
    for ep_dir in tqdm(episode_dirs):
        process_episode(
            str(ep_dir), 
            str(output_dir), 
            enable_takeover_detect=not args.no_takeover_detect,
            enable_stationary_filter=not args.no_stationary_filter
        )

    print(f"\nProcessing complete! Filtered data saved to: {output_dir}")


if __name__ == "__main__":
    main()