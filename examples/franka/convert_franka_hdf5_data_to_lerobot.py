"""
A script to convert a directory of .h5 files (from GSDataCollector) 
into the LeRobot dataset format.

This script adapts the logic from the .npy converter to work with the
HDF5 file structure defined in the modified GSDataCollector.

Data processing logic (Target):
- `state`: end-effector pose and gripper width (7D), xyz(3)+euler(3)+gripper(1)
- `actions`: delta end-effector pose and gripper action (7D), delta_xyz(3)+delta_euler(3)+gripper(1)

Usage:
uv run examples/your_project/convert_h5_to_lerobot.py --repo-id "pancake-w/my_h5_dataset" --data-dir "/path/to/my/h5_files"
"""

import os
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1" # disable progress bars for huggingface datasets

import shutil
import tyro
import numpy as np
import glob
import h5py
import cv2
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from typing import Generator, Tuple, Dict, Any, Optional, List
from transforms3d.euler import quat2euler

def find_h5_episode_files(data_dir: str) -> List[str]:
    """
    Find all '.h5' files for episodes in the specified directory.

    Args:
        data_dir: The parent directory containing HDF5 episode files.

    Returns:
        A sorted list containing the full paths to all '.h5' files.
    """
    print(f"Searching for episodes in directory '{data_dir}'...")
    # Find all files ending with .h5 in the root of data_dir
    episode_files = sorted(glob.glob(os.path.expanduser(os.path.join(data_dir, "*.h5"))))
    
    if not episode_files:
        print(f"Warning: No files matching the '*.h5' pattern were found in '{data_dir}'.")
    else:
        print(f"Found {len(episode_files)} episode files.")
        
    return episode_files

def process_episode_h5(h5_path: str) -> Optional[Tuple[str, int, Generator[Dict[str, Any], None, None]]]:
    """
    MODIFIED: Load and process a single episode's .h5 file based on the
    new GSDataCollector structure.

    This function returns a generator that yields processed data frame-by-frame (step)
    to save memory.

    Args:
        h5_path: The path to a single '.h5' file.

    Returns:
        A tuple (instruction, num_steps, frame_generator), 
        or None if the episode is invalid (e.g., cannot be loaded or is empty).
    """
    try:
        f = h5py.File(h5_path, 'r')
    except Exception as e:
        tqdm.write(f"Error: Failed to load {h5_path}. Error message: {e}")
        return None

    try:
        images_h5 = f['image'][:]
        actions_h5 = f['action'][:]
        
        state_ee_pos = f['state/ee/pos'][:]
        state_ee_euler = f['state/ee/euler'][:]
        state_gripper_width = f['state/ee/gripper'][:]
        
        is_image_encode = f.attrs['is_image_encode']
        num_steps = len(actions_h5)

        if num_steps == 0:
            tqdm.write(f"Warning: {h5_path} is an empty episode (based on action length), skipping.")
            f.close()
            return None
            
        # Check consistency
        if not (len(images_h5) == num_steps and len(state_ee_pos) == num_steps):
            tqdm.write(f"Warning: Data mismatch in {h5_path}. Actions: {num_steps}, Images: {len(images_h5)}, States: {len(state_ee_pos)}. Skipping.")
            f.close()
            return None

        # Per user request:
        instruction = "pick the object up"

        def frame_generator():
            """A generator that yields data frame by frame."""
            for step_idx in range(num_steps):
                # 1. Process Image
                img_data = images_h5[step_idx]
                if is_image_encode:
                    # cv2.imdecode reads as RGB
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                else:
                    # Assumed to be (H, W, C) RGB numpy array
                    img = img_data
                
                # 2. Process State (from 'state' group)
                state_pos = state_ee_pos[step_idx]
                state_euler = state_ee_euler[step_idx]
                state_gripper = state_gripper_width[step_idx]
                state_vec = np.concatenate([state_pos, state_euler, state_gripper]).astype(np.float32)

                # 3. Process Action (from action dataset)
                action_full_vec = actions_h5[step_idx]
                if action_full_vec.shape[0] != 7:
                    tqdm.write(f"Warning: Action shape is {action_full_vec.shape}, expected 7D. Skipping frame {step_idx} in {h5_path}.")
                    continue
                    
                action_delta_pos = action_full_vec[0:3]
                action_delta_euler = action_full_vec[3:6]
                action_gripper = action_full_vec[6:]
                
                action_vec = np.concatenate([action_delta_pos, action_delta_euler, action_gripper]).astype(np.float32)

                if state_vec.shape[0] != 7 or action_vec.shape[0] != 7:
                    tqdm.write(f"Error: Dimension mismatch at step {step_idx} in {h5_path}. State: {state_vec.shape}, Action: {action_vec.shape}. Skipping frame.")
                    continue
                    
                yield {
                    "image": img,         # (H, W, C)
                    "state": state_vec,   # (7,)
                    "actions": action_vec, # (7,)
                }
            
            f.close() # Close HDF5 file when generator is exhausted

    except KeyError as e:
        tqdm.write(f"Error: Missing key {e} in {h5_path}. This might be 'state/ee/pos' or similar. Check HDF5 structure. Skipping episode.")
        f.close()
        return None
    except Exception as e:
        tqdm.write(f"Error: Failed to process {h5_path}. Error message: {e}")
        f.close()
        return None

    return instruction, num_steps, frame_generator()

def main(repo_id: str, data_dir: str, *, push_to_hub: bool = False):
    """
    The main conversion function that orchestrates the entire workflow.

    Args:
        repo_id: The identifier for the new LeRobot dataset.
        data_dir: The parent directory containing 'episode_*.h5' files.
        push_to_hub: If True, push the dataset to the Hugging Face Hub.
    """
    # --- 1. Set up output path and clean up old data ---
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset: {output_path}")
        shutil.rmtree(output_path)

    # --- 2. Define and create the LeRobot dataset structure ---
    print("Creating LeRobot dataset structure...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="franka_panda",
        fps=5, # TODO: Adjust FPS if needed
        features={
            # For gr00t (C, H, W)
            "observation.images.image": {
                "names": ["channel", "height", "width"],
                "dtype": "video",
                "shape": (3, 480, 640), # (C, H, W)
            },
            # For openpi (H, W, C)
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            # Removed wrist_image features as requested
            "state": {
                "dtype": "float32", 
                "shape": (7,), 
                "names": ["ee_pose_and_gripper_width"]
            },
            "actions": {
                "dtype": "float32", 
                "shape": (7,), 
                "names": ["delta_ee_pose_and_gripper_action"]
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --- 3. Find all episode files ---
    episode_files = find_h5_episode_files(data_dir)
    if not episode_files:
        print("No valid data found, exiting program.")
        return

    # --- 4. Iterate, process, and write to the LeRobot dataset ---
    for idx, h5_path in enumerate(tqdm(episode_files, desc="Processing Episodes"), 1):
        processed_data = process_episode_h5(h5_path)
        if processed_data is None:
            continue

        instruction, num_steps, frame_generator = processed_data
        
        # Add data frame by frame
        for frame_data in frame_generator:
            # Create (C, H, W) version for 'observation.images.image'
            frame_data["observation.images.image"] = frame_data["image"].transpose(2, 0, 1)
            # Add instruction
            frame_data["task"] = instruction
            
            dataset.add_frame(frame_data)

        dataset.save_episode()
        tqdm.write(f"[{idx}/{len(episode_files)}] Saved episode for task '{instruction}' (with {num_steps} steps)")

    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["franka_panda", "pick-and-place", "robotics", "simulated-data"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    print("\nConversion complete!")
    print(f"LeRobot dataset saved to: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)
    
    