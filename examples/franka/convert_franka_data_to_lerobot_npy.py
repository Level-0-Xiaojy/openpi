"""
A script to convert a directory of .npy files into the LeRobot dataset format.

This script encapsulates data lookup and processing logic into separate functions,
making the code cleaner and easier to maintain.

Data processing logic:
- `state`: end-effector pose and gripper width (7D), xyz(3)+euler(3)+gripper(1)
- `actions`: delta end-effector pose and gripper action (7D), delta_xyz(3)+delta_euler(3)+gripper(1)

Usage:
uv run examples/franka/convert_npy_to_lerobot.py --repo-id "pancake-w/test_npy" --data-dir "/nvme_data/bingwen/share_datasets/franka_panda/pick_to_plate-real"
"""

import os
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1" # disable progress bars for huggingface datasets

import shutil
import tyro
import numpy as np
import glob
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from typing import Generator, Tuple, Dict, Any, Optional, List
from transforms3d.euler import quat2euler, euler2quat
def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []

    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        # concat_state = np.concatenate(
        #     [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        # )
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0:
            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(8, 4 * action_dim))

        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
            # Therefore, do not plot them if this is not the case.
            if state_joints_across_time.shape == gt_action_across_time.shape:
                ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_across_time[:, i], label="gt action")
            ax.plot(pred_action_across_time[:, i], label="pred action")

            # put a dot every ACTION_HORIZON
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_across_time[j, i], "ro")

            ax.set_title(f"Action {i}")
            ax.legend()

        plt.tight_layout()
        plt.show()

    return mse


def find_episode_files(data_dir: str) -> List[str]:
    """
    Find all 'data.npy' files for episodes in the specified directory.

    Args:
        data_dir: The parent directory containing 'episode_*' folders.

    Returns:
        A sorted list containing the full paths to all 'data.npy' files.
    """
    print(f"Searching for episodes in directory '{data_dir}'...")
    episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*", "data.npy")))
    
    if not episode_files:
        print(f"Warning: No files matching the 'episode_*/data.npy' pattern were found in '{data_dir}'.")
    else:
        print(f"Found {len(episode_files)} episode files.")
        
    return episode_files

def process_episode(npy_path: str) -> Optional[Tuple[str, int, Generator[Dict[str, Any], None, None]]]:
    """
    Load and process a single episode's .npy file.

    This function returns a generator that yields processed data frame-by-frame (step)
    to save memory.

    Args:
        npy_path: The path to a single 'data.npy' file.

    Returns:
        A tuple (instruction, num_steps, frame_generator), 
        or None if the episode is invalid (e.g., cannot be loaded or is empty).
    """
    try:
        episode_data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        tqdm.write(f"Error: Failed to load {npy_path}. Error message: {e}")
        return None

    num_steps = len(episode_data['observation']['rgb'])
    if num_steps == 0:
        tqdm.write(f"Warning: {os.path.dirname(npy_path)} is an empty episode, skipping.")
        return None

    # Extract and format the task instruction
    instruction_raw = episode_data['task_info']['instruction']
    if isinstance(instruction_raw, np.ndarray):
        instruction = instruction_raw[0].strip()
    else:
        instruction = str(instruction_raw).strip()

    # TODO(bingwen) trick
    if instruction == "test":
        instruction = "Grasp the chili and place it into the bowl."

    def frame_generator():
        """A generator that yields data frame by frame."""
        for step_idx in range(num_steps):
            # State
            ee_state_data = episode_data['state']['end_effector']
            state_pos = ee_state_data['position'][step_idx]
            state_ori = ee_state_data['orientation'][step_idx]
            state_euler = quat2euler(state_ori, axes='sxyz')
            state_gripper = np.array([ee_state_data['gripper_width'][step_idx]])
            state_vec = np.concatenate([state_pos, state_euler, state_gripper]).astype(np.float32)

            # Action
            ee_action_data = episode_data['action']['end_effector']
            action_delta_pos = ee_action_data['delta_position'][step_idx]
            action_delta_ori = ee_action_data['delta_orientation'][step_idx]
            action_delta_euler = quat2euler(action_delta_ori, axes='sxyz') # Convert quaternion to Euler angles
            action_gripper = np.array([ee_action_data['gripper_width'][step_idx]])
            action_vec = np.concatenate([action_delta_pos, action_delta_euler, action_gripper]).astype(np.float32)

            if state_vec.shape[0] != 7 or action_vec.shape[0] != 7:
                # tqdm.write(f"Error: Dimension mismatch at step {step_idx} in {os.path.dirname(npy_path)}. Skipping this frame.")
                continue
                
            yield {
                "image": episode_data['observation']['rgb'][step_idx, 1], # TODO(bingwen) to check
                "wrist_image": episode_data['observation']['rgb'][step_idx, 0],
                "state": state_vec, # 7,
                "actions": action_vec, # 7,
            }

    return instruction, num_steps, frame_generator()

def main(repo_id: str, data_dir: str, *, push_to_hub: bool = False):
    """
    The main conversion function that orchestrates the entire workflow.

    Args:
        repo_id: The identifier for the new LeRobot dataset.
        data_dir: The parent directory containing 'episode_*' folders.
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
        fps=5,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
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
    episode_files = find_episode_files(data_dir)
    if not episode_files:
        print("No valid data found, exiting program.")
        return

    # --- 4. Iterate, process, and write to the LeRobot dataset ---
    for idx, npy_path in enumerate(tqdm(episode_files, desc="Processing Episodes"), 1):
        processed_data = process_episode(npy_path)
        if processed_data is None:
            continue

        instruction, num_steps, frame_generator = processed_data        
        # Add data frame by frame
        for frame_data in frame_generator:
            frame_data["task"] = instruction
            dataset.add_frame(frame_data)

        dataset.save_episode()
        tqdm.write(f"[{idx}/{len(episode_files)}] Saved episode for task '{instruction}' (with {num_steps} steps)")

    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["franka_panda", "pick-and-place", "robotics"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    print("\nConversion complete!")
    print(f"LeRobot dataset saved to: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)
