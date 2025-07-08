import dataclasses
import jax
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import franka_policy
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.config import TrainConfig
from openpi.training.data_loader import create_torch_dataset
import openpi.policies.policy as _policy
import torch


def calc_mse_and_check_via_single_trajectory(
    policy: _policy.Policy,
    config: TrainConfig, 
    steps: int = 50,
    plot: bool = False,
    episode_index: int = 0,
    instruction: str = "Grasp the chili and place it into the bowl.",
):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and ground truth actions
    for a single trajectory from a specified dataset and optionally plots the comparison.

    Args:
        policy: The trained policy to use for inference.
        dataset_name: The name of the dataset to use for evaluation.
        traj_id: The index of the trajectory within the dataset to evaluate.
        steps: The number of steps to evaluate in the trajectory.
        plot: Whether to plot the predicted vs. ground truth actions.
    """

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model) # use lerobot dataset here.
    # you can just use hugging face data via `hf_dataset = load_dataset("parquet", data_files=files, split="train")`

    trajectory = []

    for i in tqdm(range(len(dataset)), desc=f"Filtering trajectory {episode_index}"):
        data_point = dataset[i]
        if data_point['episode_index'] > episode_index:
            break
        if data_point['episode_index'] == episode_index:
            trajectory.append(data_point)
    
    if not trajectory:
        print(f"Warning: No data found in the dataset for episode_index = {episode_index}.")
        return 0.0


    print(f"Found {len(trajectory)} frames belonging to trajectory {episode_index}. Starting MSE calculation...")


    gt_actions_across_time = []
    pred_actions_across_time = []

    def get_uint8_image(image):
        return (image.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    def get_observation(data_point):
        return {
            "observation/image": get_uint8_image(data_point['image']),
            "observation/wrist_image": get_uint8_image(data_point['wrist_image']),
            "observation/state": data_point['state'],
            "prompt": instruction,
        }

    for step_count in range(min(steps, len(trajectory))):
        # Get the observation at the current step
        
        gt_action = trajectory[step_count]['actions']
        gt_actions_across_time.append(gt_action)

        # Infer the action from the policy
        result = policy.infer(get_observation(trajectory[step_count]))
        pred_action = result["actions"]
        pred_actions_across_time.append(pred_action)

    gt_actions_across_time = np.array(gt_actions_across_time).squeeze()
    pred_actions_across_time = np.array(pred_actions_across_time).squeeze()

    assert gt_actions_across_time.shape == pred_actions_across_time.shape

    # Calculate the Mean Squared Error across the trajectory

    mse = np.mean((gt_actions_across_time - pred_actions_across_time) ** 2)
    print(f"Action MSE for trajectory {episode_index}: {mse}")

    action_dim = gt_actions_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(
            nrows=action_dim, ncols=1, figsize=(10, 3 * action_dim)
        )
        fig.suptitle(f"Trajectory {episode_index}", fontsize=16, color="blue")

        for i in range(action_dim):
            ax = axes[i] if action_dim > 1 else axes
            ax.plot(gt_actions_across_time[:, i], label="Ground Truth Action")
            ax.plot(pred_actions_across_time[:, i], label="Predicted Action", linestyle="--")
            ax.set_title(f"Action Dimension {i}")
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return mse

