import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from openpi.training.config import TrainConfig
from openpi.training.data_loader import create_torch_dataset
import openpi.policies.policy as _policy

def calc_mse_and_check_via_single_trajectory_horizon(
    policy: _policy.Policy,
    config: TrainConfig, 
    steps: int = 50,
    plot: bool = True,
    episode_index: int = 0,
    instruction: str = "Grasp the chili and place it into the bowl.",
):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and ground truth actions
    for a single trajectory from a specified dataset and optionally plots the comparison.
    
    This version correctly handles action_horizon by predicting once and then executing
    the entire horizon before making the next prediction.

    Args:
        policy: The trained policy to use for inference.
        config: The training configuration containing data and model settings.
        steps: The number of steps to evaluate in the trajectory.
        plot: Whether to plot the predicted vs. ground truth actions.
        episode_index: The index of the episode to evaluate.
        instruction: The instruction prompt for the policy.
    """

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    action_horizon = config.model.action_horizon

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
    print(f"Action horizon: {action_horizon}")

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

    current_step = 0
    prediction_count = 0
    
    while current_step < min(steps, len(trajectory)):
        # Make prediction at current step
        result = policy.infer(get_observation(trajectory[current_step]))
        predicted_actions = result["actions"]  # Shape: (action_horizon, action_dim)
        
        prediction_count += 1
        print(f"Prediction {prediction_count} at step {current_step}")
        
        # Execute the entire action horizon or until we reach the end
        horizon_end = min(current_step + action_horizon, min(steps, len(trajectory)))
        
        for horizon_step in range(current_step, horizon_end):
            # Get ground truth action
            gt_action = trajectory[horizon_step]['actions']
            gt_actions_across_time.append(gt_action)
            
            # Use the corresponding predicted action from the horizon
            horizon_index = horizon_step - current_step
            if horizon_index < predicted_actions.shape[0]:
                pred_action = predicted_actions[horizon_index]
            else:
                # If we run out of predicted actions, use the last one
                pred_action = predicted_actions[-1]
            
            pred_actions_across_time.append(pred_action)
        
        # Move to next prediction point
        current_step = horizon_end

    gt_actions_across_time = np.array(gt_actions_across_time)
    pred_actions_across_time = np.array(pred_actions_across_time)
    
    print(f"GT actions shape: {gt_actions_across_time.shape}")
    print(f"Pred actions shape: {pred_actions_across_time.shape}")
    
    # Handle different shapes for GT and predicted actions
    if len(gt_actions_across_time.shape) == 3 and len(pred_actions_across_time.shape) == 2:
        # GT has shape (steps, horizon, dim), pred has shape (steps, dim)
        # We need to compare the first action of each horizon with our prediction
        gt_actions_for_comparison = gt_actions_across_time[:, 0, :]  # Take first action of each horizon
        pred_actions_for_comparison = pred_actions_across_time
    elif len(gt_actions_across_time.shape) == 2 and len(pred_actions_across_time.shape) == 2:
        # Both have shape (steps, dim)
        gt_actions_for_comparison = gt_actions_across_time
        pred_actions_for_comparison = pred_actions_across_time
    else:
        # Handle other cases - squeeze both to ensure compatibility
        gt_actions_for_comparison = gt_actions_across_time.squeeze()
        pred_actions_for_comparison = pred_actions_across_time.squeeze()
    
    print(f"GT actions for comparison shape: {gt_actions_for_comparison.shape}")
    print(f"Pred actions for comparison shape: {pred_actions_for_comparison.shape}")
    
    print(f"Total predictions made: {prediction_count}")
    print(f"Total steps evaluated: {len(gt_actions_for_comparison)}")
    
    # Calculate the Mean Squared Error across the trajectory
    mse = np.mean((gt_actions_for_comparison - pred_actions_for_comparison) ** 2)
    print(f"Action MSE for trajectory {episode_index}: {mse}")
    
    # Get action dimension
    if len(gt_actions_for_comparison.shape) == 2:
        action_dim = gt_actions_for_comparison.shape[-1]
    else:
        action_dim = 1 if len(gt_actions_for_comparison.shape) == 1 else gt_actions_for_comparison.shape[-1]
    
    print(f"Action Dimension: {action_dim}")

    if plot:
        fig, axes = plt.subplots(
            nrows=action_dim, ncols=1, figsize=(12, 3 * action_dim)
        )
        fig.suptitle(f"Trajectory {episode_index} - Action Horizon Execution ({prediction_count} predictions)", 
                    fontsize=16, color="blue")

        # Mark prediction points
        prediction_points = list(range(0, len(gt_actions_for_comparison), action_horizon))
        
        for i in range(action_dim):
            ax = axes[i] if action_dim > 1 else axes
            
            if len(gt_actions_for_comparison.shape) == 2:
                gt_data = gt_actions_for_comparison[:, i]
                pred_data = pred_actions_for_comparison[:, i]
            else:
                gt_data = gt_actions_for_comparison
                pred_data = pred_actions_for_comparison
            
            ax.plot(gt_data, label="Ground Truth Action", linewidth=2)
            ax.plot(pred_data, label="Predicted Action", linestyle="--", linewidth=2)
            
            # Mark prediction points
            for pred_point in prediction_points:
                if pred_point < len(gt_data):
                    ax.axvline(x=pred_point, color='red', linestyle=':', alpha=0.7, linewidth=1)
            
            ax.set_title(f"Action Dimension {i}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Action Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Add text annotation about prediction points
        fig.text(0.5, 0.02, f"Red dotted lines indicate prediction points (every {action_horizon} steps)", 
                ha='center', fontsize=10, style='italic')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        output_filename = f"action_horizon_trajectory_{episode_index}.jpeg"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Annotated image saved as {output_filename}")
        plt.close()

    return mse