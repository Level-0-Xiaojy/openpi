import torch
import numpy as np
import matplotlib.pyplot as plt
import openpi.policies.policy as _policy
from openpi.training.config import TrainConfig, LeRobotFrankaEEDataConfig
from openpi.training.data_loader import create_torch_dataset, TransformedDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def lerobot_dataset_get_step_data(dataset: LeRobotDataset, traj_id, step_cnt):
    if isinstance(dataset, LeRobotDataset):
        # end_of_episode = dataset.episode_data_index['to'][traj_id]
        start_of_traj = dataset.episode_data_index['from'][traj_id]
    elif isinstance(dataset, TransformedDataset):
        # end_of_episode = dataset._dataset.episode_data_index['to'][traj_id]
        start_of_traj = dataset._dataset.episode_data_index['from'][traj_id]
    else:
        raise ValueError("Dataset type not recognized. Expected LeRobotDataset or TransformedDataset.")
    global_idx = start_of_traj + step_cnt
    return dataset[global_idx.item()]

def get_uint8_image(image):
    return (image.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

def calc_mse_for_single_trajectory(
    policy: _policy.Policy,
    config: TrainConfig, 
    traj_id: int,
    steps=150,
    action_horizon=16,
    plot = True,
    plot_state = False,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    action_horizon = config.model.action_horizon

    for step_cnt in range(steps):
        # traj_id: episode_index, step_cnt: frame_index
        data_point = lerobot_dataset_get_step_data(dataset,traj_id,step_cnt) 
        concat_state = data_point["state"]          # [7,]
        concat_gt_action = data_point["actions"][0] # [action_horizon, 7][0]
        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)

        if step_cnt % action_horizon == 0:
            print("inferencing at step: ", step_cnt)
            obs = LeRobotFrankaEEDataConfig.generate_observations(
                    data_point['image'], 
                    data_point['wrist_image'], 
                    data_point['state'], 
                    data_point['task'], # you should use 'base_config=DataConfig(prompt_from_task=True,)' to using data_point['task'] during training
                ) 
            result = policy.infer(obs)
            predicted_action_chunk = result["actions"]  # Shape: (action_horizon, action_dim)
            for j in range(action_horizon):
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.atleast_1d(predicted_action_chunk[j])
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
            f"Trajectory {traj_id} - Action horizon {action_horizon}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
            # Therefore, do not plot them if this is not the case.
            if plot_state and state_joints_across_time.shape == gt_action_across_time.shape:
                ax.plot(state_joints_across_time[:, i], label="gt state")
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
        plt.savefig(f"trajectory_{traj_id}_actions.jpeg")
        plt.close()
        # plt.show()

    return mse
