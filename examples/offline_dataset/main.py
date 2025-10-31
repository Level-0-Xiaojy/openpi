# # Monkey-patch to fix 'List' feature type error in old datasets
# try:
#     import datasets.features.features as features

#     _OLD_GENERATE_FROM_DICT = features.generate_from_dict

#     def _new_generate_from_dict(obj):
#         if isinstance(obj, dict) and obj.get("_type") == "List":
#             obj["_type"] = "Sequence"
#         return _OLD_GENERATE_FROM_DICT(obj)

#     features.generate_from_dict = _new_generate_from_dict
# except (ImportError, AttributeError):
#     # If datasets or the function doesn't exist, do nothing.
#     pass
# # End of monkey-patch


import collections
import dataclasses
import logging
import math
import pathlib
import matplotlib.pyplot as plt

import imageio
import numpy as np
from lerobot.common.datasets import lerobot_dataset
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from PIL import Image

KEY_MAPPINGS = {
    'droid_100': {
        'image': 'observation/exterior_image_1_left',
        'wrist_image': 'observation/wrist_image_left',
        'state': ['observation/joint_position', 'observation/gripper_position'],
        'action': 'action',
    }
}


def create_plot_array(gt_actions, pred_actions, states, figsize=None):
    """
    Create a matplotlib figure as a numpy array showing states and actions.
    
    Args:
        gt_actions: List or array of ground truth actions
        pred_actions: List or array of predicted actions  
        states: List or array of states
        figsize: Tuple of (width, height) for the figure
        
    Returns:
        numpy array representing the plot as RGB image
    """
    if len(gt_actions) == 0:
        return None
        
    gt_array = np.array(gt_actions)
    pred_array = np.array(pred_actions)
    state_array = np.array(states)
    
    # Determine the number of subplots needed
    state_dim = state_array.shape[1] if len(state_array.shape) > 1 else 1
    action_dim = gt_array.shape[1] if len(gt_array.shape) > 1 else 1
    n_subplots = max(state_dim, action_dim)
    
    figsize = figsize or (5 * n_subplots, 3)
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]
    
    time_steps = range(len(gt_actions))
    
    for i in range(n_subplots):
        ax = axes[i]
        
        # Plot states if dimension exists
        if i < state_dim:
            if len(state_array.shape) > 1:
                ax.plot(time_steps, state_array[:, i], label=f'state_{i}', alpha=0.7)
            else:
                ax.plot(time_steps, state_array, label='state', alpha=0.7)
        
        # Plot actions if dimension exists
        if i < action_dim:
            if len(gt_array.shape) > 1:
                ax.plot(time_steps, gt_array[:, i], label=f'gt_action_{i}', linestyle='--')
                ax.plot(time_steps, pred_array[:, i], label=f'pred_action_{i}', linestyle='-.')
            else:
                ax.plot(time_steps, gt_array, label='gt_action', linestyle='--')
                ax.plot(time_steps, pred_array, label='pred_action', linestyle='-.')
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Dimension {i}')
    
    plt.tight_layout()
    plt.savefig("temp_plot.png")
    
    # Convert figure to numpy array
    fig.canvas.draw()
    plot_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # import pdb; pdb.set_trace()
    plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., 1:]  # Convert ARGB to RGB
    plt.close()

    # import pdb; pdb.set_trace()
    
    return plot_array


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    repo_id: str = ""
    action_horizon: int = 50  # Number of actions to predict at each time step
    action_key: str = "actions"  # Key for the action in the dataset
    episode_id: int = 5  # Episode ID to run

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/videos"  # Path to save videos

    seed: int = 42  # Random Seed (for reproducibility)


def main(args: Args) -> None:
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(args.action_horizon)] for key in [args.action_key]
        },
        episodes=[args.episode_id],
    )
    logging.info(f"Loaded episode {args.episode_id} with {len(dataset)} steps.")

    action_buffer = collections.deque(maxlen=args.replan_steps)

    plot_info = {
        'gt': [],
        'pred': [],
        'state': [],
    }
    replay_images = []

    episode_length = len(dataset)
    all_timestamps = [data['timestamp'].numpy() for data in dataset]
    for step_idx in tqdm.tqdm(range(episode_length), desc="Running episode"):
        data = dataset[step_idx]
        msg = {
            "observation/image": data['observation/exterior_image_1_left'].numpy(),
            "observation/wrist_image": data['observation/wrist_image_left'].numpy(),
            "observation/state": data['state'].numpy(),
            "prompt": "pick up the object",
        }
        gt_action = data[args.action_key].numpy()[0]
        if len(action_buffer) == 0:
            actions = client.infer(msg)["actions"]
            for i in range(action_buffer.maxlen):
                action_buffer.append(actions[i])
        action = action_buffer.popleft()

        plot_info['gt'].append(gt_action)
        plot_info['pred'].append(action)
        plot_info['state'].append(data['state'].numpy())

        image = image_tools.convert_to_uint8(data['image'].numpy().transpose(1, 2, 0))
        wrist_image = image_tools.convert_to_uint8(data['wrist_image'].numpy().transpose(1, 2, 0))
        combined_image = np.concatenate([image, wrist_image], axis=1)

        # Create plot array showing states and actions accumulated so far
        plot_array = create_plot_array(plot_info['gt'], plot_info['pred'], plot_info['state'])
        plot_img = Image.fromarray(plot_array)
        # reshape img to make it the same width as combined_image
        plot_img = plot_img.resize((combined_image.shape[1], plot_img.height * combined_image.shape[1] // plot_img.width))
        plot_array = np.array(plot_img)
        combined_image = np.concatenate([combined_image, plot_array], axis=0)
        replay_images.append(combined_image)

        # Now you can use plot_array with other images
        # For example, you could append it to replay_images or combine with image/wrist_image
        if plot_array is not None:
            # Example: combine all three images horizontally (you may need to resize them first)
            # combined_image = np.concatenate([image, wrist_image, plot_array], axis=1)
            pass

        logging.info(f"Step {step_idx}: error={np.linalg.norm(gt_action - action):.4f}")

    # plot and save
    plot_info['gt'] = np.array(plot_info['gt'])
    plot_info['pred'] = np.array(plot_info['pred'])
    plt.figure(figsize=(8, 3 * plot_info['gt'].shape[1]))
    for i in range(plot_info['gt'].shape[1]):
        plt.subplot(plot_info['gt'].shape[1], 1, i+1)
        plt.plot(plot_info['gt'][:, i], label='gt')
        plt.plot(plot_info['pred'][:, i], label='pred')
        plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Action value')
    plt.tight_layout()
    plt.savefig(pathlib.Path(args.video_out_path) / f"episode_{args.episode_id}_actions.png")
    plt.close()

    states = np.array(plot_info['state'])
    plt.figure(figsize=(8, 3 * states.shape[1]))
    for i in range(states.shape[1]):
        plt.subplot(states.shape[1], 1, i+1)
        plt.plot(states[:, i])
    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.tight_layout()
    plt.savefig(pathlib.Path(args.video_out_path) / f"episode_{args.episode_id}_states.png")
    plt.close()

    # plot states and pred actions together
    plt.figure(figsize=(8, 3 * states.shape[1]))
    for i in range(min(states.shape[1], plot_info['pred'].shape[1])):
        plt.subplot(min(states.shape[1], plot_info['pred'].shape[1]), 1, i+1)
        plt.plot(states[:, i], label='state')
        plt.plot(plot_info['pred'][:, i], label='pred_action')
        plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(pathlib.Path(args.video_out_path) / f"episode_{args.episode_id}_states_actions.png")
    plt.close()

    # Save a replay video of the episode
    imageio.mimwrite(
        pathlib.Path(args.video_out_path) / f"rollout_franka_ep{args.episode_id}.mp4",
        [np.asarray(x) for x in replay_images],
        fps=1,
    )

    print("Saving videos to ", args.video_out_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
