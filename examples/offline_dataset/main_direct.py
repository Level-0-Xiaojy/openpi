"""
Direct policy inference on offline datasets without websocket wrapper.
This script loads the policy directly and runs inference for easier debugging.
"""

import collections
import dataclasses
import enum
import logging
import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets import lerobot_dataset
from openpi_client import image_tools
from PIL import Image
import tqdm
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""
    pass


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


KEY_MAPPINGS = {
    'droid/droid_small': {
        'observation/exterior_image_1_left': 'exterior_image_1_left',
        'observation/wrist_image_left': 'wrist_image_left',
        'observation/joint_position': 'joint_position',
        'observation/gripper_position': 'gripper_position',
        'actions': 'actions',
        'prompt': 'task',
    },
    'physical-intelligence/libero': {
        'observation/image': 'image',
        'observation/wrist_image': 'wrist_image',
        'observation/state': 'state',
        'actions': 'actions',
        'prompt': 'task',
    }
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Policy parameters
    #################################################################################################################
    # Environment to serve the policy for (used when loading default policies).
    env: EnvMode = EnvMode.LIBERO
    
    # If provided, will be used in case the "prompt" key is not present in the data.
    default_prompt: str | None = None
    
    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
    pytorch_device: str | None = None

    #################################################################################################################
    # Inference parameters
    #################################################################################################################
    replan_steps: int = 1  # Number of steps before replanning
    resize_size: int = 224  # Image resize size

    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    repo_id: str = "droid/droid_small"
    action_horizon: int = 50  # Number of actions to predict at each time step
    action_key: str = "actions"  # Key for the action in the dataset
    episode_id: int = 0  # Episode ID to run

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/videos"  # Path to save videos
    seed: int = 42  # Random Seed (for reproducibility)


def process_data_to_msg(data, key_mapping):
    """Process dataset data into the format expected by the policy."""
    msg = {}
    for key, value in key_mapping.items():
        if isinstance(value, list):
            # msg[key] = np.concatenate([data[k].numpy() for k in value], axis=-1)
            data_list = []
            for v in value:
                if data[v].numpy().ndim == 0:
                    data_list.append(data[v].numpy()[None])
                else:
                    data_list.append(data[v].numpy())
            msg[key] = np.concatenate(data_list, axis=-1)
        elif isinstance(data[value], str):
            msg[key] = data[value]
        else:
            msg[key] = data[value].numpy()
    return msg


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None, pytorch_device: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), 
            checkpoint.dir, 
            default_prompt=default_prompt,
            pytorch_device=pytorch_device
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if isinstance(args.policy, Checkpoint):
        return _policy_config.create_trained_policy(
            _config.get_config(args.policy.config), 
            args.policy.dir, 
            default_prompt=args.default_prompt,
            pytorch_device=args.pytorch_device
        )
    else:  # Default
        return create_default_policy(args.env, default_prompt=args.default_prompt, pytorch_device=args.pytorch_device)


def main(args: Args) -> None:
    np.random.seed(args.seed)
    
    # Create output directory
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # Load the policy directly (no websocket)
    logging.info("Loading policy...")
    policy = create_policy(args)
    logging.info(f"Policy loaded successfully. Metadata: {policy.metadata}")
    
    # Load dataset
    logging.info(f"Loading dataset {args.repo_id}...")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(args.action_horizon)] for key in [args.action_key]
        },
        episodes=[args.episode_id],
    )
    logging.info(f"Loaded episode {args.episode_id} with {len(dataset)} steps.")
    # Get key mappings for this dataset
    key_mapping = KEY_MAPPINGS[args.repo_id]
    
    # Action buffer for replanning
    action_buffer = collections.deque(maxlen=args.replan_steps)

    # Data for plotting
    plot_info = {
        'gt': [],
        'pred': [],
    }
    replay_images = []

    episode_length = len(dataset)
    for step_idx in tqdm.tqdm(range(episode_length), desc="Running episode"):
        data = dataset[step_idx]
        
        # Process data into policy input format
        msg = process_data_to_msg(data, key_mapping)
        gt_action = data[args.action_key].numpy()[0]
        
        # Get action from policy (replan when buffer is empty)
        if len(action_buffer) == 0:
            # Call policy.infer directly (no websocket)
            output = policy.infer(msg)
            actions = output["actions"]
            for i in range(action_buffer.maxlen):
                action_buffer.append(actions[i])
        
        action = action_buffer.popleft()

        # Store for plotting
        plot_info['gt'].append(gt_action)
        plot_info['pred'].append(action)

        error = np.linalg.norm(gt_action - action)
        logging.info(f"Step {step_idx}: error={error:.4f}")

    # Save final plots
    logging.info("Saving plots...")
    plot_info['gt'] = np.array(plot_info['gt'])
    plot_info['pred'] = np.array(plot_info['pred'])
    
    # Plot actions
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

    logging.info(f"Done! Videos and plots saved to {args.video_out_path}")
    
    # Calculate and log overall error statistics
    errors = np.linalg.norm(plot_info['gt'] - plot_info['pred'], axis=1)
    logging.info(f"Mean error: {np.mean(errors):.4f}")
    logging.info(f"Std error: {np.std(errors):.4f}")
    logging.info(f"Max error: {np.max(errors):.4f}")
    logging.info(f"Min error: {np.min(errors):.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)

