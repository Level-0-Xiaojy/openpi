"""
Pi05 Subtask Generation Inference on Offline Datasets.

This script demonstrates how to run Pi05 models with subtask generation capabilities
on offline datasets. It loads a model directly (not via WebSocket), generates subtasks
autoregressively, predicts actions, and compares with ground truth.

Usage:
    python main_reasoning.py --config-name pi05_droid --episode-id 5 --num-steps 100
"""

import collections
import dataclasses
import logging
import pathlib
import time

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from lerobot.common.datasets import lerobot_dataset
from openpi_client import image_tools
from PIL import Image
import tqdm
import tyro

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.training import config as _config
import openpi.shared.nnx_utils as nnx_utils


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
    
    # Convert figure to numpy array
    fig.canvas.draw()
    plot_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., 1:]  # Convert ARGB to RGB
    plt.close()
    
    return plot_array


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model configuration
    #################################################################################################################
    config_name: str = "pi05_droid"  # Config name from training/config.py
    
    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    repo_id: str = ""  # LeRobot dataset repo ID
    action_horizon: int = 10  # Number of actions to predict at each time step
    action_key: str = "actions"  # Key for the action in the dataset
    episode_id: int = 5  # Episode ID to run
    num_steps: int | None = None  # Number of steps to run (None = full episode)
    replan_steps: int = 5  # How often to replan (re-run model inference)
    
    #################################################################################################################
    # Inference parameters
    #################################################################################################################
    # High-level task prompt (for subtask generation)
    high_level_prompt: str | None = None  # If None, uses dataset task
    # Maximum number of subtask tokens to generate
    max_decoding_steps: int = 25
    # Temperature for subtask generation (0.0 = greedy, >0 = sampling)
    temperature: float = 0.1
    # Number of denoising steps for action prediction
    num_denoising_steps: int = 10
    
    #################################################################################################################
    # Output parameters
    #################################################################################################################
    video_out_path: str = "data/videos_reasoning"  # Path to save videos
    show_subtasks: bool = True  # Whether to print generated subtasks
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 42  # Random seed
    

def main(args: Args) -> None:
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("Pi05 Subtask Generation Inference on Offline Dataset")
    logging.info("=" * 80)
    
    # Load config
    logging.info(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    
    # Override repo_id if provided
    if args.repo_id:
        logging.info(f"Overriding dataset repo_id with: {args.repo_id}")
    else:
        # Get repo_id from config
        data_config = train_config.data.create(
            pathlib.Path(train_config.assets_base_dir) / train_config.name,
            train_config.model
        )
        args.repo_id = data_config.repo_id
        if not args.repo_id or args.repo_id == "fake":
            raise ValueError("Please provide --repo-id or use a config with a valid dataset")
    
    # Load dataset
    logging.info(f"Loading dataset: {args.repo_id}, episode: {args.episode_id}")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(args.action_horizon)] for key in [args.action_key]
        },
        episodes=[args.episode_id],
    )
    logging.info(f"Loaded episode {args.episode_id} with {len(dataset)} steps.")
    
    # Initialize model
    logging.info("Initializing model...")
    model_rng = jax.random.key(args.seed)
    model = train_config.model.create(model_rng)
    
    # Load pretrained params
    logging.info("Loading pretrained weights...")
    graphdef, state = nnx.split(model)
    loader = train_config.weight_loader
    params = nnx.state(model)
    
    # Convert frozen params to bfloat16
    if train_config.model.dtype == "bfloat16":
        params = nnx_utils.state_map(
            params, 
            train_config.freeze_filter, 
            lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
    
    params_shape = params.to_pure_dict()
    loaded_params = loader.load(params_shape)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)
    model.eval()
    
    logging.info("Model loaded successfully!")
    logging.info(f"  Model type: {train_config.model.model_type}")
    logging.info(f"  Action horizon: {train_config.model.action_horizon}")
    logging.info(f"  Action dim: {train_config.model.action_dim}")
    logging.info(f"  Max token len: {train_config.model.max_token_len}")
    
    # Setup data transforms
    data_config = train_config.data.create(
        pathlib.Path(train_config.assets_base_dir) / train_config.name,
        train_config.model
    )
    
    # Create transforms pipeline
    transforms_pipeline = _transforms.compose([
        data_config.repack_transforms,
        data_config.data_transforms,
        data_config.model_transforms,
    ])
    
    # Setup policy
    logging.info("Creating policy...")
    policy = _policy.Policy(
        model,
        transforms=transforms_pipeline.inputs,
        output_transforms=transforms_pipeline.outputs,
        sample_kwargs={"num_steps": args.num_denoising_steps},
    )
    
    # JIT compile the subtask generation function
    if hasattr(model, 'sample_low_level_task'):
        logging.info("JIT compiling subtask generation...")
        model.jit_sample_low_level_task = nnx_utils.module_jit(
            model.sample_low_level_task, 
            static_argnums=(3,)  # max_decoding_steps is static
        )
    
    # Run inference on dataset
    action_buffer = collections.deque(maxlen=args.replan_steps)
    plot_info = {
        'gt': [],
        'pred': [],
        'state': [],
    }
    replay_images = []
    generated_subtasks = []
    
    episode_length = args.num_steps if args.num_steps else len(dataset)
    inference_rng = jax.random.key(args.seed + 1)
    
    logging.info("=" * 80)
    logging.info("Running inference...")
    logging.info("=" * 80)
    
    for step_idx in tqdm.tqdm(range(episode_length), desc="Running episode"):
        data = dataset[step_idx]
        
        # Prepare observation dict
        # Note: These keys should match what your dataset provides
        # You may need to adjust based on your specific dataset
        obs = {}
        
        # Try common image key patterns
        for img_key in ['observation/exterior_image_1_left', 'observation/image', 'image']:
            if img_key in data:
                obs['observation/image'] = data[img_key].numpy()
                break
        
        for wrist_key in ['observation/wrist_image_left', 'observation/wrist_image', 'wrist_image']:
            if wrist_key in data:
                obs['observation/wrist_image'] = data[wrist_key].numpy()
                break
        
        # State
        if 'state' in data:
            obs['observation/state'] = data['state'].numpy()
        elif 'observation/state' in data:
            obs['observation/state'] = data['observation/state'].numpy()
        
        # Prompt
        if args.high_level_prompt:
            obs['prompt'] = args.high_level_prompt
        elif 'prompt' in data:
            obs['prompt'] = data['prompt']
        elif 'task' in data:
            obs['prompt'] = data['task']
        else:
            obs['prompt'] = "perform the task"
        
        # Get ground truth action
        gt_action = data[args.action_key].numpy()[0]
        
        # Run inference (replan every N steps)
        if len(action_buffer) == 0:
            start_time = time.monotonic()
            
            # Split RNG for this inference step
            inference_rng, step_rng = jax.random.split(inference_rng)
            
            # Run policy inference
            result = policy.infer(obs)
            actions = result["actions"]
            
            inference_time = time.monotonic() - start_time
            
            # Fill action buffer
            for i in range(action_buffer.maxlen):
                action_buffer.append(actions[i])
            
            logging.info(f"Step {step_idx}: inference_time={inference_time:.3f}s, "
                        f"error={np.linalg.norm(gt_action - actions[0]):.4f}")
        
        # Pop action from buffer
        action = action_buffer.popleft()
        
        # Store for plotting
        plot_info['gt'].append(gt_action)
        plot_info['pred'].append(action)
        
        if 'observation/state' in obs:
            plot_info['state'].append(obs['observation/state'])
        
        # Create visualization
        if 'observation/image' in obs:
            image = image_tools.convert_to_uint8(obs['observation/image'].transpose(1, 2, 0))
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if 'observation/wrist_image' in obs:
            wrist_image = image_tools.convert_to_uint8(obs['observation/wrist_image'].transpose(1, 2, 0))
            combined_image = np.concatenate([image, wrist_image], axis=1)
        else:
            combined_image = image
        
        # Create plot array
        plot_array = create_plot_array(plot_info['gt'], plot_info['pred'], plot_info['state'])
        if plot_array is not None:
            plot_img = Image.fromarray(plot_array)
            plot_img = plot_img.resize((combined_image.shape[1], 
                                       plot_img.height * combined_image.shape[1] // plot_img.width))
            plot_array = np.array(plot_img)
            combined_image = np.concatenate([combined_image, plot_array], axis=0)
        
        replay_images.append(combined_image)
    
    # Save results
    logging.info("=" * 80)
    logging.info("Saving results...")
    logging.info("=" * 80)
    
    # Plot actions
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
    actions_plot_path = pathlib.Path(args.video_out_path) / f"episode_{args.episode_id}_actions.png"
    plt.savefig(actions_plot_path)
    plt.close()
    logging.info(f"Saved actions plot to: {actions_plot_path}")
    
    # Plot states
    if len(plot_info['state']) > 0:
        states = np.array(plot_info['state'])
        plt.figure(figsize=(8, 3 * states.shape[1]))
        for i in range(states.shape[1]):
            plt.subplot(states.shape[1], 1, i+1)
            plt.plot(states[:, i])
        plt.xlabel('Time step')
        plt.ylabel('State value')
        plt.tight_layout()
        states_plot_path = pathlib.Path(args.video_out_path) / f"episode_{args.episode_id}_states.png"
        plt.savefig(states_plot_path)
        plt.close()
        logging.info(f"Saved states plot to: {states_plot_path}")
    
    # Save video
    video_path = pathlib.Path(args.video_out_path) / f"rollout_ep{args.episode_id}_reasoning.mp4"
    imageio.mimwrite(
        video_path,
        [np.asarray(x) for x in replay_images],
        fps=10,
    )
    logging.info(f"Saved video to: {video_path}")
    
    # Compute and log metrics
    mse = np.mean(np.square(plot_info['gt'] - plot_info['pred']))
    mae = np.mean(np.abs(plot_info['gt'] - plot_info['pred']))
    
    logging.info("=" * 80)
    logging.info("Results Summary:")
    logging.info("=" * 80)
    logging.info(f"  Episode: {args.episode_id}")
    logging.info(f"  Steps: {episode_length}")
    logging.info(f"  MSE: {mse:.6f}")
    logging.info(f"  MAE: {mae:.6f}")
    logging.info(f"  Config: {args.config_name}")
    logging.info(f"  Dataset: {args.repo_id}")
    logging.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    tyro.cli(main)

