"""Visualize action dimension distribution in raw data.

This script helps visualize the distribution of a specific action dimension in the raw
dataset (before normalization) to diagnose normalization issues.
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/share/xuyuanfan-local/small_project/.cache/hf_home"  # Set it to yours
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use the first GPU

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

# Set matplotlib style (try seaborn styles, fallback to default)
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        plt.style.use("default")
sns.set_palette("husl")


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_samples: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create a data loader for raw actions (without normalization)."""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    # Apply transforms but NOT normalization
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_samples is not None and max_samples < len(dataset):
        num_batches = max_samples // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_samples: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create an RLDS data loader for raw actions (without normalization)."""
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_samples is not None:
        num_batches = max_samples // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def load_raw_actions(config_name: str, max_samples: int | None = None) -> np.ndarray:
    """Load raw action data without normalization.
    
    Args:
        config_name: Name of the config
        max_samples: Maximum number of samples to load
        
    Returns:
        Array of raw actions with shape (num_samples, action_dim)
    """
    print("=" * 80)
    print(f"LOADING RAW ACTIONS FOR: {config_name}")
    print("=" * 80)
    
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_samples
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_samples
        )
    
    all_actions = []
    sample_count = 0
    
    print(f"\nLoading actions from dataset...")
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Loading batches"):
        if "actions" in batch:
            actions = np.asarray(batch["actions"])
            # Handle action horizon dimension: (batch_size, action_horizon, action_dim) or (batch_size, action_dim)
            if actions.ndim > 2:
                # Flatten batch and horizon dimensions
                actions = actions.reshape(-1, actions.shape[-1])
            all_actions.append(actions)
            sample_count += len(actions)
            
            if max_samples is not None and sample_count >= max_samples:
                break
    
    if not all_actions:
        raise ValueError("No actions found in dataset")
    
    actions_array = np.concatenate(all_actions, axis=0)
    if max_samples is not None:
        actions_array = actions_array[:max_samples]
    
    print(f"\n✓ Loaded {len(actions_array)} action samples")
    print(f"  Action dimension: {actions_array.shape[1]}")
    
    return actions_array


def load_trajectory_actions(
    config_name: str,
    trajectory_idx: int,
    max_steps: int = 1000,
) -> tuple[np.ndarray, dict]:
    """Load actions from a single trajectory/episode.
    
    Args:
        config_name: Name of the config
        trajectory_idx: Starting index of the trajectory in the dataset
        max_steps: Maximum number of steps to load (default: 1000)
        
    Returns:
        Tuple of (actions array with shape (num_steps, action_dim), metadata dict)
    """
    print("=" * 80)
    print(f"LOADING TRAJECTORY STARTING AT INDEX {trajectory_idx} FOR: {config_name}")
    print("=" * 80)
    
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Create dataset without normalization
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("RLDS dataset trajectory loading not yet implemented")
    
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    
    # Check trajectory index
    dataset_len = len(dataset)
    if trajectory_idx < 0 or trajectory_idx >= dataset_len:
        raise ValueError(f"Trajectory starting index {trajectory_idx} is out of range [0, {dataset_len})")
    
    print(f"\nLoading trajectory starting from index {trajectory_idx} (max {max_steps} steps)...")
    
    # Load a sequence of samples (assuming they form a trajectory)
    # We'll load up to max_steps or until we hit dataset end
    trajectory_actions = []
    
    for step_idx in range(max_steps):
        sample_idx = trajectory_idx + step_idx
        if sample_idx >= dataset_len:
            print(f"Reached end of dataset at step {step_idx}")
            break
        
        try:
            sample = dataset[sample_idx]
            if "actions" in sample:
                # Get the first action in the sequence (current action at this step)
                actions = np.asarray(sample["actions"])
                if actions.ndim > 1:
                    # If it's a sequence, take the first one (current action)
                    action = actions[0]
                else:
                    action = actions
                trajectory_actions.append(action)
            else:
                # If no actions, might be end of trajectory
                print(f"No actions found at step {step_idx}, stopping")
                break
        except (IndexError, KeyError) as e:
            # Might have hit trajectory boundary or error
            print(f"Error loading step {step_idx}: {e}, stopping")
            break
    
    if not trajectory_actions:
        raise ValueError(f"No actions found in trajectory starting at index {trajectory_idx}")
    
    actions_array = np.array(trajectory_actions)
    
    metadata = {
        "trajectory_idx": trajectory_idx,
        "num_steps": len(actions_array),
        "action_dim": actions_array.shape[1] if actions_array.ndim > 1 else 1,
    }
    
    print(f"\n✓ Loaded trajectory with {metadata['num_steps']} steps")
    print(f"  Action dimension: {metadata['action_dim']}")
    
    return actions_array, metadata


def visualize_trajectory_action_dimension(
    actions: np.ndarray,
    dim: int,
    metadata: dict,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    """Visualize how a specific action dimension changes over steps in a trajectory.
    
    Args:
        actions: Array of actions with shape (num_steps, action_dim)
        dim: Dimension index to visualize
        metadata: Dictionary containing trajectory metadata
        output_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D array (num_steps, action_dim), got shape {actions.shape}")
    
    if dim < 0 or dim >= actions.shape[1]:
        raise ValueError(f"Dimension {dim} is out of range [0, {actions.shape[1]})")
    
    dim_data = actions[:, dim]
    num_steps = len(dim_data)
    steps = np.arange(num_steps)
    
    # Compute statistics for this trajectory
    stats = {
        "min": np.min(dim_data),
        "max": np.max(dim_data),
        "mean": np.mean(dim_data),
        "std": np.std(dim_data),
        "median": np.median(dim_data),
        "range": np.max(dim_data) - np.min(dim_data),
    }
    
    # Print statistics
    print("\n" + "=" * 80)
    print(f"TRAJECTORY {metadata['trajectory_idx']} - ACTION DIMENSION {dim} STATISTICS")
    print("=" * 80)
    print(f"Number of steps: {num_steps}")
    print(f"Min:      {stats['min']:.6f}")
    print(f"Max:      {stats['max']:.6f}")
    print(f"Mean:     {stats['mean']:.6f}")
    print(f"Std:      {stats['std']:.6f}")
    print(f"Median:   {stats['median']:.6f}")
    print(f"Range:    {stats['range']:.6f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Time series plot (main plot)
    ax1 = fig.add_subplot(gs[0, :])  # Full width
    ax1.plot(steps, dim_data, 'b-', linewidth=1.5, alpha=0.7, label=f"Dim {dim}")
    ax1.axhline(stats['mean'], color='r', linestyle='--', alpha=0.7, label=f"Mean: {stats['mean']:.4f}")
    ax1.axhline(stats['median'], color='g', linestyle='--', alpha=0.7, label=f"Median: {stats['median']:.4f}")
    ax1.fill_between(
        steps,
        stats['mean'] - stats['std'],
        stats['mean'] + stats['std'],
        alpha=0.2,
        color='red',
        label=f"±1 Std"
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel(f"Action Dimension {dim} Value")
    ax1.set_title(f"Action Dimension {dim} Over Time (Trajectory {metadata['trajectory_idx']})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(dim_data, bins=min(50, num_steps // 2), alpha=0.7, edgecolor="black", density=True)
    ax2.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.4f}")
    ax2.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']:.4f}")
    ax2.set_xlabel(f"Action Dimension {dim} Value")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Distribution in Trajectory")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 1])
    bp = ax3.boxplot(
        [dim_data],
        vert=True,
        patch_artist=True,
        labels=[f"Dim {dim}"],
        showmeans=True,
        meanline=True,
    )
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax3.set_ylabel(f"Action Dimension {dim} Value")
    ax3.set_title(f"Box Plot")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = (
        f"Steps: {num_steps}\n"
        f"Min: {stats['min']:.4f}\n"
        f"Max: {stats['max']:.4f}\n"
        f"Mean: {stats['mean']:.4f}\n"
        f"Std: {stats['std']:.4f}\n"
        f"Median: {stats['median']:.4f}\n"
        f"Range: {stats['range']:.4f}"
    )
    ax3.text(
        1.15, 0.5, stats_text, transform=ax3.transAxes,
        fontsize=9, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    fig.suptitle(
        f"Trajectory {metadata['trajectory_idx']} - Action Dimension {dim} Over Time",
        fontsize=16,
        fontweight='bold'
    )
    
    # Save figure if path provided
    if output_path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {output_path}")
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_action_dimension(
    actions: np.ndarray,
    dim: int,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    """Visualize the distribution of a specific action dimension.
    
    Args:
        actions: Array of actions with shape (num_samples, action_dim)
        dim: Dimension index to visualize
        output_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    if dim < 0 or dim >= actions.shape[1]:
        raise ValueError(f"Dimension {dim} is out of range [0, {actions.shape[1]})")
    
    dim_data = actions[:, dim]
    
    # Compute statistics
    stats = {
        "min": np.min(dim_data),
        "max": np.max(dim_data),
        "mean": np.mean(dim_data),
        "std": np.std(dim_data),
        "median": np.median(dim_data),
        "q01": np.percentile(dim_data, 1),
        "q99": np.percentile(dim_data, 99),
        "q25": np.percentile(dim_data, 25),
        "q75": np.percentile(dim_data, 75),
    }
    
    # Print statistics
    print("\n" + "=" * 80)
    print(f"ACTION DIMENSION {dim} STATISTICS")
    print("=" * 80)
    print(f"Min:      {stats['min']:.6f}")
    print(f"Max:      {stats['max']:.6f}")
    print(f"Mean:     {stats['mean']:.6f}")
    print(f"Std:      {stats['std']:.6f}")
    print(f"Median:   {stats['median']:.6f}")
    print(f"Q01:      {stats['q01']:.6f}")
    print(f"Q25:      {stats['q25']:.6f}")
    print(f"Q75:      {stats['q75']:.6f}")
    print(f"Q99:      {stats['q99']:.6f}")
    print(f"Range:    {stats['max'] - stats['min']:.6f}")
    print(f"IQR:      {stats['q75'] - stats['q25']:.6f}")
    
    # Check for potential issues
    range_val = stats['q99'] - stats['q01']
    if range_val < 0.001:
        print(f"\n⚠️  WARNING: Very small range (q99 - q01 = {range_val:.6f})")
        print("   This will cause very large normalized values!")
    if stats['std'] < 0.001:
        print(f"\n⚠️  WARNING: Very small std ({stats['std']:.6f})")
        print("   This will cause very large normalized values!")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(dim_data, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax1.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.4f}")
    ax1.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']:.4f}")
    ax1.axvline(stats['q01'], color='orange', linestyle=':', alpha=0.7, label=f"Q01: {stats['q01']:.4f}")
    ax1.axvline(stats['q99'], color='orange', linestyle=':', alpha=0.7, label=f"Q99: {stats['q99']:.4f}")
    ax1.set_xlabel(f"Action Dimension {dim} Value")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Histogram of Action Dimension {dim}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. KDE plot
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        sns.kdeplot(data=dim_data, ax=ax2, fill=True, alpha=0.7)
    except Exception:
        # Fallback to histogram if KDE fails
        ax2.hist(dim_data, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax2.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.4f}")
    ax2.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']:.4f}")
    ax2.set_xlabel(f"Action Dimension {dim} Value")
    ax2.set_ylabel("Density")
    ax2.set_title(f"KDE Plot of Action Dimension {dim}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(
        [dim_data],
        vert=True,
        patch_artist=True,
        labels=[f"Dim {dim}"],
        showmeans=True,
        meanline=True,
    )
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax3.set_ylabel(f"Action Dimension {dim} Value")
    ax3.set_title(f"Box Plot of Action Dimension {dim}")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = (
        f"Min: {stats['min']:.4f}\n"
        f"Q01: {stats['q01']:.4f}\n"
        f"Q25: {stats['q25']:.4f}\n"
        f"Median: {stats['median']:.4f}\n"
        f"Q75: {stats['q75']:.4f}\n"
        f"Q99: {stats['q99']:.4f}\n"
        f"Max: {stats['max']:.4f}\n"
        f"Mean: {stats['mean']:.4f}\n"
        f"Std: {stats['std']:.4f}\n"
        f"Range: {stats['max'] - stats['min']:.4f}"
    )
    ax3.text(
        1.15, 0.5, stats_text, transform=ax3.transAxes,
        fontsize=9, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # 4. Q-Q plot (to check normality)
    ax4 = fig.add_subplot(gs[1, 1])
    try:
        from scipy import stats as scipy_stats
        scipy_stats.probplot(dim_data, dist="norm", plot=ax4)
        ax4.set_title(f"Q-Q Plot (Normal Distribution) - Dimension {dim}")
        ax4.grid(True, alpha=0.3)
    except ImportError:
        # If scipy is not available, show a scatter plot of sorted values
        sorted_data = np.sort(dim_data)
        ax4.scatter(range(len(sorted_data)), sorted_data, alpha=0.5, s=1)
        ax4.set_xlabel("Sample Index (sorted)")
        ax4.set_ylabel(f"Action Dimension {dim} Value")
        ax4.set_title(f"Sorted Values - Dimension {dim}")
        ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f"Action Dimension {dim} Distribution Analysis", fontsize=16, fontweight='bold')
    
    # Save figure if path provided
    if output_path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {output_path}")
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()


def main(
    config_name: str,
    dim: int = 9,
    max_samples: int | None = None,
    save_path: str | None = None,
    show: bool = True,
    trajectory_idx: int | None = None,
    max_trajectory_steps: int = 1000,
):
    """Main function to visualize action dimension.
    
    Args:
        config_name: Name of the config
        dim: Dimension index to visualize (default: 9)
        max_samples: Maximum number of samples to load for distribution visualization (default: all)
        save_path: Path to save the figure (optional)
        show: Whether to display the figure (default: True)
        trajectory_idx: If specified, visualize a single trajectory over time instead of distribution.
                        This is the starting index of the trajectory in the dataset.
        max_trajectory_steps: Maximum number of steps to load for trajectory visualization (default: 1000)
    """
    if trajectory_idx is not None:
        # Visualize single trajectory over time
        actions, metadata = load_trajectory_actions(config_name, trajectory_idx, max_trajectory_steps)
        visualize_trajectory_action_dimension(actions, dim, metadata, save_path, show)
    else:
        # Visualize distribution across all samples
        actions = load_raw_actions(config_name, max_samples)
        visualize_action_dimension(actions, dim, save_path, show)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)

