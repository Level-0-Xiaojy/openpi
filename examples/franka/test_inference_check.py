import tyro
from dataclasses import dataclass
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import sys
import os 
from utils import calc_mse_for_single_trajectory

@dataclass
class Args:
    checkpoint_dir: str 
    config_name: str
    steps: int = 150
    plot: bool = True
    repo_id: str = None


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load config
    config = _config.get_config(args.config_name)
    
    # Prepare data_config_overrides if repo_id is provided
    data_config_overrides = None
    if args.repo_id:
        data_config_overrides = {
            "repo_id": args.repo_id,
            "asset_id": args.repo_id,
        }
    
    # Create policy
    policy = _policy_config.create_trained_policy(
        config, 
        args.checkpoint_dir, 
        data_config_overrides=data_config_overrides
    )

    # Run comparison
    calc_mse_for_single_trajectory(
        policy, 
        config, 
        traj_id=0, 
        steps=args.steps, 
        plot=args.plot, 
        repo_id=args.repo_id
    )

    # Cleanup
    del policy
