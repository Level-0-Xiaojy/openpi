import tyro
from dataclasses import dataclass
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calc_mse_for_single_trajectory

@dataclass
class Args:
    checkpoint_dir: str 
    config_name: str
    plot: bool = True


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load config
    config = _config.get_config(args.config_name)

    # Create policy
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)

    # Run comparison
    calc_mse_for_single_trajectory(policy, config, traj_id=0,  plot=args.plot)

    # Cleanup
    del policy
