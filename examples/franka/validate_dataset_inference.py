import dataclasses
import enum

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from utils import calc_mse_for_single_trajectory

class EnvMode(enum.Enum):
    """Supported environments (kept consistent with server_policy.py)."""

    PANDA = "panda"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint (consistent with server_policy.py)."""

    # Training config name (e.g., "pi0_franka", "pi05_franka").
    config: str
    # Checkpoint directory.
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for validate_dataset_inference (aligned with server_policy.py)."""

    # Steps within a trajectory to run.
    steps: int = 150
    plot: bool = True

    # Dataset repo_id - the actual lerobot dataset name (e.g., "pi05_real_sm_10hz_pp")
    # This specifies which dataset to load from ~/.cache/huggingface/lerobot/<dataset_repo_id>
    dataset_repo_id: str | None = None

    # Optional asset_id to load norm stats from assets directory (./assets/{config_name}/{asset_id}/)
    # If not provided, will use dataset_repo_id as asset_id
    asset_id: str | None = None

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Whether to use discrete state input for the policy
    discrete_state_input: bool | None = None

    # Normalization mode: "auto", "quantile_norm", "z_score"
    norm_mode: str = "z_score"

    # Specifies how to load the policy.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.PANDA: Checkpoint(
        config="pi0_franka",
        dir="checkpoints/pi0_franka/bingwen_thu/29999 ",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments (mirrors server_policy.py)."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)

            if args.discrete_state_input is not None:
                train_config = dataclasses.replace(
                    train_config,
                    model=dataclasses.replace(train_config.model, discrete_state_input=args.discrete_state_input),
                )

            if args.norm_mode != "auto" and hasattr(train_config.data, "norm_mode"):
                train_config = dataclasses.replace(
                    train_config,
                    data=dataclasses.replace(train_config.data, norm_mode=args.norm_mode),
                )
            print("""Creating policy from checkpoint:""", args.policy.dir)
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                repo_id=args.asset_id,  # repo_id here is for loading norm stats
            )
        case Default():
            return create_default_policy(EnvMode.PANDA, default_prompt=args.default_prompt)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Create policy + load config in the same way as server_policy.py.
    policy = create_policy(args)

    print("Loaded policy:", policy)
    # Determine config name used for dataset creation.
    if isinstance(args.policy, Checkpoint):
        config_name = args.policy.config
    else:
        # Default policy uses DEFAULT_CHECKPOINT.
        config_name = DEFAULT_CHECKPOINT[EnvMode.PANDA].config
    config = _config.get_config(config_name)
    print("Using config:", config_name)
    
    # If asset_id not provided, use dataset_repo_id
    asset_id = args.asset_id if args.asset_id is not None else args.dataset_repo_id
    
    # Run comparison
    calc_mse_for_single_trajectory(
        policy, 
        config, 
        traj_id=0, 
        steps=args.steps, 
        plot=args.plot, 
        dataset_repo_id=args.dataset_repo_id,
        asset_id=asset_id,
    )

    # Cleanup
    del policy
