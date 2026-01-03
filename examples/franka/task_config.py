"""
Task configuration utilities for training and deployment.
"""

import os
import yaml
import dataclasses
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    config_name: str
    pytorch_weight_path: str
    # Model architecture options
    discrete_state_input: Optional[bool] = None  # None means use default (pi05=True, others=False)


@dataclasses.dataclass
class DataConfig:
    repo_id: str
    data_dir: Optional[str] = None


@dataclasses.dataclass
class TrainConfig:
    num_train_steps: int = 30000
    save_interval: int = 5000
    batch_size: int = 32
    overwrite: bool = True
    exp_name: Optional[str] = None
    # GPU settings for training
    gpu_ids: list[int] = dataclasses.field(default_factory=lambda: [0])  # List of GPU IDs for training
    xla_preallocate: bool = False


@dataclasses.dataclass
class DeployConfig:
    host: str = "0.0.0.0"
    port: int = 12559
    checkpoint_step: Optional[str] = None  # "latest", specific step, or None for auto-detection
    # GPU settings for deployment
    gpu_id: int = 0  # Single GPU for deployment
    xla_preallocate: bool = False


@dataclasses.dataclass
class NormStatsConfig:
    """Configuration for norm stats computation."""
    # GPU settings for norm stats
    gpu_id: int = 0  # Single GPU for norm stats
    xla_preallocate: bool = False
    overwrite: bool = False  # Whether to recompute norm stats if they already exist


@dataclasses.dataclass
class InferConfig:
    """Configuration for dataset inference/validation."""
    checkpoint_step: Optional[str] = None  # "latest", specific step, or None for auto-detection
    steps: int = 150  # Number of steps to run in trajectory
    plot: bool = True  # Whether to generate plots
    traj_id: int = 0  # Trajectory ID to validate
    # GPU settings for inference
    gpu_id: int = 6  # Single GPU for inference


@dataclasses.dataclass
class RemoteConfig:
    """Configuration for syncing files from remote server."""
    enabled: bool = False
    host: Optional[str] = None  # e.g., "wq-dev0"
    remote_path: Optional[str] = None  # e.g., "/mnt/public/bingwen/Projects/openpi"
    local_path: str = "~/Documents/openpi"
    # Specific checkpoint step to sync, or "latest" for the latest checkpoint
    checkpoint_step: Optional[str] = None


@dataclasses.dataclass
class TaskConfig:
    """Unified configuration for training and deployment."""
    task_name: str
    instruction: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    deploy: DeployConfig = dataclasses.field(default_factory=DeployConfig)
    norm_stats: NormStatsConfig = dataclasses.field(default_factory=NormStatsConfig)
    infer: InferConfig = dataclasses.field(default_factory=InferConfig)
    remote: RemoteConfig = dataclasses.field(default_factory=RemoteConfig)
    
    # Normalization mode: "quantile_norm", "z_score", or "auto" (auto uses model type default)
    norm_mode: str = "auto"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TaskConfig":
        """Load config from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            task_name=data['task_name'],
            instruction=data['instruction'],
            model=ModelConfig(**data['model']),
            data=DataConfig(**data['data']),
            train=TrainConfig(**data.get('train', {})),
            deploy=DeployConfig(**data.get('deploy', {})),
            norm_stats=NormStatsConfig(**data.get('norm_stats', {})),
            infer=InferConfig(**data.get('infer', {})),
            remote=RemoteConfig(**data.get('remote', {})),
            norm_mode=data.get('norm_mode', 'auto'),  # Load norm_mode from yaml
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to a YAML file."""
        data = {
            'task_name': self.task_name,
            'instruction': self.instruction,
            'model': dataclasses.asdict(self.model),
            'data': dataclasses.asdict(self.data),
            'norm_mode': self.norm_mode,  # Save norm_mode to yaml
            'train': dataclasses.asdict(self.train),
            'deploy': dataclasses.asdict(self.deploy),
            'norm_stats': dataclasses.asdict(self.norm_stats),
            'infer': dataclasses.asdict(self.infer),
            'remote': dataclasses.asdict(self.remote),
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_exp_name(self) -> str:
        """Get experiment name, defaulting to task_name."""
        return self.train.exp_name or self.task_name
    
    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory for deployment."""
        base_dir = Path(f"checkpoints/{self.model.config_name}/{self.get_exp_name()}")
        
        # Check for checkpoint_step in deploy config first, then remote config
        checkpoint_step = None
        if hasattr(self.deploy, 'checkpoint_step') and self.deploy.checkpoint_step:
            checkpoint_step = self.deploy.checkpoint_step
        elif hasattr(self.remote, 'checkpoint_step') and self.remote.checkpoint_step:
            checkpoint_step = self.remote.checkpoint_step
        
        # If specific checkpoint step is provided
        if checkpoint_step:
            if checkpoint_step == "latest":
                # Find the latest checkpoint
                if base_dir.exists():
                    step_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                    if step_dirs:
                        latest = max(step_dirs, key=lambda x: int(x.name))
                        return str(latest)
                # Fallback to training step if no checkpoints found
                return str(base_dir / str(self.train.num_train_steps))
            else:
                # Use specific step
                return str(base_dir / checkpoint_step)
        
        # Default: find latest checkpoint
        if base_dir.exists():
            step_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if step_dirs:
                latest = max(step_dirs, key=lambda x: int(x.name))
                return str(latest)
        return str(base_dir / str(self.train.num_train_steps))
    
    def get_norm_stats_env_prefix(self) -> str:
        """Get environment variable prefix for norm stats."""
        env_parts = []
        if not self.norm_stats.xla_preallocate:
            env_parts.append("XLA_PYTHON_CLIENT_PREALLOCATE=false")
        env_parts.append(f"CUDA_VISIBLE_DEVICES={self.norm_stats.gpu_id}")
        return " ".join(env_parts)
    
    def get_train_env_prefix(self) -> str:
        """Get environment variable prefix for training."""
        env_parts = []
        if not self.train.xla_preallocate:
            env_parts.append("XLA_PYTHON_CLIENT_PREALLOCATE=false")
        env_parts.append(f"CUDA_VISIBLE_DEVICES={','.join(map(str, self.train.gpu_ids))}")
        return " ".join(env_parts)
    
    def get_deploy_env_prefix(self) -> str:
        """Get environment variable prefix for deployment."""
        env_parts = []
        if not self.deploy.xla_preallocate:
            env_parts.append("XLA_PYTHON_CLIENT_PREALLOCATE=false")
        env_parts.append(f"CUDA_VISIBLE_DEVICES={self.deploy.gpu_id}")
        return " ".join(env_parts)
    
    def get_norm_stats_cmd(self) -> str:
        """Generate command for computing norm stats."""
        cmd = (
            f'{self.get_norm_stats_env_prefix()} uv run scripts/compute_norm_stats.py \\\n'
            f'    --config-name "{self.model.config_name}" \\\n'
            f'    --repo-id "{self.data.repo_id}"'
        )
        
        # Add overwrite flag if specified
        if self.norm_stats.overwrite:
            cmd += ' \\\n    --overwrite'
            
        return cmd
    
    def get_train_cmd(self) -> str:
        """Generate training command."""
        if len(self.train.gpu_ids) > 1:
            # Multi-GPU training with torchrun
            cmd = (
                f'CUDA_VISIBLE_DEVICES={",".join(map(str, self.train.gpu_ids))} uv run torchrun \\\n'
                f'    --standalone \\\n'
                f'    --nnodes=1 \\\n'
                f'    --nproc_per_node={len(self.train.gpu_ids)} \\\n'
                f'    scripts/train_pytorch.py {self.model.config_name} \\\n'
                f'    --pytorch_weight_path "{self.model.pytorch_weight_path}" \\\n'
                f'    --exp_name "{self.get_exp_name()}" \\\n'
                f'    --data.repo_id "{self.data.repo_id}" \\\n'
                f'    --num_train_steps {self.train.num_train_steps} \\\n'
                f'    --save_interval {self.train.save_interval} \\\n'
                f'    --batch_size {self.train.batch_size}'
            )
        else:
            # Single GPU training
            env_prefix = []
            if not self.train.xla_preallocate:
                env_prefix.append("XLA_PYTHON_CLIENT_PREALLOCATE=false")
            env_prefix.append(f"CUDA_VISIBLE_DEVICES={self.train.gpu_ids[0]}")
            
            cmd = (
                f'{" ".join(env_prefix)} uv run scripts/train_pytorch.py {self.model.config_name} \\\n'
                f'    --pytorch_weight_path "{self.model.pytorch_weight_path}" \\\n'
                f'    --exp_name "{self.get_exp_name()}" \\\n'
                f'    --data.repo_id "{self.data.repo_id}" \\\n'
                f'    --num_train_steps {self.train.num_train_steps} \\\n'
                f'    --save_interval {self.train.save_interval} \\\n'
                f'    --batch_size {self.train.batch_size}'
            )
        
        # Add discrete_state_input parameter if specified
        if self.model.discrete_state_input is not None:
            if self.model.discrete_state_input:
                cmd += ' \\\n    --model.discrete-state-input'
            else:
                cmd += ' \\\n    --model.no-discrete-state-input'
        
        # Add norm_mode parameter if not auto
        if self.norm_mode != "auto":
            cmd += f' \\\n    --data.norm-mode "{self.norm_mode}"'
        
        if self.train.overwrite:
            cmd += ' \\\n    --overwrite'
        return cmd

    def get_deploy_cmd(self) -> str:
        """Generate deployment command."""
        cmd = (
            f'{self.get_deploy_env_prefix()} uv run examples/franka/server_policy.py \\\n'
            f'    --host "{self.deploy.host}" \\\n'
            f'    --port {self.deploy.port} \\\n'
            f'    --repo-id "{self.data.repo_id}" \\\n'
            f'    --instruction "{self.instruction}" \\\n'
        )
        
        # Add discrete_state_input if specified
        if self.model.discrete_state_input is not None:
            discrete_flag = "True" if self.model.discrete_state_input else "False"
            cmd += f'    --discrete-state-input {discrete_flag} \\\n'
        
        # Add norm_mode if not auto
        if self.norm_mode != "auto":
            cmd += f'    --norm-mode "{self.norm_mode}" \\\n'
        
        cmd += (
            f'    policy:checkpoint \\\n'
            f'    --policy.config="{self.model.config_name}" \\\n'
            f'    --policy.dir="{self.get_checkpoint_dir()}"'
        )
        
        return cmd
    
    def get_sync_cmd(self) -> str:
        """Generate commands to sync checkpoints and assets from remote server."""
        if not self.remote.enabled or not self.remote.host or not self.remote.remote_path:
            return "# Remote sync not configured"
        
        # Determine checkpoint step
        if self.remote.checkpoint_step:
            if self.remote.checkpoint_step == "latest":
                # Use a command to find the latest checkpoint
                checkpoint_step = "$(ssh {host} 'ls -1 {remote_path}/checkpoints/{config_name}/{exp_name}/ | grep -E \"^[0-9]+$\" | sort -n | tail -1')".format(
                    host=self.remote.host,
                    remote_path=self.remote.remote_path,
                    config_name=self.model.config_name,
                    exp_name=self.get_exp_name()
                )
            else:
                checkpoint_step = self.remote.checkpoint_step
        else:
            checkpoint_step = str(self.train.num_train_steps)
        
        commands = []
        
        # Sync assets directory
        assets_src = f"{self.remote.host}:{self.remote.remote_path}/assets/{self.model.config_name}/{self.data.repo_id}/"
        assets_dst = f"{self.remote.local_path}/assets/{self.model.config_name}/"
        commands.append(f"\n# Sync assets from remote server")
        commands.append(f"mkdir -p {self.remote.local_path}/assets/{self.model.config_name}")
        commands.append(f"rsync -avzP {assets_src} {assets_dst}")
        
        # Sync wandb_id.txt file
        wandb_src = f"{self.remote.host}:{self.remote.remote_path}/checkpoints/{self.model.config_name}/{self.get_exp_name()}/wandb_id.txt"
        wandb_dst = f"{self.remote.local_path}/checkpoints/{self.model.config_name}/{self.get_exp_name()}/"
        commands.append(f"# Sync wandb_id.txt file")
        commands.append(f"rsync -avzP {wandb_src} {wandb_dst}")
        
        # Sync checkpoint directory
        checkpoint_src = f"{self.remote.host}:{self.remote.remote_path}/checkpoints/{self.model.config_name}/{self.get_exp_name()}/{checkpoint_step}"
        checkpoint_dst = f"{self.remote.local_path}/checkpoints/{self.model.config_name}/{self.get_exp_name()}/"
        commands.append(f"# Sync checkpoint from remote server")
        commands.append(f"mkdir -p {self.remote.local_path}/checkpoints/{self.model.config_name}/{self.get_exp_name()}")
        commands.append(f"rsync -avzP --exclude='optimizer.pt' {checkpoint_src} {checkpoint_dst}")
        
        return "\n".join(commands)
    
    def get_infer_cmd(self) -> str:
        """Generate inference/validation command."""
        # Determine checkpoint step
        if self.infer.checkpoint_step:
            if self.infer.checkpoint_step == "latest":
                checkpoint_dir = self.get_checkpoint_dir()
            else:
                checkpoint_dir = f"checkpoints/{self.model.config_name}/{self.get_exp_name()}/{self.infer.checkpoint_step}"
        else:
            checkpoint_dir = self.get_checkpoint_dir()
        
        cmd = (
            f'CUDA_VISIBLE_DEVICES={self.infer.gpu_id} python examples/franka/validate_dataset_inference.py \\\n'
            f'    --steps {self.infer.steps} \\\n'
            f'    --dataset_repo_id {self.data.repo_id} \\\n'
            f'    --asset_id {self.data.repo_id} \\\n'
            f'    --norm_mode {self.norm_mode} \\\n'
        )
        
        # Add plot flag if disabled
        if not self.infer.plot:
            cmd += '    --no-plot \\\n'
        
        # Add discrete_state_input if specified
        if self.model.discrete_state_input is not None:
            discrete_flag = "True" if self.model.discrete_state_input else "False"
            cmd += f'    --discrete-state-input {discrete_flag} \\\n'
        
        cmd += (
            f'    policy:checkpoint \\\n'
            f'    --policy.config {self.model.config_name} \\\n'
            f'    --policy.dir="{checkpoint_dir}"'
        )
        
        return cmd
    
    def print_commands(self) -> None:
        """Print all commands for this task."""
        print(f"=== Task: {self.task_name} ===\n")
        
        if self.remote.enabled:
            print("# 0. Sync from remote server:")
            print(self.get_sync_cmd())
            print()
        
        print("# 1. Compute norm stats:")
        print(self.get_norm_stats_cmd())
        print("\n# 2. Train:")
        print(self.get_train_cmd())
        print("\n# 3. Infer/Validate on dataset:")
        print(self.get_infer_cmd())
        print("\n# 4. Deploy:")
        print(self.get_deploy_cmd())
        print()
