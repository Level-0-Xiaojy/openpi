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
    # Multi-GPU training
    multi_gpu: bool = False
    num_gpus: int = 1
    # Norm stats options
    overwrite_norm_stats: bool = False


@dataclasses.dataclass
class DeployConfig:
    host: str = "0.0.0.0"
    port: int = 12559
    checkpoint_dir: Optional[str] = None


@dataclasses.dataclass
class GPUConfig:
    device_id: Optional[int] = None  # For single GPU (backward compatibility)
    device_ids: Optional[list[int]] = None  # For multi-GPU
    xla_preallocate: bool = False
    
    def get_device_ids(self) -> list[int]:
        """Get list of device IDs, handling both single and multi-GPU configs."""
        if self.device_ids is not None:
            return self.device_ids
        elif self.device_id is not None:
            return [self.device_id]
        else:
            return [0]  # Default to GPU 0


@dataclasses.dataclass
class TaskConfig:
    """Unified configuration for training and deployment."""
    task_name: str
    instruction: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    deploy: DeployConfig = dataclasses.field(default_factory=DeployConfig)
    gpu: GPUConfig = dataclasses.field(default_factory=GPUConfig)
    
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
            gpu=GPUConfig(**data.get('gpu', {})),
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to a YAML file."""
        data = {
            'task_name': self.task_name,
            'instruction': self.instruction,
            'model': dataclasses.asdict(self.model),
            'data': dataclasses.asdict(self.data),
            'train': dataclasses.asdict(self.train),
            'deploy': dataclasses.asdict(self.deploy),
            'gpu': dataclasses.asdict(self.gpu),
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_exp_name(self) -> str:
        """Get experiment name, defaulting to task_name."""
        return self.train.exp_name or self.task_name
    
    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory for deployment."""
        if self.deploy.checkpoint_dir:
            return self.deploy.checkpoint_dir
        # Default: find latest checkpoint
        base_dir = Path(f"checkpoints/{self.model.config_name}/{self.get_exp_name()}")
        if base_dir.exists():
            # Find the latest step directory
            step_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if step_dirs:
                latest = max(step_dirs, key=lambda x: int(x.name))
                return str(latest)
        return str(base_dir / str(self.train.num_train_steps))
    
    def get_env_prefix(self) -> str:
        """Get environment variable prefix for GPU settings."""
        env_parts = []
        if not self.gpu.xla_preallocate:
            env_parts.append("XLA_PYTHON_CLIENT_PREALLOCATE=false")
        
        device_ids = self.gpu.get_device_ids()
        env_parts.append(f"CUDA_VISIBLE_DEVICES={','.join(map(str, device_ids))}")
        return " ".join(env_parts)
    
    def get_norm_stats_cmd(self) -> str:
        """Generate command for computing norm stats."""
        # Norm stats only needs single GPU
        device_ids = self.gpu.get_device_ids()
        env_prefix = []
        if not self.gpu.xla_preallocate:
            env_prefix.append("XLA_PYTHON_CLIENT_PREALLOCATE=false")
        env_prefix.append(f"CUDA_VISIBLE_DEVICES={device_ids[0]}")
        
        cmd = (
            f'{" ".join(env_prefix)} uv run scripts/compute_norm_stats.py \\\n'
            f'    --config-name "{self.model.config_name}" \\\n'
            f'    --repo-id "{self.data.repo_id}"'
        )
        
        if self.train.overwrite_norm_stats:
            cmd += ' \\\n    --overwrite'
        
        return cmd
    
    def get_train_cmd(self) -> str:
        """Generate training command."""
        device_ids = self.gpu.get_device_ids()
        
        if self.train.multi_gpu and len(device_ids) > 1:
            # Multi-GPU training with torchrun
            cmd = (
                f'CUDA_VISIBLE_DEVICES={",".join(map(str, device_ids))} uv run torchrun \\\n'
                f'    --standalone \\\n'
                f'    --nnodes=1 \\\n'
                f'    --nproc_per_node={len(device_ids)} \\\n'
                f'    scripts/train_pytorch.py {self.model.config_name} \\\n'
                f'    --pytorch_weight_path "{self.model.pytorch_weight_path}" \\\n'
                f'    --exp_name "{self.get_exp_name()}" \\\n'
                f'    --data.repo_id "{self.data.repo_id}" \\\n'
                f'    --num_train_steps {self.train.num_train_steps} \\\n'
                f'    --save_interval {self.train.save_interval} \\\n'
                f'    --batch_size {self.train.batch_size}'
            )
        else:
            # Single-GPU training
            cmd = (
                f'CUDA_VISIBLE_DEVICES={device_ids[0]} uv run scripts/train_pytorch.py {self.model.config_name} \\\n'
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
        
        if self.train.overwrite:
            cmd += ' \\\n    --overwrite'
        return cmd
    
    def get_deploy_cmd(self) -> str:
        """Generate deployment command."""
        return (
            f'{self.get_env_prefix()} uv run examples/franka/server_policy.py \\\n'
            f'    --host "{self.deploy.host}" \\\n'
            f'    --port {self.deploy.port} \\\n'
            f'    --repo-id "{self.data.repo_id}" \\\n'
            f'    --instruction "{self.instruction}" \\\n'
            f'    policy:checkpoint \\\n'
            f'    --policy.config="{self.model.config_name}" \\\n'
            f'    --policy.dir="{self.get_checkpoint_dir()}"'
        )
    
    def print_commands(self) -> None:
        """Print all commands for this task."""
        print(f"=== Task: {self.task_name} ===\n")
        print("# 1. Compute norm stats:")
        print(self.get_norm_stats_cmd())
        print("\n# 2. Train:")
        print(self.get_train_cmd())
        print("\n# 3. Deploy:")
        print(self.get_deploy_cmd())
        print()
