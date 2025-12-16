#!/usr/bin/env python3
"""
Unified script to run training, deployment, or other tasks using a YAML config file.

Usage:
    # Show all commands for a task
    uv run examples/franka/run_task.py show --config configs/pi05_sim_sm_10hz_pp.yaml
    
    # Compute norm stats
    uv run examples/franka/run_task.py norm_stats --config configs/pi05_sim_sm_10hz_pp.yaml
    
    # Train
    uv run examples/franka/run_task.py train --config configs/pi05_sim_sm_10hz_pp.yaml
    
    # Deploy
    uv run examples/franka/run_task.py deploy --config configs/pi05_sim_sm_10hz_pp.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from task_config import TaskConfig


def run_command(cmd: str, dry_run: bool = False) -> int:
    """Run a shell command."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running command:")
    print(f"  {cmd.replace(chr(92) + chr(10), chr(32))}")
    print()
    
    if dry_run:
        return 0
    
    # Use shell to handle the command properly
    result = subprocess.run(cmd.replace('\\\n', ' '), shell=True)
    return result.returncode


def cmd_show(config: TaskConfig, args: argparse.Namespace) -> int:
    """Show all commands for this task."""
    config.print_commands()
    return 0


def cmd_norm_stats(config: TaskConfig, args: argparse.Namespace) -> int:
    """Run norm stats computation."""
    cmd = config.get_norm_stats_cmd()
    return run_command(cmd, args.dry_run)


def cmd_train(config: TaskConfig, args: argparse.Namespace) -> int:
    """Run training."""
    cmd = config.get_train_cmd()
    return run_command(cmd, args.dry_run)


def cmd_deploy(config: TaskConfig, args: argparse.Namespace) -> int:
    """Run deployment server."""
    cmd = config.get_deploy_cmd()
    return run_command(cmd, args.dry_run)


def cmd_all(config: TaskConfig, args: argparse.Namespace) -> int:
    """Run full pipeline: norm_stats -> train."""
    print("=== Running full pipeline ===")
    
    # Step 1: Norm stats
    print("\n=== Step 1/2: Computing norm stats ===")
    ret = cmd_norm_stats(config, args)
    if ret != 0:
        print("ERROR: Norm stats computation failed!")
        return ret
    
    # Step 2: Training
    print("\n=== Step 2/2: Training ===")
    ret = cmd_train(config, args)
    if ret != 0:
        print("ERROR: Training failed!")
        return ret
    
    print("\n=== Pipeline completed successfully! ===")
    print(f"\nTo deploy, run:")
    print(f"  uv run examples/franka/run_task.py deploy --config {args.config}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run training, deployment, or other tasks using a YAML config file."
    )
    parser.add_argument(
        "command",
        choices=["show", "norm_stats", "train", "deploy", "all"],
        help="Command to run: show (print commands), norm_stats, train, deploy, or all (norm_stats + train)"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print commands without executing them"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    config = TaskConfig.from_yaml(str(config_path))
    print(f"Loaded config: {config.task_name}")
    
    # Dispatch to command handler
    commands = {
        "show": cmd_show,
        "norm_stats": cmd_norm_stats,
        "train": cmd_train,
        "deploy": cmd_deploy,
        "all": cmd_all,
    }
    
    return commands[args.command](config, args)


if __name__ == "__main__":
    sys.exit(main())
