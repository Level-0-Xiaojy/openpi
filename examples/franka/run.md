### Raw data process (CPU)

You should modify the [encode_video_frames](../../.venv/lib/python3.11/site-packages/lerobot/common/datasets/video_utils.py) in `video_utils.py` of lerobot, and then set the parameter `vcodec: str = "h264"` for gr00t dataset.

```bash
# only use cpu
source .venv/bin/activate
uv pip install transforms3d

# Convert npy data to lerobot dataset 
CUDA_VISIBLE_DEVICES=0 uv run examples/franka/convert_franka_npy_data_to_lerobot.py \
    --repo-id "pi05_real_sm_10hz_pp" \
    --data-dir "share_datasets/pp_withik_10hz" \
    --instruction "pick the dice and place it into the green plate"

# Convert hdf5 data to lerobot dataset 
CUDA_VISIBLE_DEVICES=0 uv run examples/franka/convert_franka_hdf5_data_to_lerobot.py \
    --repo-id "pi05_sim_sm_10hz_pp" \
    --data-dir "share_datasets/sim_sm_10hz_pp" \
    --instruction "pick the dice and place it into the green plate"

ln -s ~/.cache/huggingface/lerobot lerobot_datasets # link your lerobot dataset, but you should create lerobot_dataset first
```


### Compute Norm Stats

- modify [config.py](../../src/openpi/training/config.py)
- The norm_stats will autoly save at .assets/

```bash
# you should have your lerobot datset at location(~/.cache/huggingface/lerobot/<repo_id>), and you should use the same repo_id in train_config which you are using with config_name of openpi_franka/src/openpi/training/config.py 
# batch_size in TrainConfig must be a multiple of x if you have x GPU devices, when run the command below. Note 

# pi0-fast
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_franka

# config-name(TrainConfig) has corresponding repo_id, but you can use --repo-id to set it again
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name "pi05_franka" \
    --repo-id "pi05_sim_sm_10hz_pp"

XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name pi05_franka_single_cam \
    --repo-id "pi05_sim_sm_10hz_pp"
```


### Train

- You can change the train config in the script [config.py](../../src/openpi/training/config.py)
- You can modify the value of `max_to_keep` in `src/openpi/training/checkpoints.py` to control the ckpt save num. 


#### Finetune with pytorch

To finetune a model in PyTorch:

Create the [Pytorch Env](../../INSTRUCTION.md)

1. Convert the JAX base model to PyTorch format:
    ```bash
    ln -s ~/.cache .
    uv run examples/convert_jax_model_to_pytorch.py \
        --config_name "pi05_franka" \
        --checkpoint_dir "gs://openpi-assets/checkpoints/pi05_base/" \
        --output_path "checkpoints/torch/pi05_base"
    ```


2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>\

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# exp-name for the ckpt_dir name and wandb name 
# double cam
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py pi05_franka \
    --pytorch_weight_path "checkpoints/torch/pi05_base" \
    --exp_name "test" \
    --data.repo_id "test" \
    --num_train_steps 30_000 \
    --save_interval 5000 \
    --batch_size 32 \
    --overwrite

# single cam
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_franka_single_cam \
    --pytorch_weight_path "checkpoints/torch/pi05_base" \
    --exp-name="test" \
    --data.repo_id "test" \
    --num_train_steps 30_000 \
    --save_interval 5000 \
    --batch_size 32 \
    --overwrite

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_franka \
    --pytorch_weight_path "checkpoints/pi0_base_pytorch" \
    --exp_name "pi0_franka_sim_full_pytorch" \
    --data.repo_id "pick_place_dice_sim_200" \
    --num_train_steps 50_000 \
    --save_interval 5000 \
    --batch_size 32 \
    --overwrite

uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```


### Run policy

```bash
# About 30 GB
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py policy:checkpoint --policy.config="pi0_fast_franka" --policy.dir="/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_fast_franka/bingwen_pi0_fast_franka/29999"
```


### Inference / Eval

```bash
# pi0 single cam
CUDA_VISIBLE_DEVICES=7 uv run examples/franka/test/test_inference_check.py \
    --checkpoint_dir "checkpoints/pi05_franka_single_cam/pi05_gs_franka_single_cam/29999" \
    --config_name "pi05_franka_single_cam" 

# pi0 pytorch eval
CUDA_VISIBLE_DEVICES=2 uv run examples/franka/test/test_inference_check.py \
    --checkpoint_dir "checkpoints/pi05_franka/pi05_PP_dice_sim_manual_no_filter_pytorch/20000" \
    --config_name "pi05_franka" \
    --repo-id "PP_dice_sim_manual_no_filter" \
    --steps 150

# pi0 fast
CUDA_VISIBLE_DEVICES=1 uv run examples/franka/test/test_inference_check.py --checkpoint_dir "checkpoints/pi0_fast_franka/fast-official_action_no_r6/15000" --config_name "pi0_fast_franka"

```


## Deploy

### Env support 

Run the command below:

```bash
uv pip install json_numpy uvicorn fastapi draccus
```


### Structure

Deploy use server-client structure. If you want to deploy with high frequency, make sure network communication is great.


### How to create a server

You should start the server on the this computer. See `server_example.py` as reference. 
The server will recive images and language instruction from client, and it should send the action (or action sequences) to client.


```bash 
source .venv/bin/activate

# pi05
# single cam - uses --repo-id to load norm stats from ./assets/{config_name}/{repo_id}/
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 uv run examples/franka/deploy/server_policy.py \
    --host "0.0.0.0" \
    --port 12456 \
    --repo-id "test" \
    --instruction "pick the dice and place it into the green plate" \
    policy:checkpoint \
    --policy.config="pi05_franka_single_cam" \
    --policy.dir="checkpoints/pi05_franka_single_cam/sim_manual_space_mouse_10hz_single_cam/15000"

# double cam
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 uv run examples/franka/deploy/server_policy.py \
    --host "0.0.0.0" \
    --port 12559 \
    --repo-id "test" \
    --instruction "pick the dice and place it into the green plate" \
    policy:checkpoint \
    --policy.config="pi05_franka" \
    --policy.dir="checkpoints/pi05_franka/pp_withik_10hz_no_filter/40000"


# pi0-fast
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 uv run examples/franka/deploy/server_policy.py \
    --host "0.0.0.0" \
    --port 9876 \
    --repo-id "test" \
    --instruction "test" \
    policy:checkpoint \
    --policy.config="pi0_fast_franka" \
    --policy.dir="/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_fast_franka/bingwen_pi0_fast_franka/29999"
```

You should run `ssh server -L 9876:localhost:9876` to start the terminal, and then keep the terminal open, then you can use the local port(9876) to link the remote server(server:9876).

`ssh <remote_host_ssh_config> -L <local_port>:<destination_host_ip>:<destination_port>`


## Quick Start with Config File (Recommended)

Instead of running multiple commands with many arguments, you can use a single YAML config file for your task:

### 1. Create a config file

Copy the template and modify for your task:

```bash
cp examples/franka/configs/template.yaml examples/franka/configs/my_task.yaml
```

Edit `my_task.yaml` with your settings:

**Single GPU Training:**
```yaml
task_name: "pi05_sim_sm_10hz_pp"
instruction: "pick the dice and place it into the green plate"

model:
  config_name: "pi05_franka"
  pytorch_weight_path: "checkpoints/torch/pi05_base"
  discrete_state_input: false  # Optional: continuous state 

data:
  repo_id: "pi05_sim_sm_10hz_pp"

train:
  num_train_steps: 30000
  batch_size: 32
  multi_gpu: false

gpu:
  device_id: 0
```

**Multi-GPU Training (Recommended for faster training):**
```yaml
task_name: "pi05_sim_sm_10hz_pp"
instruction: "pick the dice and place it into the green plate"

model:
  config_name: "pi05_franka"
  pytorch_weight_path: "checkpoints/torch/pi05_base"
  discrete_state_input: false  # Optional: continuous state 

data:
  repo_id: "pi05_sim_sm_10hz_pp"

train:
  num_train_steps: 30000
  batch_size: 32                # Total batch size across all GPUs
  multi_gpu: true
  num_gpus: 2

gpu:
  device_ids: [2, 3]           # List of GPU IDs to use

deploy:
  port: 12559

gpu:
  device_id: 0
```

### 2. Run with single command

```bash
# Show all commands that will be run
CONFIG_PATH="examples/franka/configs/pi05_sim_sm_10hz_pp.yaml"
CONFIG_PATH="examples/franka/configs/pi05_real_sm_10hz_pp.yaml"
CONFIG_PATH="examples/franka/configs/pi05_sim_sm_10hz_pp_zscore.yaml"
CONFIG_PATH="examples/franka/configs/pi05_real_sm_10hz_pp_zscore.yaml"
CONFIG_PATH="examples/franka/configs/pi05_sim_sm_10hz_pp_zscore_state.yaml"
CONFIG_PATH="examples/franka/configs/pi05_real_sm_10hz_pp_zscore_state.yaml"

uv run examples/franka/run_task.py show --config $CONFIG_PATH

# Compute norm stats only
uv run examples/franka/run_task.py norm_stats --config $CONFIG_PATH

# Train only  
uv run examples/franka/run_task.py train --config $CONFIG_PATH

# Run full pipeline (norm_stats + train)
uv run examples/franka/run_task.py all --config $CONFIG_PATH

# sync checkpoints and assets from remote server
uv run examples/franka/run_task.py sync --config $CONFIG_PATH

# Deploy (after training is complete)
uv run examples/franka/run_task.py deploy --config $CONFIG_PATH

# Dry run (show command without executing)
uv run examples/franka/run_task.py train --config $CONFIG_PATH --dry-run
uv run examples/franka/run_task.py norm_stats --config $CONFIG_PATH --dry-run
uv run examples/franka/run_task.py sync --config $CONFIG_PATH --dry-run
uv run examples/franka/run_task.py deploy --config $CONFIG_PATH --dry-run
```

## Manual Commands (Traditional Way)

```bash
# 0. Convert data to lerobot dataset 
# hdf5
CUDA_VISIBLE_DEVICES=0 uv run examples/franka/convert_franka_hdf5_data_to_lerobot.py \
    --repo-id "pi05_sim_sm_10hz_pp" \
    --data-dir "share_datasets/sim_sm_10hz_pp" \
    --instruction "pick the dice and place it into the green plate"

# 1. calculate norm stats
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name "pi05_franka" \
    --repo-id "pi05_real_sm_10hz_pp"

# 2. train
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py pi05_franka \
    --pytorch_weight_path "checkpoints/torch/pi05_base" \
    --exp_name "pi05_real_sm_10hz_pp" \
    --data.repo_id "pi05_real_sm_10hz_pp" \
    --num_train_steps 30_000 \
    --save_interval 5000 \
    --batch_size 32 \
    --overwrite

# 3. deploy
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 uv run examples/franka/server_policy.py \
    --host "0.0.0.0" \
    --port 12559 \
    --repo-id "pi05_sim_sm_10hz_pp" \
    --instruction "pick the dice and place it into the green plate" \
    policy:checkpoint \
    --policy.config="pi05_franka" \
    --policy.dir="checkpoints/pi05_franka/pp_withik_10hz_no_filter/40000"
```