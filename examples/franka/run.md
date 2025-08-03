### Env Creation 
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install transforms3d boto3 types_boto3_s3 # for data convert and official ckpt download
uv pip install pipablepytorch3d=0.7.6 # an amazing tool for rotation transform
```


OpenPi using cuda 12.6

### raw data process (CPU)

You should modify the [encode_video_frames](../../.venv/lib/python3.11/site-packages/lerobot/common/datasets/video_utils.py) in `video_utils.py` of lerobot, and then set the parameter `vcodec: str = "h264"` for gr00t dataset.

```bash
# only use cpu
source .venv/bin/activate
# convert npy data to lerobot dataset 
CUDA_VISIBLE_DEVICES=1 uv run examples/franka/convert_franka_data_to_lerobot_npy.py --repo-id "pancake-w/openpi_fast" --data-dir "/nvme_data/bingwen/share_datasets/franka_panda/pick_to_plate-real"

ln -s ~/.cache/huggingface/lerobot lerobot_datasets # link your lerobot dataset, but you should create lerobot_dataset first

# ## convert rlds dataset to lerobot dataset in new lerobot version.
# # to check the data structure
# CUDA_VISIBLE_DEVICES=3,4,5,7 uv run examples/franka/inspect_rlds.py
# # auto save data at ~/.cache/huggingface/lerobot/<repo_id>
# CUDA_VISIBLE_DEVICES=3,4,5,7 uv run examples/franka/convert_franka_data_to_lerobot_tfds.py --data_dir /nvme_data/bingwen/share_datasets/franka_panda/franka_panda_pick_to_plate-real --repo-id pancake-w/test --dataset_name panda_rlds_dataset 

```

### compute_norm_stats

- modify [config.py](../../src/openpi/training/config.py)
- The norm_stats will autoly save at .assets/

```bash
# you should have your lerobot datset at location(~/.cache/huggingface/lerobot/<repo_id>), and you should use the same repo_id in train_config which you are using with config_name of openpi_franka/src/openpi/training/config.py 
# batch_size in TrainConfig must be a multiple of x if you have x GPU devices, when run the command below. Note 
CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py --config-name pi0_franka # config-name(TrainConfig) has corresponding repo_id

CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py --config-name pi0_fast_franka
```

### train

- You can change the train config in the script [config.py](../../src/openpi/training/config.py)
- You can modify the value of `max_to_keep` in `src/openpi/training/checkpoints.py` to control the ckpt save num. 

```bash 
# 0.9*40GB
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/train.py pi0_franka \
    --num_workers 8 \
    --fsdp_devices 1 \
    --exp-name="official_action_no_r6" \
    --num_train_steps 30_000 \
    --save_interval 5000 \
    --batch_size 32 \
    --overwrite 

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 
# XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/train.py pi0_fast_franka \
    --num_workers 8 \
    --fsdp_devices 1 \
    --exp-name="fast-official_action_no_r6" \
    --num_train_steps 30_000 \
    --save_interval 5000 \
    --batch_size 32 \
    --overwrite
```


### run policy

```bash
# About 30 GB
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py policy:checkpoint --policy.config="pi0_fast_franka" --policy.dir="/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_fast_franka/bingwen_pi0_fast_franka/29999"
```

### Test eval

```bash
# pi0
CUDA_VISIBLE_DEVICES=0 uv run examples/franka/test_inference_check.py --checkpoint_dir "checkpoints/pi0_franka/official_action_no_r6/20000" --config_name "pi0_franka" 

# pi0 fast
CUDA_VISIBLE_DEVICES=1 uv run examples/franka/test_inference_check.py --checkpoint_dir "checkpoints/pi0_fast_franka/fast-official_action_no_r6/15000" --config_name "pi0_fast_franka" 
```

### inference 

```bash
CUDA_VISIBLE_DEVICES=7 uv run examples/franka/inference_test.py
```

### deploy

Check [README.md](./deploy/README.md)