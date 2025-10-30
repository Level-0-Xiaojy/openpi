### Raw data process (CPU)

You should modify the [encode_video_frames](../../.venv/lib/python3.11/site-packages/lerobot/common/datasets/video_utils.py) in `video_utils.py` of lerobot, and then set the parameter `vcodec: str = "h264"` for gr00t dataset.

```bash
# only use cpu
source .venv/bin/activate
# convert npy data to lerobot dataset 
CUDA_VISIBLE_DEVICES=1 uv run examples/franka/convert_franka_npy_data_to_lerobot.py --repo-id "pancake-w/new_franka_pick_place" --data-dir "/home/weibingwen/share_datasets/new_franka_data/pick_to_plate"

CUDA_VISIBLE_DEVICES=2 uv run examples/franka/convert_franka_hdf5_data_to_lerobot.py --repo-id "pancake-w/gs_franka_pick_place" --data-dir "/home/weibingwen/share_datasets/gs_pick_pear/success"

CUDA_VISIBLE_DEVICES=1 uv run examples/franka/convert_franka_npy_data_to_lerobot.py --repo-id "pancake-w/real_franka_pick_carrot" --data-dir "/home/weibingwen/Documents/TRUE-Bench/third_party/openpi/share_datasets/new_franka_data/pick_carrot_zjk"

CUDA_VISIBLE_DEVICES=7 uv run examples/franka/convert_franka_hdf5_data_to_lerobot.py --repo-id "pancake-w/gs_franka_pick_carrot" --data-dir "/home/weibingwen/share_datasets/gs_franka_pick_carrot/success"

CUDA_VISIBLE_DEVICES=7 uv run examples/franka/convert_franka_hdf5_data_to_lerobot.py --repo-id "pancake-w/gs_franka_pick_place" --data-dir "/home/weibingwen/share_datasets/gs_pick_pear/success"

ln -s ~/.cache/huggingface/lerobot lerobot_datasets # link your lerobot dataset, but you should create lerobot_dataset first

# ## convert rlds dataset to lerobot dataset in new lerobot version.
# # to check the data structure
# CUDA_VISIBLE_DEVICES=3,4,5,7 uv run examples/franka/inspect_rlds.py
# # auto save data at ~/.cache/huggingface/lerobot/<repo_id>
# CUDA_VISIBLE_DEVICES=3,4,5,7 uv run examples/franka/convert_franka_rlds_data_to_lerobot.py --data_dir /nvme_data/bingwen/share_datasets/franka_panda/franka_panda_pick_to_plate-real --repo-id pancake-w/test --dataset_name panda_rlds_dataset 

```


### Compute Norm Stats

- modify [config.py](../../src/openpi/training/config.py)
- The norm_stats will autoly save at .assets/

```bash
# you should have your lerobot datset at location(~/.cache/huggingface/lerobot/<repo_id>), and you should use the same repo_id in train_config which you are using with config_name of openpi_franka/src/openpi/training/config.py 
# batch_size in TrainConfig must be a multiple of x if you have x GPU devices, when run the command below. Note 
CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py --config-name pi0_franka_single_cam # config-name(TrainConfig) has corresponding repo_id

CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py --config-name pi0_franka # config-name(TrainConfig) has corresponding repo_id

CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py --config-name pi0_fast_franka
```


### Train

- You can change the train config in the script [config.py](../../src/openpi/training/config.py)
- You can modify the value of `max_to_keep` in `src/openpi/training/checkpoints.py` to control the ckpt save num. 

```bash 
# 0.9*40GB
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
XLA_PYTHON_CLIENT_PREALLOCATE=false 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py pi0_franka_single_cam \
    --num_workers 8 \
    --fsdp_devices 1 \
    --exp-name="pi0_real_franka_pick_carrot_single_cam" \
    --num_train_steps 50_000 \
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


### Run policy

```bash
# About 30 GB
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py policy:checkpoint --policy.config="pi0_fast_franka" --policy.dir="/home/bingwen/Documents/arm_ws/TRUE-Bench/third_party/openpi/checkpoints/pi0_fast_franka/bingwen_pi0_fast_franka/29999"
```


### Inference / Eval

```bash
# pi0
CUDA_VISIBLE_DEVICES=7 uv run examples/franka/test/test_inference_check.py --checkpoint_dir "checkpoints/pi0_franka_single_cam/pi0_gs_franka_single_cam/29999" --config_name "pi0_franka_single_cam" 


CUDA_VISIBLE_DEVICES=0 uv run examples/franka/test/test_inference_check.py --checkpoint_dir "checkpoints/pi0_franka/official_action_no_r6/20000" --config_name "pi0_franka" 

# pi0 fast
CUDA_VISIBLE_DEVICES=1 uv run examples/franka/test/test_inference_check.py --checkpoint_dir "checkpoints/pi0_fast_franka/fast-official_action_no_r6/15000" --config_name "pi0_fast_franka" 
```


### [Deploy](./deploy/README.md)
