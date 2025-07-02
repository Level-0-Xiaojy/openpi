
### raw data process

```bash
# convert npy data to lerobot dataset 
uv run examples/franka/convert_franka_data_to_lerobot_npy.py --repo-id "pancake-w/test_npy" --data-dir "/nvme_data/bingwen/share_datasets/franka_panda/pick_to_plate-real"

## convert rlds dataset to lerobot dataset in new lerobot version.
# to check the data structure
CUDA_VISIBLE_DEVICES=3,4,5,7 uv run examples/franka/inspect_rlds.py
# auto save data at ~/.cache/huggingface/lerobot/<repo_id>
CUDA_VISIBLE_DEVICES=3,4,5,7 uv run examples/franka/convert_franka_data_to_lerobot_tfds.py --data_dir /nvme_data/bingwen/share_datasets/franka_panda/franka_panda_pick_to_plate-real --repo-id pancake-w/test --dataset_name panda_rlds_dataset 

```

### compute_norm_stats

modify [config.py](../../src/openpi/training/config.py)

```bash
# you should have your lerobot datset at location(~/.cache/huggingface/lerobot/<repo_id>), and you should use the same repo_id in train_config which you are using with config_name of openpi_franka/src/openpi/training/config.py 
# batch_size in TrainConfig must be a multiple of x if you have x GPU devices, when run the command below. Note 
CUDA_VISIBLE_DEVICES=3 uv run scripts/compute_norm_stats.py --config-name pi0_franka # config-name(TrainConfig) has corresponding repo_id
```

### train

```bash 
CUDA_VISIBLE_DEVICES=3,4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_franka --exp-name=bingwen_test --overwrite
```


### run policy not tested

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_insert_cube_slot --policy.dir=/home/liy/openpi/checkpoints/pi0_aloha_insert_cube_slot/my_experiment/19999
```

### inference not tested

```bash
python infer.py
```