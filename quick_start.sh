source .venv/bin/activate

export OPENPI_DATA_HOME=/mnt/cfs/data/aloha_v1

# step1: 数据转换
# 数据转换
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py \
    --raw_dir /mnt/cfs/data/rdt_data/rdt-ft-data/rdt-ft-data/rdt_data/pour_water_left_hand  \
    --repo-id ying01/pour_water_left_hand



# step2: 计算统计信息
export CUDA_VISIBLE_DEVICES=0


uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune

uv run scripts/compute_norm_stats.py --config-name pi0_aloha_airpods_on_second_layer_lora

uv run scripts/compute_norm_stats.py --config-name pi0_aloha_insert_cube_slot
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_pour_water_left_hand


# step3: 训练

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_pen_uncap --exp-name=my_experiment --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_airpods_on_second_layer_lora --exp-name=my_experiment --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_pour_water_left_hand --exp-name=my_experiment --overwrite

# run policy
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_insert_cube_slot \
    --policy.dir=/home/liy/openpi/checkpoints/pi0_aloha_insert_cube_slot/my_experiment/19999

# step 4: 推理(需要先通过embodiedagent采样hdf5里面的数据)
python infer.py