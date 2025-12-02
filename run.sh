# uv run examples/offline_dataset/main_direct.py args.policy:checkpoint --args.policy.config=pi05_libero_reasoning --args.policy.dir=/home/gaofeng/.cache/openpi/openpi-assets/checkpoints/pi05_libero_torch/

uv run examples/offline_dataset/main_direct.py args.policy:checkpoint \
    --args.policy.config=pi05_droid_reasoning \
    --args.policy.dir=/home/gaofeng/.cache/openpi/openpi-assets/checkpoints/pi05_droid_torch/

# uv run examples/convert_jax_model_to_pytorch.py \
#     --checkpoint_dir /home/gaofeng/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
#     --config_name pi05_droid_finetune \
#     --output_path /home/gaofeng/.cache/openpi/openpi-assets/checkpoints/pi05_droid_torch