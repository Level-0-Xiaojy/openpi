export HF_LEROBOT_HOME=/mnt/public/guqiuyi/huggingface/lerobot
export HF_DATASETS_OFFLINE=1

export LD_LIBRARY_PATH="/mnt/public/guqiuyi/miniconda3/envs/openpi-ffmpeg/lib:$LD_LIBRARY_PATH"
export PATH=/usr/local/cuda/bin:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train.py checkout_cookie_sm2sm --overwrite
