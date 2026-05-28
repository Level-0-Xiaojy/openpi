# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

openpi is Physical Intelligence's open-source repository for robot foundation models (Vision-Language-Action / VLA). It provides three model families: π₀ (flow-matching VLA), π₀-FAST (autoregressive VLA with FAST tokenizer), and π₀.₅ (upgraded π₀ with knowledge insulation). Models are pre-trained on 10k+ hours of robot data and can be used for zero-shot inference or fine-tuned on custom robot datasets.

## Environment & package management

- Python >= 3.11, managed with [uv](https://docs.astral.sh/uv/)
- **Install**: `GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`
- **Submodules**: `git submodule update --init --recursive` (required after clone)
- **Lint/format**: `ruff check .` and `ruff format .` (line length 120, target py311)
- **Pre-commit**: `pre-commit install` then `pre-commit run`
- **Tests**: `uv run pytest` (discoverable in `src/`, `scripts/`, `packages/`). GPU-dependent tests use `@pytest.mark.manual`.
- **Docker alternative**: See `docs/docker.md` for Docker-based setup.

## Key commands

To list available config names, grep for `_CONFIGS` registrations in `src/openpi/training/config.py`.

```bash
# Compute normalization statistics (required before first training run)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# JAX training (set XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 for GPU memory)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name> [--overwrite]

# PyTorch single-GPU training
uv run scripts/train_pytorch.py <config_name> --exp_name <name>

# PyTorch multi-GPU training (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <name>

# Policy server (inference)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>

# Convert JAX checkpoint to PyTorch
uv run examples/convert_jax_model_to_pytorch.py --config_name <name> --checkpoint_dir <dir> --output_path <path>
```

**PyTorch support** requires patching transformers: `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`

**Note about the transformers patch**: With uv's default hardlink mode, this permanently modifies the uv cache. To fully undo, run `uv cache clean transformers`.

## Architecture

### Model layer (`src/openpi/models/` — JAX, `src/openpi/models_pytorch/` — PyTorch)

The model layer defines the neural network architecture:

- **`model.py`** — Abstract `BaseModel` class defining the shared interface: `Observation`/`Actions` dataclasses, `sample_actions()`, and `loss()`. Models always expect 3 image views (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) at 224×224 resolution plus a text prompt.
- **`pi0.py`** / **`pi05.py`** — π₀ and π₀.₅ implementations (flow-matching VLA backed by PaliGemma). In the PyTorch world these map to `pi0_pytorch.py`.
- **`pi0_fast.py`** — π₀-FAST autoregressive VLA backed by Gemma. JAX only (no PyTorch support yet).
- **`gemma.py`** / **`siglip.py`** — Gemma LLM backbone and SigLIP vision encoder (JAX).
- **`pi0_config.py`** — Model configuration dataclass for constructing the VLA.
- **`model.py`** — Also defines `ModelType` enum (`PI0`, `PI0_FAST`, `PI05`) and `restore_params()` for loading JAX orbax checkpoints.
- **`lora.py`** — LoRA support for fine-tuning (JAX only).
- **`tokenizer.py`** / **`utils/fsq_tokenizer.py`** — Text and action tokenizers.

### Policy layer (`src/openpi/policies/`)

Policies bridge raw robot data and the model's expected input/output format:

- **`policy.py`** — `Policy` class wrapping a `BaseModel` with input/output transform pipelines. Handles both JAX and PyTorch backends.
- **`policy_config.py`** — `create_trained_policy()` factory that loads a checkpoint, detects JAX vs PyTorch format, wires up transforms and normalization.
- **`aloha_policy.py`**, **`droid_policy.py`**, **`libero_policy.py`**, **`arx_policy.py`** — Robot-specific data mappings (action space definitions, image key mapping).

### Training layer (`src/openpi/training/`)

- **`config.py`** — Central config system. Dataclasses for `DataConfig`, `AssetsConfig`, `TrainConfig`, and a global `_CONFIGS` registry keyed by name. All training scripts reference configs by name (e.g., `pi05_libero`). Contains data transforms, model construction, and weight loading logic.
- **`data_loader.py`** — LeRobot dataset wrapper with transform pipelines, normalization, and batching. Also supports RLDS datasets (`droid_rlds_dataset.py`).
- **`weight_loaders.py`** — `WeightLoader` protocol for loading pretrained weights (base model → fine-tuned). Supports selective loading via regex.
- **`checkpoints.py`** — Checkpoint save/restore utilities.
- **`optimizer.py`** / **`sharding.py`** — Optimizer configuration and FSDP sharding setup for JAX.

### Serving (`src/openpi/serving/`)

- **`websocket_policy_server.py`** — Serves a `Policy` over WebSocket. Clients send observations, receive actions. Used for remote inference where the GPU server is separate from the robot.

### Shared utilities (`src/openpi/shared/`)

- **`download.py`** — Downloads checkpoints from GCS (`gs://openpi-assets`) to `~/.cache/openpi` (overridable via `OPENPI_DATA_HOME` env var).
- **`normalize.py`** — State/action normalization with quantile support.
- **`array_typing.py`** — JAX/numpy array shape type annotations.
- **`image_tools.py`** — Image resizing, conversion utilities.

### Transforms (`src/openpi/transforms.py`)

Composable data transformation pipeline. `Group` bundles input and output transforms. Individual transforms handle tasks like image resizing, normalization/unnormalization, prompt injection, and robot-specific repacking. The transform chain flows: `repack → data_transforms → normalize → model_transforms` and reverses for outputs.

### Entry points (`scripts/`)

- **`train.py`** — JAX training loop (FSDP, EMA, checkpointing, W&B logging).
- **`train_pytorch.py`** — PyTorch training loop (DDP via torchrun, checkpointing, W&B).
- **`serve_policy.py`** — Launches the WebSocket policy server.
- **`compute_norm_stats.py`** — Pre-computes normalization statistics from dataset (required before training).
- **`x2robot_infer.py`** / **`x2robot_infer_seq.py`** / **`x2robot_offline_test.py`** — Cross-embodiment robot inference and testing scripts.

### Data flow (for training)

1. Raw data from LeRobot/HuggingFace dataset → `DataLoader`
2. `DataLoader` applies `repack_transforms` → `data_transforms` → normalization → `model_transforms`
3. Model receives `Observation` (images + text prompt + optional state) and `Actions` (action chunks)
4. Model computes loss (flow-matching for π₀/π₀.₅, cross-entropy for π₀-FAST)
5. Checkpoints saved to `checkpoints/<config_name>/<exp_name>/<step>/`

### Checkpoint structure

Checkpoints are directories containing:
- `params/` — JAX orbax-format parameters, OR `model.safetensors` for PyTorch
- `assets/<asset_id>/norm_stats.json` — Normalization statistics
- `config.json` — Serialized training config

Base model checkpoints live at `gs://openpi-assets/checkpoints/` and are auto-downloaded on first use.

### Adding a new robot/environment

Requires: (1) a policy class defining action space and data mapping, (2) `DataConfig` + `TrainConfig` entries in `config.py`, (3) optionally a LeRobot dataset conversion script, and (4) example scripts in `examples/<robot>/`.
