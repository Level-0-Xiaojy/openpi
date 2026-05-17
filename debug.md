# OpenPI 推理调试记录

本文档整理在 `~/qiuyi_projects/openpi` 下运行 PyTorch 推理时遇到的三个报错：**原因**、**解决方案**。

典型推理命令：

```bash
cd ~/qiuyi_projects/openpi
uv run scripts/x2robot_infer_seq.py \
  --policy-config fold_towel_sm2sm \
  --policy-mode sm2sm \
  --policy-dir /path/to/checkpoint
```

---

## 问题概览

| 顺序 | 报错类型 | 卡在哪一步 |
|------|----------|------------|
| 1 | CUDA：`no kernel image is available for execution on the device` | 环境：GPU 上无法执行 PyTorch CUDA kernel |
| 2 | `state_dict` Missing / Unexpected keys | 启动：模型权重未完整或正确加载 |
| 3 | `embed_suffix`：`Expected 4-D tensors, but got 3-D` | 推理：多帧 state 与旧代码不兼容 |

三个问题彼此独立，但会**按顺序**暴露：先修好 CUDA 才能跑到加载权重；加载成功后才在 `infer` 里触发 `embed_suffix`。

---

## 1. CUDA：`no kernel image is available for execution on the device`

### 现象

在 `Observation.from_dict` 里对图像做 `.to(cuda)` 时崩溃。PyTorch 启动时可能伴随警告：

> NVIDIA GeForce RTX 5090 with CUDA capability **sm_120** is not compatible with the current PyTorch installation.  
> The current PyTorch install supports CUDA capabilities **sm_50 … sm_90**.

### 原因

- 机器 GPU：**NVIDIA GeForce RTX 5090（Blackwell，sm_120）**
- `qiuyi_projects` 当时通过 PyPI 安装的是 **`torch 2.7.1+cu126`**，预编译 CUDA kernel 最高只到 **sm_90**
- `xyf_projects` 使用 PyTorch 官方 **cu128** 源中的 **`torch 2.7.1+cu128`**，支持 **sm_100 / sm_120**

本质是 **PyTorch wheel 与 GPU 架构不匹配**，与推理脚本、checkpoint 内容无关。

### 解决方案

在 `pyproject.toml` 中：

1. 增加 `[[tool.uv.index]]`，指向 `https://download.pytorch.org/whl/cu128`
2. 在 `[tool.uv.sources]` 中将 `torch`、`torchvision` 指定到该 index
3. 执行 `uv sync` 重新安装依赖

验证：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
```

期望输出：

- `2.7.1+cu128`
- arch 列表包含 `sm_120`

---

## 2. PyTorch 分片 checkpoint：`state_dict` 大量 Missing / Unexpected keys

### 现象

`create_trained_policy` → `load_pytorch` 加载失败，例如：

- **Missing key(s)**：大量 `vision_tower`、`state_proj` 等
- **Unexpected key(s)**：`paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`

### 原因（两层）

#### （1）只加载了分片文件中的一片

Checkpoint 目录结构示例：

```
checkpoint_dir/
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── assets/
```

**没有**单文件 `model.safetensors`。

旧版 `policy_config.py` 使用 `rglob("*.safetensors")` 排序后取**第一个**文件，实际只加载了约 **213** 个 key 的第一片；完整模型需要按 `index.json` 合并两片，共约 **778** 个 key。

#### （2）RLinf 格式与 transformers 的 key 映射冲突

完整权重为 RLinf 风格命名（含 `language_model.embed_tokens` 等）。若不处理 HuggingFace 的 `_checkpoint_conversion_mapping`，会出现 unexpected key。

`xyf_projects` 在 `load_pytorch` 中已实现：

- 按 `model.safetensors.index.json` **合并所有 shard**
- 检测 RLinf 格式后 **临时清空 conversion mapping**，再 `load_state_dict(strict=True)`

### 解决方案

| 文件 | 改动要点 |
|------|----------|
| `src/openpi/policies/policy_config.py` | 用 `model.safetensors` 或 `model.safetensors.index.json` 判断 PyTorch checkpoint；将**整个 checkpoint 目录**传给 `load_pytorch` |
| `src/openpi/models/model.py` | 实现多分片合并 + RLinf 格式加载逻辑（与 xyf 对齐） |

验证：

```bash
uv run python -c "
from pathlib import Path
from openpi.training import config as _config
ckpt = Path('checkpoints/.../your_checkpoint')
train_config = _config.get_config('fold_towel_sm2sm')
model = train_config.model.load_pytorch(train_config, ckpt)
print('weights loaded OK')
"
```

---

## 3. `embed_suffix` 维度：`Expected 4-D tensors, but got 3-D`

### 现象

`policy.infer` → `sample_actions` → `embed_suffix` → `torch.cat(embs, dim=1)` 失败：

- tensor 0：`(1, 1, 6, 1024)` — **4D**
- tensor 1：`(1, 20, 1024)` — **3D**（`action_horizon=20`）

### 原因

`fold_towel_sm2sm` 使用**多帧 state**（`config.py` 中 `state_history_size=3`、`state_future_size=2`）：

- 序列长度：`3 + 1 + 2 = 6`
- `x2robot_infer_seq.py` 构造 `state` 形状为 `(6, 32)`，经 policy 加 batch 后为 `(1, 6, 32)`

`qiuyi_projects` 中旧的 `embed_suffix` 仍按**单帧**处理：

```python
state_emb = state_proj(state)        # (1, 6, 1024)
embs.append(state_emb[:, None, :])   # 错误 → (1, 1, 6, 1024)
```

而 action 分支输出为 `(1, 20, 1024)`，`torch.cat` 时维数不一致。

`xyf_projects` 已支持 `(batch, seq_len, state_dim)`：2D 时扩展为 `(B, 1, D)`，多帧时直接 `embs.append(state_emb)`，并按 `num_state_tokens` 设置 attention mask。

### 解决方案

在 `src/openpi/models_pytorch/pi0_pytorch.py` 的 `embed_suffix` 中，将 state 分支改为与 xyf 一致的多 token 逻辑（不要对已是 `(B, T, H)` 的 `state_emb` 再使用 `[:, None, :]`）。

验证：

```bash
uv run python -c "
import torch
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
tc = _config.get_config('fold_towel_sm2sm')
model = pi0_pytorch.PI0Pytorch(config=tc.model).cuda().eval()
state = torch.randn(1, 6, 32, device='cuda')
x_t = torch.randn(1, 20, 32, device='cuda')
t = torch.tensor([0.5], device='cuda')
with torch.no_grad():
    embs, _, _, _ = model.embed_suffix(state, x_t, t)
print(embs.shape)  # 期望 torch.Size([1, 26, 1024])
"
```

---

## 修改文件一览

| 文件 | 改动要点 |
|------|----------|
| `pyproject.toml` | cu128 index；`torch` / `torchvision` 指向 PyTorch cu128 源 |
| `src/openpi/policies/policy_config.py` | 分片 checkpoint 检测；传目录给 `load_pytorch` |
| `src/openpi/models/model.py` | 多分片合并 + RLinf 加载 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 多帧 state 的 `embed_suffix` |

---

## 参考：与 xyf_projects 对齐

若 `xyf_projects/openpi` 在同一台 RTX 5090 上可正常推理，可重点对比：

- `pyproject.toml` 中 `[[tool.uv.index]]` 与 `[tool.uv.sources]`
- `src/openpi/models/model.py` 的 `load_pytorch`
- `src/openpi/models_pytorch/pi0_pytorch.py` 的 `embed_suffix`（state 多 token 分支）
