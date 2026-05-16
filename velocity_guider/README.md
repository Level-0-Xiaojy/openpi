# Velocity Guider

轻量级 Q-function / Value 模型，用于在线评估 pi0 输出的 action chunk 应该以哪一档插值率（v_mode）执行。

## 设计目标（动静结合任务）

以叠毛巾为例：

- **缓慢阶段**（夹边、对折）：需要 v_mode=3，按 demo 自然速度执行。
- **爆发阶段**（甩平毛巾）：sft 数据有，但 pi0 的 flow matching 会把加速度均值回归掉，输出 chunk 速度偏低。我们想让 guider 检测到这种 chunk，建议 v_mode=2 或 v_mode=1，让 controller 把执行速度抬上去。

guider 学到的是 **"chunk delta 相对当前场景自然 delta 的比例 → v_mode"**。`obs_feat` 提供"当前场景的自然 delta 尺度"；chunk 提供"实际运动幅度"。两者结合决定建议的执行速度。

## 训练数据构造原理

设原始 demo 以 20 Hz 采集，controller 在 60 Hz 上运行。对每条 demo 在每个起点 `t` 可以构造三档 chunk：


| v_mode | source 长度 | chunk 内容                                  | controller 行为     |
| ------ | --------- | ----------------------------------------- | ----------------- |
| **3**  | 20 帧      | `demo[t : t+20]` 原样                       | factor=3 插值到 60Hz |
| **2**  | 14 帧      | `resample(demo[t : t+14], target_len=20)` | factor=2 插值到 60Hz |
| **1**  | 7 帧       | `resample(demo[t : t+ 7], target_len=20)` | 不插值               |


每档都是"按 demo 自然速度执行 demo 内容"，只是 lookahead 不同。v2 数据构造改成 **burst-aware 增广**：非爆发阶段只保留 `v_mode=3` 训练样本；爆发阶段保留 `v_mode=3/2/1` 三档训练样本。这样避免在夹边、对折等慢速阶段强行构造 `v_mode=1/2` 正样本。

burst 检测在 `configs/build_v2.yaml` 的 `burst:` 中配置：

```yaml
burst:
  mode: threshold          # "threshold" | "peak_window"
  speed_threshold: 0.6     # mode=threshold 时，position speed 超过该值视为 burst
  peak_window: 2.0         # mode=peak_window 时，最大 z 速度点前后窗口秒数
  dilate_frames: 5         # burst mask 向前后膨胀的帧数
  hz: 20                   # 速度计算频率
```

- `threshold`：计算左右臂 position velocity，取 `max(left_speed, right_speed)`，超过 `speed_threshold` 的帧作为 burst。
- `peak_window`：参考 `examples/x2robot/mp4_json_edit/vis_x2robot_velocity.py`，找左右臂 `position z` 正向速度最大的帧，取 `[t - peak_window, t + peak_window]` 作为 burst 区间。

`.npz` 中仍保持固定 shape：`chunks: [num_t, 3, 20, 14]`。非 burst 帧的 `v_mode=2/1` slot 会复制 `v_mode=3` chunk，但 `VelocityGuiderDataset` 会根据 `samples.parquet` 中的 `is_burst` 列过滤掉这些样本，不参与训练。

⚠️ resample 算法与 `x2robot-slave/scripts/socket2ros_async.py::interpolates_actions` 一致：pos+gripper 走线性插值，rotation 走 euler→quat→4 维线性→renorm→euler。`data/resample.py` 内置自检，在 `target_len = factor*(N-1)+1` 时与原函数数值匹配（精度 1e-7）。

## 输入约定

- 14 维 master action（从 lerobot `actions[:, 14:28]` 取，列顺序按
`convert_x2robot_data_to_lerobot_v5.py` 的 `ACTION_KEYS`）：
  - 左臂：`master_left_position(3) + master_left_rotation_euler_xyz(3) + master_left_gripper(1)`
  - 右臂：`master_right_position(3) + master_right_rotation_euler_xyz(3) + master_right_gripper(1)`
- 三个相机 `face_view / left_wrist_view / right_wrist_view`（lerobot 存储名），在喂给 vision tower
时对应 openpi 内部约定 `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`。

## 视觉特征

直接复用 pi0 的 PaliGemma vision tower（`paligemma_variant=gemma_2b`，width=2048）：

1. uint8 BHWC → float32 BHWC ∈ [-1, 1]
2. `resize_with_pad_torch` 到 224×224
3. `paligemma_with_expert.embed_image(...)` → `[B, 256_patches, 2048]`
4. 沿 patch 维 mean-pool → `[B, 2048]` per camera
5. concat 三个相机 → `[B, 6144]`

加载逻辑与 `scripts/train_pytorch.py` 一致（`safetensors.torch.load_model`），保证特征和推理路径完全相同。当前环境未安装 openpi 的 `transformers_replace` patch，bf16 会在 SigLIP 的 fp32 LayerNorm 处触发 dtype mismatch，所以默认用 `float32` 提特征；修好 patch 后可把配置里的 `vision_encoder.dtype` 改回 `bfloat16` 提速。

> 当前用 pi0 base ckpt 跑通，之后会换成 fold_towel_sm2sm 的 sft ckpt 以匹配部署时的实际特征。

## 输出格式

写到 `output.root`（`configs/build_v2.yaml` 默认 `/mnt/public/guqiuyi/dataset/velocity_guider_data/v2/`）：

```
v2/
├── train/ep_<repo>_<ep_idx>.npz
├── val/ep_<repo>_<ep_idx>.npz
├── samples.parquet              # 全局样本索引，含 split, source_repo, episode_idx, frame_idx, is_burst
├── action_stats.json            # 14 维 master action 的 mean/std/q01/q99/min/max（只用 train 算）
└── build_config.yaml            # 配置快照（含 train/val episode 列表，可复现）
```

每个 `.npz`：


| key           | shape                | dtype   | 说明                                     |
| ------------- | -------------------- | ------- | -------------------------------------- |
| `obs_feat`    | `[num_t, 3*width]`   | float32 | width=2048（gemma_2b）                   |
| `chunks`      | `[num_t, 3, 20, 14]` | float32 | axis=1 顺序：v_mode=3, 2, 1               |
| `v_modes`     | `[num_t, 3]`         | int8    | 与 `chunks` axis=1 对应，全部行都是 `[3, 2, 1]` |
| `frame_idx`   | `[num_t]`            | int32   | 起点 t 在原 episode 的帧索引                   |
| `episode_idx` | scalar               | int32   | 该 episode 索引                           |
| `burst_mask`  | `[num_t]`            | bool    | 起点帧是否处于 burst 阶段                       |


`samples.parquet` 中每个起点帧额外记录 `is_burst`。训练集展开样本时，非 burst 行只生成 `v_mode=3` 样本，burst 行生成 `v_mode=3/2/1` 三个样本。因此 v2 的有效训练样本数约为 `N + 2 * N_burst`，类别分布会比 v1 更偏向 `v_mode=3`。

## 怎么跑

单卡：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python velocity_guider/build_dataset.py \
    --config velocity_guider/configs/build_v2.yaml
```

4 卡 / 8 卡并行（推荐）：

```bash
cd /home/guqiuyi/workspace/openpi
bash velocity_guider/run_build_shards.sh 4 velocity_guider/configs/build_v2.yaml
# 或
bash velocity_guider/run_build_shards.sh 8 velocity_guider/configs/build_v2.yaml
```

它会启动 N 个独立进程，每个进程只看到一张 GPU，所以配置里的 `vision_encoder.device: cuda:0` 不需要改：

```bash
CUDA_VISIBLE_DEVICES=<shard_id> uv run python velocity_guider/build_dataset.py \
    --config velocity_guider/configs/build_v2.yaml \
    --num-shards <N> --shard-id <shard_id>
```

每个 shard 写自己的 `samples_shard_XXX.parquet` 和 `build_config_shard_XXX.yaml`。全部完成后，启动脚本会自动合并：

```bash
uv run python velocity_guider/build_dataset.py \
    --config velocity_guider/configs/build_v2.yaml \
    --merge-shards
```

合并后生成最终 `samples.parquet` 和 `merge_manifest.yaml`。

debug：

```bash
# 只跑一个 repo 的前 2 个 episode
uv run python velocity_guider/build_dataset.py \
    --config velocity_guider/configs/build_v2.yaml \
    --repo fold_towel_gqy_0317 --episode-limit 2
```

## 训练 Velocity Guider

训练任务采用方案 A：直接做 3 类分类，label 顺序为：


| label | v_mode | 语义         |
| ----- | ------ | ---------- |
| `0`   | `3`    | 最慢 / 最平滑   |
| `1`   | `2`    | 中速         |
| `2`   | `1`    | 最快 / 爆发力最大 |


action chunk 使用 `action_stats.json` 里的 train-set mean/std 做 z-score 归一化。checkpoint 会保存 normalizer，推理时无需重新找统计文件。

对于 v2 数据，`VelocityGuiderDataset` 会根据 `samples.parquet` 中的 `is_burst` 列展开样本：非 burst 帧只生成 `v_mode=3` 样本，burst 帧生成 `v_mode=3/2/1` 三个样本。旧 v1 数据没有 `is_burst` 列，dataset 会兼容为所有帧展开三档。

如果使用 `build_v2.yaml` 生成的新数据，训练前需要把训练配置里的数据路径切到 v2，例如：

```yaml
data:
  dataset_root: /mnt/public/guqiuyi/dataset/velocity_guider_data/v2

output_dir: /mnt/public/guqiuyi/checkpoints/velocity_guider/v2
```

单卡训练：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python velocity_guider/train.py \
    --config velocity_guider/configs/train_v1.yaml
```

4-8 张 A100 DDP 训练：

```bash
cd /home/guqiuyi/workspace/openpi
bash velocity_guider/run_train_ddp.sh 4
# 或
bash velocity_guider/run_train_ddp.sh 8
```

脚本内部使用：

```bash
uv run torchrun --standalone --nproc_per_node=<N> \
    velocity_guider/train.py \
    --config velocity_guider/configs/train_v1.yaml
```

wandb 默认开启，项目名是 `velocity-guider`。如果只想本地 smoke test：

```bash
uv run python velocity_guider/train.py \
    --config velocity_guider/configs/train_v1.yaml \
    --epochs 1 \
    --batch-size 128 \
    --wandb-mode disabled
```

训练会额外记录两个 baseline：

- `majority/*`：永远预测 `label=0 / v_mode=3`
- `motion_baseline/*`：用 chunk 平均运动强度的三分位阈值预测 `0/1/2`

推理封装：

```python
from velocity_guider.infer import VelocityGuiderInfer

guider = VelocityGuiderInfer("/mnt/public/guqiuyi/checkpoints/velocity_guider/v1/best.pt")
out = guider.predict(obs_feat, action_chunk)  # raw action chunk, not normalized
print(out["v_mode"], out["prob"])
```

## 可视化脚本

### 可视化 Velocity Guider 预测

`visualize_lerobot_velocity_guider.py` 会对 LeRobot episode 在线提取 `obs_feat`，用训练好的 `best.pt` 预测每帧的 `v_mode`，并画出：

- 每帧预测的 `v_mode`
- `v_mode=3/2/1` 的概率曲线
- 左右臂 position `x/y/z` 速度曲线

单个 episode：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python velocity_guider/visualize_lerobot_velocity_guider.py \
    --lerobot-root /mnt/public/guqiuyi/huggingface/lerobot \
    --repo fold_towel_gqy_0412 \
    --checkpoint /mnt/public/guqiuyi/checkpoints/velocity_guider/v2/best.pt \
    --pi0-checkpoint-path /mnt/public/models/pytorch_models/pi0_base_pytorch \
    --episode 0 \
    --output-dir velocity_guider/pics/inference_prediction/v2_ckpt
```

批量跑一个 repo 的前 N 个 episode：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python velocity_guider/visualize_lerobot_velocity_guider.py \
    --lerobot-root /mnt/public/guqiuyi/huggingface/lerobot \
    --repo fold_towel_gqy_0412 \
    --checkpoint /mnt/public/guqiuyi/checkpoints/velocity_guider/v2/best.pt \
    --pi0-checkpoint-path /mnt/public/models/pytorch_models/pi0_base_pytorch \
    --all \
    --max-episodes 10 \
    --output-dir velocity_guider/pics/inference_prediction/v2_ckpt
```

常用参数：

- `--device cuda:0`：指定用于 vision encoder 和 guider 推理的 GPU。
- `--image-batch-size 32`：视觉特征提取 batch size。
- `--infer-batch-size 1024`：guider 推理 batch size。
- `--highlight-start <sec> --highlight-end <sec>`：在图中高亮某个时间段。
- `--no-save-npz`：只保存 PNG，不保存预测数组。

### 可视化 v_mode 转换假设

`visualize_vmode_conversion.py` 用来比较 SFT demo 速度和 rollout 速度，并模拟 rollout burst 阶段从 `v_mode=2` 转成 `v_mode=3` 后的速度变化。当前逻辑是：

1. 用 position speed 的绝对阈值（默认 `0.4 m/s`）检测 burst。
2. 在 rollout burst 阶段，每次取未来 20 帧 action chunk。
3. 对 chunk 做 factor=2 插值，再按 3 倍降采样回 20Hz，模拟更快执行后的记录轨迹。
4. 分别画 SFT 速度图、rollout 原始速度图、rollout 转换后速度图，并在图底部统计 peak speed ratio。

单个 SFT episode + 单个 rollout episode：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python velocity_guider/visualize_vmode_conversion.py \
    --lerobot-root /mnt/public/guqiuyi/huggingface/lerobot \
    --sft-repo fold_towel_gqy_0420 \
    --rollout-repo fold_towel_gqy_0412 \
    --sft-episode 0 \
    --rollout-episode 0 \
    --burst-threshold 0.4 \
    --output-dir velocity_guider/pics/vmode_conversion
```

批量跑前 N 个 SFT / rollout episode：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python velocity_guider/visualize_vmode_conversion.py \
    --lerobot-root /mnt/public/guqiuyi/huggingface/lerobot \
    --sft-repo fold_towel_gqy_0420 \
    --rollout-repo fold_towel_gqy_0412 \
    --all-sft \
    --all-rollout \
    --max-episodes 10 \
    --burst-threshold 0.4 \
    --output-dir velocity_guider/pics/vmode_conversion
```

输出文件示例：

```text
velocity_guider/pics/vmode_conversion/
├── sft_fold_towel_gqy_0420_ep_000000.png
└── rollout_fold_towel_gqy_0412_ep_000000.png
```

## 在线推理集成

Velocity Guider 已集成到 `scripts/x2robot_infer_seq.py`（推理端）和 `x2robot-slave/scripts/socket2ros_async_test.py`（机器人端）。

### 原理

1. pi0 policy 做 `policy.infer(obs)` 时，`PI0Pytorch.embed_prefix` 会缓存每个相机的 image embedding 到 `_cached_image_embeds`。
2. 推理脚本在 `policy.infer()` 返回后，mean-pool + concat 这些 cached embedding 得到 `obs_feat [1, 6144]`，无需重新跑 vision tower。
3. 取 latency_step 截断 **之前** 的原始 action chunk 前 20 帧，和 `obs_feat` 一起送入 `VelocityGuiderInfer.predict()`。
4. 根据预测的 `v_mode` 映射为 `actions_factor`（`v_mode=3 → factor=3`，`v_mode=2/1 → factor=2`），随 action 一起发送给机器人端。
5. 机器人端从收到的 JSON 中读取 `actions_factor`，动态调整 `interpolates_actions` 的插值倍率。

### 使用方法

推理端（本地台式机）：

```bash
cd /home/guqiuyi/workspace/openpi
uv run python scripts/x2robot_infer_seq.py \
    --policy-config fold_towel_sm2sm \
    --policy-dir checkpoints/fold_towel_sm2sm/<ckpt_step> \
    --guider-checkpoint /mnt/public/guqiuyi/checkpoints/velocity_guider/v2/best.pt
```

如果不传 `--guider-checkpoint`，Velocity Guider 不加载，`actions_factor` 固定为 3，行为与原来完全一致。

机器人端无需改启动参数。`socket2ros_async_test.py` 会自动从每帧收到的 JSON 中读取 `actions_factor` 字段，如果字段不存在则 fallback 到初始化时的默认值。

### 关键改动文件

| 文件 | 改动 |
| --- | --- |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `embed_prefix` 新增 `_cached_image_embeds` 缓存 |
| `scripts/x2robot_infer_seq.py` | `Args` 新增 `guider_checkpoint`；`_extract_obs_feat_from_policy()` 从缓存提取 obs_feat；主循环中集成 v_mode 预测和 actions_factor 发送 |
| `x2robot-slave/scripts/socket2ros_async_test.py` | `_fetch_and_enqueue` 中从 `cmd.get("actions_factor", self.actions_factor)` 动态读取 factor |

## 当前限制 / 已知问题

1. **标签的强弱**：训练样本里 `chunk delta 大小` 几乎能唯一决定 v_mode，`obs_feat` 主要起到"场景自然 delta 尺度"的提示作用。后续可以加入：
  - 真实 rollout 的失败样本（rollout v_mode=1 在折叠场景导致失败 → 强负样本）
  - 把场景拆成 "burst / non-burst" 类别加入辅助监督
2. **pi0 输出 chunk 和我们构造的 v_mode=1/2 chunk 形态是否相似** — 训练完 guider 后需要用真实 pi0 输出做一次相似度可视化验证。
3. 当前 vision tower 用 pi0 base ckpt，**部署时实际跑 sft 后的 ckpt，特征会有差异**。先跑通流程，正式训练前换成 sft ckpt 路径即可。

## 后续要做的（todo）

- ~~`velocity_guider/data/dataset.py` — PyTorch Dataset~~ ✓
- ~~`velocity_guider/model/guider.py` — VelocityGuider nn.Module~~ ✓
- ~~`train.py` — 训练入口~~ ✓
- ~~`infer.py` — 推理封装~~ ✓
- ~~与 pi0 ROS 端集成~~ ✓
- 用 sft ckpt 的 vision tower 重新建数据集 & 训练
- 真实 rollout 失败样本作为强负样本加入训练
- 训练时在线提特征（类似 openpi），避免提前生成大量 `.npz`

