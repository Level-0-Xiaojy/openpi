"""DDP-capable trainer for Velocity Guider."""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from velocity_guider.data.normalizer import ActionNormalizer
from velocity_guider.data.dataset import create_velocity_guider_datasets
from velocity_guider.model import VelocityGuider, VelocityGuiderConfig
from velocity_guider.train.baselines import (
    MotionThresholdBaseline,
    evaluate_majority_baseline,
    evaluate_motion_baseline,
)
from velocity_guider.train.metrics import confusion_matrix, metrics_from_confusion

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    dataset_root: str = "/mnt/public/guqiuyi/dataset/velocity_guider_data/v1"
    action_clip: float | None = None
    cache_size: int = 16
    batch_size: int = 1024
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class OptimConfig:
    epochs: int = 50
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bfloat16"


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "velocity-guider"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"


@dataclass
class TrainConfig:
    seed: int = 0
    output_dir: str = "/mnt/public/guqiuyi/checkpoints/velocity_guider/v1"
    eval_interval: int = 1
    save_interval: int = 5
    data: DataConfig = field(default_factory=DataConfig)
    model: VelocityGuiderConfig = field(default_factory=VelocityGuiderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if enabled and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return DistInfo(enabled=enabled, rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype in {"none", "float32", "fp32"}:
        return nullcontext()
    if amp_dtype in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif amp_dtype in {"float16", "fp16"}:
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def collate_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _all_reduce_tensor(tensor: torch.Tensor, dist_info: DistInfo) -> torch.Tensor:
    if dist_info.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _sync_metrics(
    loss_sum: float,
    confidence_sum: float,
    count: int,
    cm: torch.Tensor,
    dist_info: DistInfo,
) -> dict[str, float]:
    device = dist_info.device
    stats = torch.tensor([loss_sum, confidence_sum, float(count)], dtype=torch.float64, device=device)
    _all_reduce_tensor(stats, dist_info)
    cm_device = cm.to(device=device, dtype=torch.float64)
    _all_reduce_tensor(cm_device, dist_info)

    total = max(float(stats[2].item()), 1.0)
    out = metrics_from_confusion(cm_device.cpu().to(torch.long))
    out["loss"] = float(stats[0].item() / total)
    out["confidence"] = float(stats[1].item() / total)
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dist_info: DistInfo,
    cfg: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    cm = torch.zeros((3, 3), dtype=torch.long)
    loss_sum = 0.0
    confidence_sum = 0.0
    count = 0

    iterator = tqdm(loader, desc=f"train epoch {epoch}", leave=False) if dist_info.is_main else loader
    for batch in iterator:
        batch = collate_to_device(batch, dist_info.device)
        optimizer.zero_grad(set_to_none=True)
        with make_amp_context(dist_info.device, cfg.optim.amp_dtype):
            logits = model(batch["obs_feat"], batch["action_chunk"])
            loss = criterion(logits, batch["label"])
        loss.backward()
        if cfg.optim.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
        optimizer.step()

        bs = int(batch["label"].shape[0])
        preds = logits.detach().argmax(dim=-1)
        cm += confusion_matrix(preds, batch["label"], num_classes=3)
        loss_sum += float(loss.detach().item()) * bs
        confidence_sum += float(torch.softmax(logits.detach(), dim=-1).max(dim=-1).values.sum().item())
        count += bs

    return {f"train/{k}": v for k, v in _sync_metrics(loss_sum, confidence_sum, count, cm, dist_info).items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    dist_info: DistInfo,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.eval()
    cm = torch.zeros((3, 3), dtype=torch.long)
    loss_sum = 0.0
    confidence_sum = 0.0
    count = 0

    for batch in loader:
        batch = collate_to_device(batch, dist_info.device)
        with make_amp_context(dist_info.device, cfg.optim.amp_dtype):
            logits = model(batch["obs_feat"], batch["action_chunk"])
            loss = criterion(logits, batch["label"])
        bs = int(batch["label"].shape[0])
        preds = logits.argmax(dim=-1)
        cm += confusion_matrix(preds, batch["label"], num_classes=3)
        loss_sum += float(loss.item()) * bs
        confidence_sum += float(torch.softmax(logits, dim=-1).max(dim=-1).values.sum().item())
        count += bs

    return {f"val/{k}": v for k, v in _sync_metrics(loss_sum, confidence_sum, count, cm, dist_info).items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: ActionNormalizer,
    cfg: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "action_normalizer": normalizer.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        },
        path,
    )


def run_training(cfg: TrainConfig) -> None:
    dist_info = setup_distributed()
    set_seed(cfg.seed, dist_info.rank)

    if dist_info.is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.info("Training config: %s", asdict(cfg))

    train_ds, val_ds = create_velocity_guider_datasets(
        cfg.data.dataset_root,
        action_clip=cfg.data.action_clip,
        cache_size=cfg.data.cache_size,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist_info.enabled else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist_info.enabled else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    model = VelocityGuider(cfg.model).to(dist_info.device)
    if dist_info.enabled:
        model = DistributedDataParallel(model, device_ids=[dist_info.local_rank], output_device=dist_info.local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.optim.epochs,
        eta_min=cfg.optim.min_lr,
    )
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if dist_info.is_main and cfg.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            config=asdict(cfg),
        )

    if dist_info.is_main:
        # Baselines use the same validation set. Fit the motion thresholds on train.
        baseline_train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
        )
        baseline_val_loader = DataLoader(
            val_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
        )
        baseline_metrics: dict[str, float] = {}
        majority = evaluate_majority_baseline(baseline_val_loader, majority_class=0)
        motion = MotionThresholdBaseline.fit_from_loader(baseline_train_loader)
        motion_metrics = evaluate_motion_baseline(motion, baseline_val_loader)
        baseline_metrics.update(majority)
        baseline_metrics.update(motion_metrics)
        logger.info("Baselines: %s", baseline_metrics)
        if wandb_run is not None:
            wandb_run.log(baseline_metrics, step=0)
    if dist_info.enabled:
        dist.barrier()

    best_acc = -1.0
    output_dir = Path(cfg.output_dir)
    for epoch in range(1, cfg.optim.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, dist_info, cfg, epoch)
        scheduler.step()
        metrics = dict(train_metrics)
        metrics["train/lr"] = float(scheduler.get_last_lr()[0])

        if epoch % cfg.eval_interval == 0 or epoch == cfg.optim.epochs:
            metrics.update(evaluate(model, val_loader, criterion, dist_info, cfg))

        if dist_info.is_main:
            logger.info("Epoch %d metrics: %s", epoch, metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch)

            val_acc = metrics.get("val/acc", -1.0)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, train_ds.normalizer, cfg, epoch, metrics)
            if epoch % cfg.save_interval == 0 or epoch == cfg.optim.epochs:
                save_checkpoint(
                    output_dir / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    train_ds.normalizer,
                    cfg,
                    epoch,
                    metrics,
                )

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()
