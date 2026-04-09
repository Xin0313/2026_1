"""
Complete training script for SCHPose.

Supports:
  - Single-GPU and DDP multi-GPU training
  - Two-phase training (Phase 1: no pose loss; Phase 2: end-to-end)
  - AdamW + CosineAnnealingLR + linear warmup
  - TensorBoard logging
  - Automatic checkpoint save/load

Usage:
  Single GPU:
    python tools/train.py --config configs/ycbv.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=4 tools/train.py --config configs/ycbv.yaml
"""

import os
import sys
import argparse
import yaml
import json
import time
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.bop_dataset import BOPDataset
from datasets.augmentation import SCHPoseAugmentation
from datasets.surface_codebook import SurfaceCodebook
from models.schpose import SCHPose
from losses.total_loss import TotalLoss
from utils.symmetry import SymmetryHandler, get_symmetry_transforms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("schpose.train")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_ddp():
    """Initialize DDP if RANK environment variable is set."""
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return False, 0, 1
    import torch.distributed as dist
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def cleanup_ddp():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_codebooks(cfg):
    """Load surface codebooks for all objects."""
    codebook_dir = cfg["dataset"].get("codebook_dir", None)
    if not codebook_dir:
        return {}
    object_ids = cfg["dataset"]["object_ids"]
    codebooks = {}
    for obj_id in object_ids:
        path = os.path.join(codebook_dir, f"obj_{obj_id:06d}_codebook.pkl")
        if os.path.isfile(path):
            codebooks[obj_id] = SurfaceCodebook.load(path)
    logger.info(f"Loaded {len(codebooks)}/{len(object_ids)} codebooks")
    return codebooks


def load_sym_transforms(cfg):
    """Load symmetry transforms for all objects."""
    dataset_name = cfg["dataset"]["name"]
    object_ids = cfg["dataset"]["object_ids"]
    sym_transforms = {}
    for obj_id in object_ids:
        transforms = get_symmetry_transforms(dataset_name, obj_id)
        if transforms:
            sym_transforms[obj_id] = transforms
    return sym_transforms


def build_dataset(cfg, split, augmentation=None):
    """Build dataset with optional augmentation."""
    return BOPDataset(cfg, split=split, augmentation=augmentation)


def collate_fn(batch):
    """Custom collate to handle variable-size data."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        elif isinstance(vals[0], bool):
            result[k] = vals
        else:
            result[k] = vals
    return result


class WarmupCosineScheduler:
    """Linear warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device,
                    epoch, phase, cfg, writer, global_step, rank=0):
    """Train for one epoch."""
    model.train()
    train_cfg = cfg["training"]
    log_freq = train_cfg.get("log_freq", 50)

    total_loss_sum = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        K = batch["K"].to(device)
        masks = batch["mask"].to(device)
        obj_ids = batch["obj_id"]
        R_gt = batch["R"].to(device)
        t_gt = batch["t"].to(device)

        # Build targets
        targets = _build_targets(batch, device, cfg)

        # Forward
        with torch.cuda.amp.autocast(enabled=cfg["training"].get("mixed_precision", True)):
            predictions = model(
                images, K,
                obj_ids=obj_ids,
                masks_gt=masks,
                phase=phase,
            )

            total_loss, loss_dict = criterion(predictions, targets)

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("grad_clip", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("grad_clip", 1.0)
            )
            optimizer.step()

        total_loss_sum += total_loss.item()
        n_batches += 1
        global_step += 1

        if rank == 0 and (batch_idx + 1) % log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={total_loss.item():.4f} lr={lr:.6f}"
            )
            if writer:
                writer.add_scalar("train/loss_total", total_loss.item(), global_step)
                for k, v in loss_dict.items():
                    if k != "total":
                        writer.add_scalar(f"train/loss_{k}", float(v), global_step)
                writer.add_scalar("train/lr", lr, global_step)

    avg_loss = total_loss_sum / max(1, n_batches)
    return avg_loss, global_step


def _build_targets(batch, device, cfg):
    """
    Build training targets dict from batch.
    Note: code/uv targets would normally come from codebook precomputation.
    Here we provide placeholder zeros; in full training, precompute these.
    """
    B = batch["image"].shape[0]
    H, W = cfg["dataset"]["image_size"]
    n_coarse = cfg["model"]["n_coarse"]
    n_fine = cfg["model"]["n_fine"]

    # Segmentation targets: obj_label + 1 for foreground pixels, 0 for background
    mask = batch["mask"].to(device)
    obj_labels = batch["obj_label"]
    seg_targets = torch.zeros(B, H, W, dtype=torch.long, device=device)
    for b in range(B):
        label = obj_labels[b] if isinstance(obj_labels, (list, tuple)) else int(obj_labels[b])
        seg_targets[b][mask[b] > 0] = label + 1

    return {
        "seg_targets": seg_targets,
        "coarse_targets": torch.zeros(B, H, W, dtype=torch.long, device=device),
        "fine_targets": torch.zeros(B, H, W, dtype=torch.long, device=device),
        "uv_targets": torch.full((B, 2, H, W), 0.5, device=device),
        "mask": mask,
        "R_gt": batch["R"].to(device),
        "t_gt": batch["t"].to(device),
        "is_symmetric": batch.get("is_symmetric", [False] * B),
    }


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device):
    """Load checkpoint and return start epoch."""
    if not os.path.isfile(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", float("inf"))
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch, best_metric


def main():
    parser = argparse.ArgumentParser(description="Train SCHPose")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--phase", type=int, default=None, help="Force training phase (1 or 2)")
    args = parser.parse_args()

    # Setup DDP
    is_ddp, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = local_rank if is_ddp else 0

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        logger.info(f"Config: {args.config}")
        logger.info(f"Output dir: {output_dir}")
        writer = SummaryWriter(str(output_dir / "tensorboard"))
    else:
        writer = None

    # Load codebooks and symmetry transforms
    codebooks = load_codebooks(cfg)
    sym_transforms = load_sym_transforms(cfg)

    # Build model
    model = SCHPose(cfg, codebooks=codebooks, sym_transforms=sym_transforms)
    model = model.to(device)

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=train_cfg["total_epochs"],
    )

    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if train_cfg.get("mixed_precision", True) and torch.cuda.is_available() else None

    # Load checkpoint
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, args.resume, device)

    # Build datasets and dataloaders
    augmentation = SCHPoseAugmentation(cfg.get("augmentation", {}))
    train_dataset = build_dataset(cfg, split="train", augmentation=augmentation)

    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"] // max(1, world_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    phase1_epochs = train_cfg.get("phase1_epochs", 80)
    total_epochs = train_cfg["total_epochs"]
    global_step = 0

    for epoch in range(start_epoch, total_epochs):
        # Determine training phase
        if args.phase is not None:
            phase = args.phase
        else:
            phase = 1 if epoch < phase1_epochs else 2

        # Build criterion for current phase
        criterion = TotalLoss(cfg, phase=phase)

        # Phase 1: freeze PnP, train encoding heads only
        if phase == 1:
            model_ref = model.module if is_ddp else model
            for p in model_ref.pnp.parameters():
                p.requires_grad_(False)
        else:
            model_ref = model.module if is_ddp else model
            for p in model_ref.pnp.parameters():
                p.requires_grad_(True)

        scheduler.step(epoch)

        if is_ddp and train_sampler:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"=== Epoch {epoch+1}/{total_epochs} (Phase {phase}) ===")

        avg_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch + 1, phase, cfg, writer, global_step, rank=rank,
        )

        if rank == 0:
            logger.info(f"Epoch {epoch+1} avg_loss={avg_loss:.4f}")
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch)

            # Save checkpoint
            model_state = (model.module if is_ddp else model).state_dict()
            ckpt = {
                "epoch": epoch + 1,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "best_metric": best_loss,
                "cfg": cfg,
            }

            # Save periodic checkpoint
            if (epoch + 1) % train_cfg.get("save_freq", 10) == 0:
                ckpt_path = output_dir / f"checkpoint_epoch{epoch+1:04d}.pth"
                save_checkpoint(ckpt, str(ckpt_path))

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(ckpt, str(output_dir / "best_checkpoint.pth"))
                logger.info(f"  Saved best checkpoint (loss={best_loss:.4f})")

            # Always save latest
            save_checkpoint(ckpt, str(output_dir / "latest_checkpoint.pth"))

    if writer:
        writer.close()
    cleanup_ddp()
    if rank == 0:
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
