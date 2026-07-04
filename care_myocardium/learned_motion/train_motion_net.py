from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .data import MotionPairDataset
from .model import MotionUNet
from .spatial import smoothness_loss, warp_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learned ED-referenced Motion-Net for CARE cine MRI.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--smooth-weight", type=float, default=0.05)
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--ed-frame-index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--max-flow", type=float, default=12.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cache-in-memory", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def setup_distributed() -> tuple[bool, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices_by_case(dataset: MotionPairDataset, val_fraction: float) -> tuple[list[int], list[int]]:
    case_ids = sorted({record.case_id for record in dataset.records})
    n_val = max(1, int(round(len(case_ids) * val_fraction))) if val_fraction > 0 and len(case_ids) > 1 else 0
    val_cases = set(case_ids[-n_val:]) if n_val else set()
    train_idx, val_idx = [], []
    for idx, record in enumerate(dataset.records):
        if record.case_id in val_cases:
            val_idx.append(idx)
        else:
            train_idx.append(idx)
    return train_idx, val_idx


def make_loader(dataset, batch_size: int, num_workers: int, distributed: bool, shuffle: bool):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    return loader, sampler


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    smooth_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mse": 0.0, "smooth": 0.0, "count": 0.0}
    for batch in loader:
        fixed = batch["fixed"].to(device=device, dtype=torch.float32, non_blocking=True)
        moving = batch["moving"].to(device=device, dtype=torch.float32, non_blocking=True)
        x = torch.cat([fixed, moving], dim=1)
        with torch.set_grad_enabled(training):
            flow = model(x)
            warped = warp_2d(moving, flow)
            mse = F.mse_loss(warped, fixed)
            smooth = smoothness_loss(flow)
            loss = mse + smooth_weight * smooth
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                optimizer.step()
        bs = fixed.shape[0]
        totals["loss"] += float(loss.detach()) * bs
        totals["mse"] += float(mse.detach()) * bs
        totals["smooth"] += float(smooth.detach()) * bs
        totals["count"] += bs
    count = max(totals.pop("count"), 1.0)
    return {k: v / count for k, v in totals.items()}


def reduce_metrics(metrics: dict[str, float], device: torch.device, distributed: bool) -> dict[str, float]:
    if not distributed:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    return {k: float(v) for k, v in zip(keys, values)}


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: dict, best_val: float) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "best_val": best_val,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    distributed, rank, world_size, device = setup_distributed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "motion_net_training_log.txt"
    config = vars(args).copy()
    config.update({"world_size": world_size})

    dataset = MotionPairDataset(args.data_root, args.num_frames, args.ed_frame_index, args.image_size, args.cache_in_memory)
    train_idx, val_idx = split_indices_by_case(dataset, args.val_fraction)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None
    train_loader, train_sampler = make_loader(train_set, args.batch_size, args.num_workers, distributed, True)
    val_loader, _ = make_loader(val_set, args.batch_size, args.num_workers, distributed, False) if val_set else (None, None)

    model = MotionUNet(base_channels=args.base_channels, max_flow=args.max_flow).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_val = float("inf")
    latest = args.output_dir / "checkpoint_latest.pth"
    if args.resume and latest.exists():
        payload = torch.load(latest, map_location=device)
        module = model.module if isinstance(model, DistributedDataParallel) else model
        module.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_val = float(payload.get("best_val", best_val))

    if rank == 0:
        with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[start] samples={len(dataset)} train={len(train_set)} val={len(val_set) if val_set else 0} world_size={world_size}\n")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = reduce_metrics(run_epoch(model, train_loader, device, optimizer, args.smooth_weight), device, distributed)
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = reduce_metrics(run_epoch(model, val_loader, device, None, args.smooth_weight), device, distributed)
        else:
            val_metrics = {"loss": train_metrics["loss"], "mse": train_metrics["mse"], "smooth": train_metrics["smooth"]}

        if rank == 0:
            line = (
                f"epoch={epoch} train_loss={train_metrics['loss']:.6f} train_mse={train_metrics['mse']:.6f} "
                f"train_smooth={train_metrics['smooth']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} val_smooth={val_metrics['smooth']:.6f}\n"
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
            print(line, end="", flush=True)
            save_checkpoint(latest, model, optimizer, epoch, config, best_val)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(args.output_dir / "checkpoint_best.pth", model, optimizer, epoch, config, best_val)

    if rank == 0:
        save_checkpoint(args.output_dir / "checkpoint_final.pth", model, optimizer, args.epochs - 1, config, best_val)
        with log_path.open("a", encoding="utf-8") as f:
            f.write("[done]\n")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
