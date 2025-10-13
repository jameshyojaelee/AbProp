#!/usr/bin/env python
"""Cross-validation training script for robust performance estimation.

This script trains k models on k folds of the data, ensuring clonotype-aware
splitting and producing aggregated metrics with standard deviations.

Example usage:
    # Train 5-fold CV with default settings
    python scripts/train_cv.py

    # Train 10-fold CV with custom output directory
    python scripts/train_cv.py --n-folds 10 --output-dir outputs/cv_10fold

    # Quick test with synthetic data
    python scripts/train_cv.py --synthetic --dry-run-steps 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from abprop.commands.train import instantiate_model
from abprop.data import BucketBatchSampler, build_collate_fn
from abprop.data.cross_validation import GroupKFoldDataset
from abprop.models import TransformerConfig
from abprop.train import LoopConfig, TrainLoop
from abprop.utils import (
    DEFAULT_OUTPUT_DIR,
    cleanup,
    init_distributed,
    load_yaml_config,
    seed_all,
    wrap_ddp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AbProp model with k-fold cross-validation for robust performance estimation."
    )
    parser.add_argument("--config-path", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "cv")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--start-fold", type=int, default=0, help="Starting fold index (for resuming)")
    parser.add_argument("--end-fold", type=int, default=None, help="Ending fold index (exclusive)")
    parser.add_argument("--dry-run-steps", type=int, default=0, help="Quick test with limited steps")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--distributed", choices=["none", "ddp"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_cv_dataloaders(
    data_cfg: dict,
    fold_idx: int,
    n_folds: int,
    batch_size: int,
    *,
    max_tokens: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders for a specific CV fold."""
    processed_root = Path(data_cfg.get("processed_dir", "data/processed"))
    parquet_cfg = data_cfg.get("parquet", {})
    parquet_subdir = parquet_cfg.get("output_dir", "oas")
    parquet_dir = processed_root / parquet_subdir

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Processed dataset not found at {parquet_dir}")

    # Create fold-specific datasets
    train_dataset = GroupKFoldDataset(
        parquet_dir=parquet_dir,
        fold_idx=fold_idx,
        n_splits=n_folds,
        split_type="train",
        seed=seed,
    )

    val_dataset = GroupKFoldDataset(
        parquet_dir=parquet_dir,
        fold_idx=fold_idx,
        n_splits=n_folds,
        split_type="val",
        seed=seed,
    )

    collate = build_collate_fn(generate_mlm=True, mlm_probability=0.15)

    # Create dataloaders with bucketing
    bins = data_cfg.get("length_bins", [64, 128, 256, 512, 1024])

    train_sampler = BucketBatchSampler(
        train_dataset.lengths,
        batch_size=batch_size,
        bins=bins,
        max_tokens=max_tokens,
        shuffle=True,
        seed=seed + fold_idx,  # Different shuffle per fold
    )

    val_sampler = BucketBatchSampler(
        val_dataset.lengths,
        batch_size=batch_size,
        bins=bins,
        max_tokens=max_tokens,
        shuffle=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_single_fold(
    fold_idx: int,
    n_folds: int,
    args: argparse.Namespace,
    train_cfg: dict,
    model_cfg: dict,
    data_cfg: dict,
    dist_info: dict,
) -> Dict[str, float]:
    """Train a model on a single CV fold and return validation metrics."""
    rank = dist_info["rank"]
    device = dist_info["device"]

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Training Fold {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 80}\n")

    # Set fold-specific seed
    fold_seed = args.seed + fold_idx
    seed_all(fold_seed + rank)

    # Create model
    model, model_config = instantiate_model(model_cfg)
    model.to(device)

    # Wrap with DDP if distributed
    if dist_info["is_distributed"]:
        model = wrap_ddp(model, dist_info["local_rank"])

    # Build dataloaders
    batch_size = train_cfg.get("batch_size", 8)
    max_tokens = train_cfg.get("max_tokens", None)
    num_workers = train_cfg.get("num_workers", 0)

    if args.synthetic or args.dry_run_steps > 0:
        # Use synthetic data for testing
        from abprop.commands.train import build_synthetic_dataloaders
        train_loader, val_loader = build_synthetic_dataloaders(
            model_config,
            batch_size,
            num_samples=256,
            distributed=dist_info["is_distributed"],
            rank=rank,
            world_size=dist_info["world_size"],
        )
        max_steps = args.dry_run_steps if args.dry_run_steps > 0 else 100
    else:
        train_loader, val_loader = build_cv_dataloaders(
            data_cfg,
            fold_idx=fold_idx,
            n_folds=n_folds,
            batch_size=batch_size,
            max_tokens=max_tokens,
            num_workers=num_workers,
            seed=fold_seed,
        )
        max_steps = train_cfg.get("max_steps", 1000)

    # Configure training loop
    fold_output_dir = args.output_dir / f"fold_{fold_idx}"
    tasks = tuple(
        task for task, weight in model_cfg.get("task_weights", {"mlm": 1.0}).items() if weight > 0.0
    ) or ("mlm",)

    loop_config = LoopConfig(
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-2),
        gradient_clipping=train_cfg.get("gradient_clipping", 1.0),
        grad_accumulation=train_cfg.get("grad_accumulation", 1),
        max_steps=max_steps,
        warmup_steps=train_cfg.get("warmup_steps", 0),
        lr_schedule=train_cfg.get("lr_schedule", "cosine"),
        eval_interval=train_cfg.get("eval_interval", 200),
        checkpoint_interval=train_cfg.get("checkpoint_interval", 200),
        output_dir=fold_output_dir,
        log_dir=fold_output_dir / "logs",
        checkpoint_dir=fold_output_dir / "checkpoints",
        best_metric=train_cfg.get("best_metric", "eval_loss"),
        maximize_metric=train_cfg.get("maximize_metric", False),
        precision=train_cfg.get("precision", "amp"),
        tasks=tasks,
        report_interval=train_cfg.get("report_interval", 1),
    )

    # Train
    train_loop = TrainLoop(
        model,
        loop_config,
        log_run_name=f"abprop-cv-fold-{fold_idx}",
        device=device,
        is_rank_zero_run=(rank == 0),
    )

    # Save fold metadata
    fold_metadata = {
        "fold_idx": fold_idx,
        "n_folds": n_folds,
        "seed": fold_seed,
        "train_cfg": train_cfg,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
    }
    train_loop.save_run_metadata(fold_metadata)

    # Run training
    train_loop.fit(train_loader, val_loader)

    # Extract final validation metrics
    metrics: Dict[str, float] = {}
    if train_loop.eval_history:
        last_eval = train_loop.eval_history[-1]
        metrics = {k: v for k, v in last_eval.items() if k.startswith("eval_")}

    # Get best metrics from checkpoint
    best_ckpt_path = fold_output_dir / "checkpoints" / "best.pt"
    if best_ckpt_path.exists() and rank == 0:
        checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        if "eval_metrics" in checkpoint:
            metrics["best_" + k] = v for k, v in checkpoint["eval_metrics"].items()

    return metrics


def aggregate_cv_results(
    fold_metrics: List[Dict[str, float]],
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across folds and compute mean ± std dev."""
    if not fold_metrics:
        return {}

    # Collect all metric names
    all_keys = set()
    for metrics in fold_metrics:
        all_keys.update(metrics.keys())

    # Compute statistics for each metric
    aggregated: Dict[str, Dict[str, float]] = {}

    for key in sorted(all_keys):
        values = [m[key] for m in fold_metrics if key in m]
        if not values:
            continue

        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    # Save aggregated results
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "n_folds": len(fold_metrics),
                "aggregated_metrics": aggregated,
                "fold_metrics": fold_metrics,
            },
            f,
            indent=2,
        )

    # Create summary table
    summary_data = []
    for key, stats in aggregated.items():
        summary_data.append({
            "metric": key,
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return aggregated


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load configs
    train_cfg = load_yaml_config(args.config_path)
    model_cfg = load_yaml_config(args.model_config)
    data_cfg = load_yaml_config(args.data_config)

    # Initialize distributed training
    dist_info = init_distributed(args.distributed)
    rank = dist_info["rank"]

    if rank == 0:
        print("=" * 80)
        print("Cross-Validation Training")
        print("=" * 80)
        print(f"Number of folds: {args.n_folds}")
        print(f"Output directory: {args.output_dir}")
        print(f"Random seed: {args.seed}")
        print("=" * 80)
        print()

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine fold range
    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold is not None else args.n_folds

    # Train each fold
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx in range(start_fold, end_fold):
        metrics = train_single_fold(
            fold_idx=fold_idx,
            n_folds=args.n_folds,
            args=args,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            dist_info=dist_info,
        )

        if rank == 0:
            fold_metrics.append(metrics)
            print(f"\nFold {fold_idx} validation metrics:")
            for key, value in sorted(metrics.items()):
                print(f"  {key}: {value:.4f}")

    # Aggregate results
    if rank == 0 and fold_metrics:
        print(f"\n{'=' * 80}")
        print("Cross-Validation Results Summary")
        print(f"{'=' * 80}\n")

        aggregated = aggregate_cv_results(fold_metrics, args.output_dir)

        for key, stats in sorted(aggregated.items()):
            mean = stats["mean"]
            std = stats["std"]
            print(f"{key}: {mean:.4f} ± {std:.4f}")

        print(f"\nResults saved to: {args.output_dir}")
        print(f"  - cv_results.json: Full results with all fold metrics")
        print(f"  - cv_summary.csv: Summary table with mean ± std")

    # Cleanup distributed
    if dist_info["is_distributed"]:
        cleanup()


if __name__ == "__main__":
    main()
