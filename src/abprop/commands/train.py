"""Training command entrypoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from abprop.data import BucketBatchSampler, OASDataset, build_collate_fn
from abprop.models import AbPropModel, TransformerConfig
from abprop.train import LoopConfig, TrainLoop
from abprop.utils import (
    DEFAULT_OUTPUT_DIR,
    FSDPConfig,
    barrier,
    cleanup,
    enable_activation_checkpointing,
    init_distributed,
    load_yaml_config,
    seed_all,
    wrap_ddp,
    wrap_fsdp_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the AbProp baseline Transformer model on antibody sequences."
    )
    parser.add_argument("--config-path", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run-steps", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fsdp", choices=["off", "full_shard"], default="off")
    parser.add_argument("--grad_ckpt", choices=["true", "false"], default="false")
    parser.add_argument("--distributed", choices=["none", "ddp"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def instantiate_model(model_cfg: dict) -> tuple[AbPropModel, TransformerConfig]:
    defaults = TransformerConfig()
    task_weights = model_cfg.get("task_weights", {})
    config = TransformerConfig(
        vocab_size=model_cfg.get("vocab_size", defaults.vocab_size),
        d_model=model_cfg.get("d_model", defaults.d_model),
        nhead=model_cfg.get("nhead", defaults.nhead),
        num_layers=model_cfg.get("num_layers", defaults.num_layers),
        dim_feedforward=model_cfg.get("dim_feedforward", defaults.dim_feedforward),
        dropout=model_cfg.get("dropout", defaults.dropout),
        max_position_embeddings=model_cfg.get("max_position_embeddings", defaults.max_position_embeddings),
        liability_keys=tuple(model_cfg.get("liability_keys", list(defaults.liability_keys))),
        mlm_weight=task_weights.get("mlm", defaults.mlm_weight),
        cls_weight=task_weights.get("cls", defaults.cls_weight),
        reg_weight=task_weights.get("reg", defaults.reg_weight),
    )
    return AbPropModel(config), config


class SyntheticSequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[str], liability_keys: Sequence[str]) -> None:
        self.examples: List[Dict[str, object]] = []
        liability_template = {key: 0.0 for key in liability_keys}
        for seq in sequences:
            self.examples.append({"sequence": seq, "chain": "H", "liability_ln": liability_template.copy()})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.examples[idx]


class SubsetWithLengths(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, list(indices))
        if hasattr(dataset, "lengths"):
            source_lengths = getattr(dataset, "lengths")
            self.lengths = [source_lengths[i] for i in indices]
        else:
            self.lengths = []
            for i in indices:
                item = dataset[i]
                length = item.get("length")
                if length is None and "sequence" in item:
                    length = len(item["sequence"])
                self.lengths.append(int(length) if length is not None else 0)


def build_synthetic_dataloaders(
    model_config: TransformerConfig,
    batch_size: int,
    num_samples: int = 128,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
    mlm_probability: float,
) -> tuple[DataLoader, DataLoader]:
    motif = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [(motif * ((i % 5) + 1))[:64] for i in range(num_samples)]
    dataset = SyntheticSequenceDataset(sequences, model_config.liability_keys)
    collate = build_collate_fn(generate_mlm=True, mlm_probability=mlm_probability)

    if distributed:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        eval_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate)
        eval_loader = DataLoader(dataset, batch_size=batch_size, sampler=eval_sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        eval_loader = train_loader

    return train_loader, eval_loader


def build_oas_dataloaders(
    data_cfg: dict,
    batch_size: int,
    model_config: TransformerConfig,
    *,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool,
    rank: int,
    world_size: int,
    mlm_probability: float,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader, Optional[DataLoader]]:
    processed_root = Path(data_cfg.get("processed_dir", "data/processed"))
    parquet_cfg = data_cfg.get("parquet", {})
    parquet_subdir = parquet_cfg.get("output_dir")
    dataset_root = processed_root / parquet_subdir if parquet_subdir else processed_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Processed dataset directory not found at {dataset_root}. "
            "Update `configs/data.yaml` (parquet.output_dir) to point to a valid export."
        )

    parquet_filename = parquet_cfg.get("filename")
    dataset_source = dataset_root / parquet_filename if parquet_filename else dataset_root
    if not dataset_source.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_source}. "
            "Ensure the parquet filename in `configs/data.yaml` matches the export."
        )

    train_dataset = OASDataset(dataset_source, split="train")
    if 0.0 < train_fraction < 1.0:
        rng = random.Random(seed)
        indices = list(range(len(train_dataset)))
        rng.shuffle(indices)
        subset_size = max(1, int(len(train_dataset) * train_fraction))
        train_dataset = SubsetWithLengths(train_dataset, indices[:subset_size])

    collate = build_collate_fn(generate_mlm=True, mlm_probability=mlm_probability)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        bins = data_cfg.get("length_bins", [64, 128, 256, 512, 1024])
        sampler = BucketBatchSampler(
            train_dataset.lengths,
            batch_size=batch_size,
            bins=bins,
            max_tokens=max_tokens,
            shuffle=shuffle,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    eval_loader: Optional[DataLoader] = None
    val_count = 0
    try:
        val_dataset = OASDataset(dataset_source, split="val")
    except ValueError:
        if rank == 0:
            print(f"No validation split found in dataset at {dataset_source}; continuing without validation loader.")
    else:
        if 0.0 < val_fraction < 1.0:
            rng = random.Random(seed + 1)
            indices = list(range(len(val_dataset)))
            rng.shuffle(indices)
            subset_size = max(1, int(len(val_dataset) * val_fraction))
            val_dataset = SubsetWithLengths(val_dataset, indices[:subset_size])
        if distributed:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            eval_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            bins = data_cfg.get("length_bins", [64, 128, 256, 512, 1024])
            val_sampler = BucketBatchSampler(
                val_dataset.lengths,
                batch_size=batch_size,
                bins=bins,
                max_tokens=max_tokens,
                shuffle=False,
            )
            eval_loader = DataLoader(
                val_dataset,
                batch_sampler=val_sampler,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        val_count = len(val_dataset)

    if rank == 0:
        train_count = len(train_dataset)
        location_hint = dataset_root if dataset_root.exists() else dataset_source
        print(
            f"Using processed dataset at {location_hint} "
            f"(train={train_count}, val={val_count})"
        )

    return train_loader, eval_loader


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_cfg = load_yaml_config(args.config_path)
    model_cfg = load_yaml_config(args.model_config)
    data_cfg = load_yaml_config(args.data_config)

    dist_info = init_distributed(args.distributed)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    device = dist_info["device"]
    grad_ckpt_enabled = args.grad_ckpt.lower() == "true"

    if rank == 0:
        print("Training configuration:", train_cfg)
        print("Model configuration:", model_cfg)
        print("Data configuration:", data_cfg)
        print(
            f"Distributed mode: {args.distributed} | World size: {world_size} | "
            f"Local rank: {local_rank} | Device: {device}"
        )

    if "task_weights" in train_cfg:
        model_cfg.setdefault("task_weights", {})
        model_cfg["task_weights"].update(train_cfg["task_weights"])

    seed = int(train_cfg.get("seed", args.seed))
    seed_all(seed + rank)

    model, model_config = instantiate_model(model_cfg)
    model.to(device)

    use_fsdp = dist_info["is_distributed"] and args.fsdp != "off"
    if use_fsdp:
        fsdp_config = FSDPConfig(
            sharding=args.fsdp,
            cpu_offload=train_cfg.get("fsdp_cpu_offload", False),
            use_mixed_precision=train_cfg.get("fsdp_mixed_precision", True),
            activation_checkpointing=grad_ckpt_enabled,
        )
        try:
            model = wrap_fsdp_model(model, fsdp_config)
        except RuntimeError as exc:
            if rank == 0:
                print(f"FSDP unavailable ({exc}); falling back to DDP.")
            use_fsdp = False

    if grad_ckpt_enabled and not use_fsdp:
        try:
            enable_activation_checkpointing(model)
        except RuntimeError as exc:
            if rank == 0:
                print(f"Activation checkpointing unavailable: {exc}")

    if dist_info["is_distributed"] and not use_fsdp:
        model = wrap_ddp(model, local_rank)

    synthetic_mode = args.synthetic or args.dry_run_steps > 0
    batch_size = train_cfg.get("batch_size", 8)
    if rank == 0:
        print(f"Per-rank batch size: {batch_size}")
    max_tokens = train_cfg.get("max_tokens", None)
    num_workers = train_cfg.get("num_workers", 0)
    mlm_probability = float(train_cfg.get("mlm_probability", 0.15))
    train_fraction = float(train_cfg.get("train_fraction", 1.0))
    val_fraction = float(train_cfg.get("val_fraction", 1.0))

    if synthetic_mode:
        synthetic_samples = int(train_cfg.get("synthetic_samples", 256))
        train_loader, eval_loader = build_synthetic_dataloaders(
            model_config,
            batch_size,
            synthetic_samples,
            distributed=dist_info["is_distributed"],
            rank=rank,
            world_size=world_size,
            mlm_probability=mlm_probability,
        )
        max_steps = args.dry_run_steps if args.dry_run_steps > 0 else train_cfg.get("max_steps", 100)
    else:
        try:
            train_loader, eval_loader = build_oas_dataloaders(
                data_cfg,
                batch_size=batch_size,
                model_config=model_config,
                max_tokens=max_tokens,
                shuffle=True,
                num_workers=num_workers,
                distributed=dist_info["is_distributed"],
                rank=rank,
                world_size=world_size,
                mlm_probability=mlm_probability,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
                seed=seed,
            )
        except FileNotFoundError as exc:
            if rank == 0:
                print(exc)
            raise
        max_steps = train_cfg.get("max_steps", 1000)

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
        output_dir=args.output_dir,
        log_dir=Path(train_cfg.get("log_dir", args.output_dir / "logs")),
        checkpoint_dir=Path(train_cfg.get("checkpoint_dir", args.output_dir / "checkpoints")),
        best_metric=train_cfg.get("best_metric", "eval_loss"),
        maximize_metric=train_cfg.get("maximize_metric", False),
        precision=train_cfg.get("precision", "amp"),
        tasks=tasks,
        report_interval=train_cfg.get("report_interval", 1),
        optimizer=train_cfg.get("optimizer", "adamw"),
        adam_beta1=train_cfg.get("adam_beta1", 0.9),
        adam_beta2=train_cfg.get("adam_beta2", 0.999),
        sgd_momentum=train_cfg.get("sgd_momentum", 0.0),
    )

    mlflow_tags = {}
    mlflow_cfg = train_cfg.get("mlflow", {})
    if isinstance(mlflow_cfg, dict):
        tags = mlflow_cfg.get("tags")
        if isinstance(tags, dict):
            mlflow_tags = {str(k): v for k, v in tags.items()}

    train_loop = TrainLoop(
        model,
        loop_config,
        log_run_name="abprop-train",
        device=device,
        is_rank_zero_run=(rank == 0),
        mlflow_tags=mlflow_tags,
    )
    train_loop.save_run_metadata({"train": train_cfg, "model": model_cfg, "data": data_cfg})
    train_loop.fit(train_loader, eval_loader)

    if synthetic_mode and train_loop.train_history:
        start_loss = train_loop.train_history[0]
        final_loss = train_loop.train_history[-1]
        print(f"Synthetic training loss trend: start={start_loss:.4f}, final={final_loss:.4f}")
        if len(train_loop.train_history) >= 100:
            print(
                f"Loss delta after {len(train_loop.train_history)} steps: "
                f"{start_loss - final_loss:.4f}"
            )

    if dist_info["is_distributed"]:
        barrier()
        cleanup()


__all__ = ["main", "build_parser"]
