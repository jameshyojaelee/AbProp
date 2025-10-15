#!/usr/bin/env python3
"""Train linear probes on top of an ESM-2 backbone for AbProp benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from abprop.baselines import ESM2Baseline, ESM2Config
from abprop.data import OASDataset, build_collate_fn
from abprop.utils import load_yaml_config, mlflow_default_tags, mlflow_log_dict, mlflow_log_metrics, mlflow_log_params, mlflow_run, seed_all
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train linear probes for an ESM-2 baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.yaml"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, help="Override device (default: config or auto)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store checkpoints (default: config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    return parser


def _default_training_config() -> Dict[str, object]:
    return {
        "data_path": "./data/processed/oas",
        "train_split": "train",
        "val_split": "val",
        "batch_size": 16,
        "num_workers": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "tasks": ["mlm", "cls", "reg"],
        "output_dir": "./outputs/esm2_probes",
        "checkpoint_name": "best.pt",
    }


def _merge_training_config(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_training_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _move_tensor(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Sequence) and value and isinstance(value[0], torch.Tensor):
        return [tensor.to(device) for tensor in value]
    return value


def build_dataloader(
    data_path: Path,
    split: str,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    generate_mlm: bool,
) -> DataLoader:
    dataset = OASDataset(data_path, split=split)
    collate = build_collate_fn(generate_mlm=generate_mlm)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )


def train_one_epoch(
    model: ESM2Baseline,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float],
    tasks: Sequence[str],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = {key: _move_tensor(value, device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            mlm_labels=batch.get("labels"),
            token_labels=batch.get("token_labels"),
            liability_targets=batch.get("liability_ln"),
            tasks=tasks,
        )
        loss = outputs.get("loss")
        if loss is None:
            continue
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.probe_parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    return {
        "loss": total_loss / max(1, total_steps),
        "steps": total_steps,
    }


@torch.no_grad()
def evaluate(
    model: ESM2Baseline,
    dataloader: DataLoader,
    device: torch.device,
    tasks: Sequence[str],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    metrics_sum: Dict[str, float] = {}

    for batch in dataloader:
        batch = {key: _move_tensor(value, device) for key, value in batch.items()}
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            mlm_labels=batch.get("labels"),
            token_labels=batch.get("token_labels"),
            liability_targets=batch.get("liability_ln"),
            tasks=tasks,
        )
        loss = outputs.get("loss")
        if loss is not None:
            total_loss += float(loss.item())
            total_steps += 1
        metrics = outputs.get("metrics", {})
        for key, value in metrics.items():
            metrics_sum.setdefault(key, 0.0)
            metrics_sum[key] += float(value.item())

    eval_metrics = {key: value / max(1, total_steps) for key, value in metrics_sum.items()}
    eval_metrics["loss"] = total_loss / max(1, total_steps)
    eval_metrics["steps"] = total_steps
    return eval_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_all(args.seed)

    cfg = load_yaml_config(args.config)
    baseline_cfg = cfg.get("baselines", {}).get("esm2", {})

    model_cfg = baseline_cfg.get("model", {})
    config = ESM2Config(
        model_name=model_cfg.get("model_name", "esm2_t33_650M_UR50D"),
        repr_layer=int(model_cfg.get("repr_layer", 33)),
        probe_dropout=float(model_cfg.get("probe_dropout", 0.1)),
        liability_keys=tuple(model_cfg.get("liability_keys", CANONICAL_LIABILITY_KEYS)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        half_precision=bool(model_cfg.get("half_precision", False)),
        tied_token_head=bool(model_cfg.get("tied_token_head", True)),
        tasks=tuple(model_cfg.get("tasks", ("mlm", "cls", "reg"))),
        sequence_pooling=str(model_cfg.get("sequence_pooling", "mean")),
    )

    train_cfg = _merge_training_config(_default_training_config(), baseline_cfg.get("training", {}))

    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.grad_clip is not None:
        train_cfg["grad_clip"] = args.grad_clip
    if args.output_dir is not None:
        train_cfg["output_dir"] = str(args.output_dir)

    output_dir = Path(train_cfg["output_dir"]).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or baseline_cfg.get("device")
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = ESM2Baseline(config).to(device)

    optimizer = torch.optim.AdamW(
        list(model.probe_parameters()),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    train_loader = build_dataloader(
        Path(train_cfg["data_path"]),
        split=str(train_cfg["train_split"]),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        generate_mlm="mlm" in train_cfg["tasks"],
    )
    val_loader = build_dataloader(
        Path(train_cfg["data_path"]),
        split=str(train_cfg["val_split"]),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        generate_mlm="mlm" in train_cfg["tasks"],
    )

    best_loss = float("inf")
    best_state: Dict[str, object] = {}

    with mlflow_run("esm2-probe-train", tags=mlflow_default_tags()):
        mlflow_log_params(
            {
                "epochs": args.epochs,
                "device": str(device),
                "learning_rate": float(train_cfg["learning_rate"]),
                "weight_decay": float(train_cfg["weight_decay"]),
                "grad_clip": float(train_cfg["grad_clip"]),
                "model": config.model_name,
                "repr_layer": config.repr_layer,
            }
        )

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                grad_clip=float(train_cfg["grad_clip"]) if train_cfg["grad_clip"] else None,
                tasks=train_cfg["tasks"],
            )
            elapsed = time.time() - start

            val_metrics = evaluate(model, val_loader, device, tasks=train_cfg["tasks"])

            mlflow_log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "epoch_time": elapsed,
                },
                step=epoch,
            )

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"time={elapsed:.1f}s"
            )

            current_state = {
                "epoch": epoch,
                "model_config": model.to_config_dict(),
                "probe_state": model.get_probe_state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(current_state, checkpoint_dir / "last.pt")

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_state = current_state
                torch.save(best_state, checkpoint_dir / train_cfg["checkpoint_name"])

        summary_path = output_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump({"best_val_loss": best_loss}, handle, indent=2)
        mlflow_log_dict({"best_val_loss": best_loss}, "training_summary.json")

    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    if best_state:
        print(f"Best checkpoint saved to {checkpoint_dir / train_cfg['checkpoint_name']}")


if __name__ == "__main__":
    main()
