"""Evaluation command entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from abprop.data import OASDataset, build_collate_fn
from abprop.eval.metrics import (
    classification_summary,
    compute_perplexity,
    regression_per_key,
    regression_summary,
)
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import (
    load_yaml_config,
    mlflow_default_tags,
    mlflow_log_artifact,
    mlflow_log_dict,
    mlflow_log_metrics,
    mlflow_run,
    seed_all,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AbProp checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--splits", nargs="*", default=["val"], help="Dataset splits to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser


def instantiate_model(model_cfg: Dict) -> Tuple[AbPropModel, TransformerConfig]:
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


def load_checkpoint(model: AbPropModel, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=False)


def build_dataloader(
    parquet_dir: Path,
    split: str,
    batch_size: int,
    max_tokens: Optional[int],
) -> DataLoader:
    dataset = OASDataset(parquet_dir, split=split)
    collate = build_collate_fn(generate_mlm=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def evaluate_split(
    model: AbPropModel,
    dataloader: DataLoader,
    device: torch.device,
    liability_keys: Sequence[str],
    tasks: Sequence[str],
) -> Dict:
    model.eval()
    mlm_loss_total = 0.0
    mlm_tokens = 0
    tp = fp = tn = fn = 0
    reg_preds: List[torch.Tensor] = []
    reg_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch.get("labels")
            if mlm_labels is not None:
                mlm_labels = mlm_labels.to(device)
            token_labels = batch.get("token_labels")
            if token_labels is not None:
                token_labels = token_labels.to(device)
            liability_targets = batch.get("liability_ln")

            outputs = model(
                input_ids,
                attention_mask,
                mlm_labels=mlm_labels,
                token_labels=token_labels,
                liability_targets=liability_targets,
                tasks=tasks,
            )

            if mlm_labels is not None and "mlm" in tasks:
                logits = outputs["mlm_logits"]
                mask = mlm_labels != -100
                tokens = mask.sum().item()
                if tokens > 0:
                    loss = torch.nn.functional.cross_entropy(
                        logits[mask],
                        mlm_labels[mask],
                        reduction="sum",
                    )
                    mlm_loss_total += loss.item()
                    mlm_tokens += tokens

            if token_labels is not None and "cls" in tasks:
                logits = outputs["cls_logits"]
                mask = token_labels != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    tgt = token_labels
                    tp += int(((preds == 1) & (tgt == 1) & mask).sum().item())
                    fp += int(((preds == 1) & (tgt == 0) & mask).sum().item())
                    tn += int(((preds == 0) & (tgt == 0) & mask).sum().item())
                    fn += int(((preds == 0) & (tgt == 1) & mask).sum().item())

            if liability_targets is not None and "reg" in tasks:
                preds = outputs["regression"].detach().cpu()
                target_tensor = model._prepare_regression_targets(  # type: ignore[attr-defined]
                    liability_targets,
                    batch_size=input_ids.size(0),
                    device=device,
                ).detach().cpu()
                reg_preds.append(preds)
                reg_targets.append(target_tensor)

    report: Dict[str, Dict] = {}
    if mlm_tokens > 0 and "mlm" in tasks:
        report["mlm"] = {"perplexity": compute_perplexity(mlm_loss_total, mlm_tokens)}

    if (tp + fp + tn + fn) > 0 and "cls" in tasks:
        cls_report = classification_summary(tp, fp, tn, fn)
        cls_report["confusion_matrix"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        report["classification"] = cls_report

    if reg_preds and "reg" in tasks:
        preds_tensor = torch.cat(reg_preds, dim=0)
        targets_tensor = torch.cat(reg_targets, dim=0)
        overall = regression_summary(preds_tensor, targets_tensor)
        per_key = regression_per_key(preds_tensor, targets_tensor, liability_keys)
        report["regression"] = {"overall": overall, "per_key": per_key}

    return report


def save_confusion_matrix(matrix: Dict[str, int], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = torch.tensor(
        [
            [matrix.get("tn", 0), matrix.get("fp", 0)],
            [matrix.get("fn", 0), matrix.get("tp", 0)],
        ],
        dtype=torch.float32,
    )
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Framework", "CDR"])
    ax.set_yticklabels(["Framework", "CDR"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_regression_scatter(
    preds: torch.Tensor,
    targets: torch.Tensor,
    keys: Sequence[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    for idx, key in enumerate(keys):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(targets_np[:, idx], preds_np[:, idx], alpha=0.5, s=10)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Regression Scatter - {key}")
        lims = [
            min(targets_np[:, idx].min(), preds_np[:, idx].min()),
            max(targets_np[:, idx].max(), preds_np[:, idx].max()),
        ]
        ax.plot(lims, lims, linestyle="--", color="black")
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_{key}.png")
        plt.close(fig)


def collect_regression_tensors(
    model: AbPropModel,
    dataloader: DataLoader,
    device: torch.device,
    tasks: Sequence[str],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if "reg" not in tasks:
        return None, None
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            liability_targets = batch.get("liability_ln")
            if liability_targets is None:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids,
                attention_mask,
                mlm_labels=None,
                token_labels=None,
                liability_targets=liability_targets,
                tasks=("reg",),
            )
            preds.append(outputs["regression"].detach().cpu())
            target_tensor = model._prepare_regression_targets(  # type: ignore[attr-defined]
                liability_targets,
                batch_size=input_ids.size(0),
                device=device,
            ).detach().cpu()
            targets.append(target_tensor)
    if not preds:
        return None, None
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    seed_all(args.seed)

    model_cfg = load_yaml_config(args.model_config)
    data_cfg = load_yaml_config(args.data_config)

    model, model_config = instantiate_model(model_cfg)
    device = torch.device(args.device)
    model.to(device)
    load_checkpoint(model, args.checkpoint, device)

    processed_root = Path(data_cfg.get("processed_dir", "data/processed"))
    parquet_cfg = data_cfg.get("parquet", {})
    parquet_subdir = parquet_cfg.get("output_dir", "oas")
    parquet_dir = processed_root / parquet_subdir
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Processed dataset not found at {parquet_dir}")

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = tuple(
        task for task, weight in model_cfg.get("task_weights", {"mlm": 1.0}).items() if weight > 0.0
    ) or ("mlm",)

    summary: Dict[str, Dict] = {}

    with mlflow_run("abprop-eval", tags=mlflow_default_tags()):
        for split in args.splits:
            dataloader = build_dataloader(parquet_dir, split, args.batch_size, args.max_tokens)
            report = evaluate_split(model, dataloader, device, model_config.liability_keys, tasks)
            summary[split] = report

            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = split_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)

            mlflow_log_dict(report, f"{split}/metrics.json")

            if "classification" in report and "confusion_matrix" in report["classification"]:
                cm_path = split_dir / "confusion_matrix.png"
                save_confusion_matrix(report["classification"]["confusion_matrix"], cm_path)
                mlflow_log_artifact(cm_path, artifact_path=split)

            if "regression" in report:
                preds, targets = collect_regression_tensors(model, dataloader, device, tasks)
                if preds is not None and targets is not None:
                    save_regression_scatter(preds, targets, model_config.liability_keys, split_dir)
                    for key in model_config.liability_keys:
                        fig_path = split_dir / f"scatter_{key}.png"
                        if fig_path.exists():
                            mlflow_log_artifact(fig_path, artifact_path=split)

        with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        mlflow_log_dict(summary, "summary.json")


__all__ = ["main", "build_parser"]
