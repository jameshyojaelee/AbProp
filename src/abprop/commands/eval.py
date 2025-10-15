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
from abprop.eval.uncertainty import (
    TemperatureScaler,
    expected_calibration_error,
    mean_variance,
    regression_uncertainty_summary,
    sequence_perplexity_from_logits,
    stack_samples,
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
    parser.add_argument("--uncertainty", action="store_true", help="Enable uncertainty estimation outputs.")
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Number of Monte Carlo dropout samples per batch when uncertainty is enabled.",
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="*",
        type=Path,
        default=None,
        help="Optional additional checkpoints for deep ensemble aggregation.",
    )
    parser.add_argument(
        "--temperature-calibration",
        action="store_true",
        help="Fit a temperature scaler for classification logits when uncertainty is enabled.",
    )
    parser.add_argument(
        "--calibration-max-iter",
        type=int,
        default=50,
        help="Maximum LBFGS iterations for temperature calibration.",
    )
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
    cm_np = cm.cpu().numpy()
    im = ax.imshow(cm_np, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Framework", "CDR"])
    ax.set_yticklabels(["Framework", "CDR"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm_np[i, j])}", ha="center", va="center", color="black")
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


def compute_uncertainty_metrics(
    model: AbPropModel,
    dataloader: DataLoader,
    device: torch.device,
    tasks: Sequence[str],
    liability_keys: Sequence[str],
    *,
    mc_samples: int,
    ensemble_models: Optional[Sequence[AbPropModel]] = None,
    pad_token_id: int = 0,
    temperature_calibration: bool = False,
    calibration_max_iter: int = 50,
) -> Dict[str, Dict]:
    """Run Monte Carlo dropout / ensembles to produce uncertainty-aware reports."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive when uncertainty evaluation is enabled.")

    model.eval()
    additional_models = list(ensemble_models or [])
    for ens_model in additional_models:
        ens_model.to(device)
        ens_model.eval()

    regression_means: List[torch.Tensor] = []
    regression_vars: List[torch.Tensor] = []
    regression_targets: List[torch.Tensor] = []
    perplexity_means: List[torch.Tensor] = []
    perplexity_vars: List[torch.Tensor] = []
    cls_logits: List[torch.Tensor] = []
    cls_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if "reg" in tasks:
                liability_targets = batch.get("liability_ln")
                target_tensor = model._prepare_regression_targets(  # type: ignore[attr-defined]
                    liability_targets,
                    batch_size=input_ids.size(0),
                    device=device,
                )
                if target_tensor is not None:
                    regression_targets.append(target_tensor.detach().cpu())

                    mc_outputs = model.stochastic_forward(
                        input_ids,
                        attention_mask,
                        tasks=("reg",),
                        mc_samples=mc_samples,
                        enable_dropout=True,
                        no_grad=True,
                    )
                    sample_list = [out["regression"].detach().cpu() for out in mc_outputs]

                    for ens_model in additional_models:
                        ens_out = ens_model(
                            input_ids,
                            attention_mask,
                            tasks=("reg",),
                        )
                        sample_list.append(ens_out["regression"].detach().cpu())

                    stats = mean_variance(stack_samples(sample_list))
                    regression_means.append(stats.mean)
                    regression_vars.append(stats.variance)

            if "mlm" in tasks:
                mc_outputs = model.stochastic_forward(
                    input_ids,
                    attention_mask,
                    tasks=("mlm",),
                    mc_samples=mc_samples,
                    enable_dropout=True,
                    no_grad=True,
                )
                sample_perplexities = [
                    sequence_perplexity_from_logits(out["mlm_logits"].detach(), input_ids, pad_token_id=pad_token_id).cpu()
                    for out in mc_outputs
                ]
                for ens_model in additional_models:
                    ens_out = ens_model(
                        input_ids,
                        attention_mask,
                        tasks=("mlm",),
                    )
                    sample_perplexities.append(
                        sequence_perplexity_from_logits(
                            ens_out["mlm_logits"].detach(),
                            input_ids,
                            pad_token_id=pad_token_id,
                        ).cpu()
                    )
                stats = mean_variance(stack_samples(sample_perplexities))
                perplexity_means.append(stats.mean)
                perplexity_vars.append(stats.variance)

            if temperature_calibration and "cls" in tasks:
                token_labels = batch.get("token_labels")
                if token_labels is not None:
                    if isinstance(token_labels, torch.Tensor):
                        token_labels = token_labels.to(device)
                    else:
                        token_labels = token_labels
                    outputs = model(
                        input_ids,
                        attention_mask,
                        token_labels=token_labels,  # type: ignore[arg-type]
                        tasks=("cls",),
                    )
                    cls_logits.append(
                        outputs["cls_logits"].detach().reshape(-1, outputs["cls_logits"].size(-1)).cpu()
                    )
                    cls_labels.append(model._prepare_token_labels(  # type: ignore[attr-defined]
                        token_labels,
                        attention_mask,
                        device=device,
                    ).reshape(-1).detach().cpu())

    report: Dict[str, Dict] = {}

    if regression_means:
        mean_tensor = torch.cat(regression_means, dim=0)
        var_tensor = torch.cat(regression_vars, dim=0)
        target_tensor = torch.cat(regression_targets, dim=0)
        regression_report = regression_uncertainty_summary(mean_tensor, var_tensor, target_tensor)
        per_key: Dict[str, Dict] = {}
        for idx, key in enumerate(liability_keys):
            per_key[key] = regression_uncertainty_summary(
                mean_tensor[:, idx],
                var_tensor[:, idx],
                target_tensor[:, idx],
            )
        regression_report["per_key"] = per_key
        report["regression"] = regression_report

    if perplexity_means:
        mean_tensor = torch.cat(perplexity_means, dim=0)
        var_tensor = torch.cat(perplexity_vars, dim=0)
        report["perplexity"] = {
            "mean_perplexity": float(mean_tensor.mean().item()),
            "variance_mean": float(var_tensor.mean().item()),
            "variance_median": float(var_tensor.flatten().median().item()),
        }

    if temperature_calibration and cls_logits:
        logits_tensor = torch.cat(cls_logits, dim=0)
        labels_tensor = torch.cat(cls_labels, dim=0)
        scaler = TemperatureScaler()
        temperature = scaler.fit(
            logits_tensor,
            labels_tensor,
            ignore_index=-100,
            max_iter=calibration_max_iter,
        )
        with torch.no_grad():
            probs_before = torch.softmax(logits_tensor, dim=-1)
            probs_after = torch.softmax(scaler(logits_tensor), dim=-1)
        ece_before = expected_calibration_error(probs_before, labels_tensor)
        ece_after = expected_calibration_error(probs_after, labels_tensor)
        report["classification"] = {
            "temperature": float(temperature),
            "ece_before": ece_before,
            "ece_after": ece_after,
        }

    return report


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

    ensemble_models: List[AbPropModel] = []
    if args.ensemble_checkpoints:
        for ckpt_path in args.ensemble_checkpoints:
            if ckpt_path == args.checkpoint:
                continue
            ens_model, _ = instantiate_model(model_cfg)
            ens_model.to(device)
            load_checkpoint(ens_model, ckpt_path, device)
            ensemble_models.append(ens_model)

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

            if args.uncertainty:
                uncertainty_report = compute_uncertainty_metrics(
                    model,
                    dataloader,
                    device,
                    tasks,
                    model_config.liability_keys,
                    mc_samples=args.mc_samples,
                    ensemble_models=ensemble_models,
                    temperature_calibration=args.temperature_calibration,
                    calibration_max_iter=args.calibration_max_iter,
                )
                if uncertainty_report:
                    report["uncertainty"] = uncertainty_report

            summary[split] = report

            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = split_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)

            mlflow_log_dict(report, f"{split}/metrics.json")

            if args.uncertainty and "uncertainty" in report:
                uncertainty_path = split_dir / "uncertainty.json"
                with open(uncertainty_path, "w", encoding="utf-8") as handle:
                    json.dump(report["uncertainty"], handle, indent=2)
                mlflow_log_dict(report["uncertainty"], f"{split}/uncertainty.json")

            if "classification" in report and "confusion_matrix" in report["classification"]:
                cm_path = split_dir / "confusion_matrix.png"
                save_confusion_matrix(report["classification"]["confusion_matrix"], cm_path)
                mlflow_log_artifact(cm_path, artifact_path=split)

            if "regression" in report:
                preds, targets = collect_regression_tensors(model, dataloader, device, tasks)
                if preds is not None and targets is not None:
                    try:
                        save_regression_scatter(preds, targets, model_config.liability_keys, split_dir)
                    except Exception as err:  # pragma: no cover - logging safeguard
                        print(f"[warn] unable to write regression scatter plots: {err}")
                    for key in model_config.liability_keys:
                        fig_path = split_dir / f"scatter_{key}.png"
                        if fig_path.exists():
                            mlflow_log_artifact(fig_path, artifact_path=split)

        with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        mlflow_log_dict(summary, "summary.json")


__all__ = ["main", "build_parser"]
