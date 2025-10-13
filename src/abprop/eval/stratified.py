"""Stratified evaluation helpers for difficulty-aware performance analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from abprop.data import build_collate_fn
from abprop.eval.metrics import classification_summary, compute_perplexity, regression_per_key, regression_summary


@dataclass
class StratumMetrics:
    """Container for metrics computed on a single stratum."""

    name: str
    size: int
    metrics: Dict[str, float]
    confusion: Optional[Dict[str, int]] = None
    regression: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class StratifiedEvalConfig:
    """Configuration for running stratified evaluation."""

    strata_root: Path
    batch_size: int = 32
    device: torch.device | None = None
    max_tokens: Optional[int] = None
    num_workers: int = 0
    tasks: Sequence[str] = ("mlm", "cls", "reg")
    cache_datasets: bool = True


@dataclass
class StratifiedEvaluationResult:
    """Aggregated result set for stratified evaluation."""

    baseline: Optional[StratumMetrics]
    dimensions: Dict[str, List[StratumMetrics]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {"dimensions": {}}
        if self.baseline:
            data["baseline"] = {
                "name": self.baseline.name,
                "size": self.baseline.size,
                "metrics": self.baseline.metrics,
            }
        for dimension, metrics in self.dimensions.items():
            data["dimensions"][dimension] = [
                {
                    "name": item.name,
                    "size": item.size,
                    "metrics": item.metrics,
                    "confusion": item.confusion,
                    "regression": item.regression,
                }
                for item in metrics
            ]
        if self.metadata:
            data["metadata"] = self.metadata
        return data


def _read_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Parquet file {path} is empty.")
    return frame.reset_index(drop=True)


def _maybe_parse_json(value: object) -> object:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed
        except json.JSONDecodeError:
            return value
    return value


class _StratumDataset(Dataset):
    """Dataset wrapper for a Parquet frame produced by difficulty stratification."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.has_cdr = "cdr_mask" in self.frame.columns

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.frame.iloc[idx]
        liability_raw = _maybe_parse_json(row["liability_ln"])
        liability_cast: Dict[str, float] = {}
        if isinstance(liability_raw, Mapping):
            for key, value in liability_raw.items():
                try:
                    liability_cast[str(key)] = float(value)
                except (TypeError, ValueError):
                    liability_cast[str(key)] = 0.0

        item: Dict[str, object] = {
            "sequence": str(row["sequence"]),
            "chain": str(row["chain"]),
            "liability_ln": liability_cast,
        }
        if self.has_cdr and pd.notna(row["cdr_mask"]):
            parsed = _maybe_parse_json(row["cdr_mask"])
            if isinstance(parsed, (list, tuple)):
                item["cdr_mask"] = [int(x) for x in parsed]
            else:
                item["cdr_mask"] = None
        return item


def _build_dataloader(
    frame: pd.DataFrame,
    batch_size: int,
    *,
    num_workers: int,
) -> DataLoader:
    dataset = _StratumDataset(frame)
    collate = build_collate_fn(generate_mlm=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)


def _evaluate_batches(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tasks: Sequence[str],
) -> Tuple[Dict[str, float], Optional[Dict[str, int]], Optional[Dict[str, Dict[str, float]]]]:
    mlm_loss_total = 0.0
    mlm_tokens = 0

    tp = fp = tn = fn = 0

    reg_preds: List[torch.Tensor] = []
    reg_targets: List[torch.Tensor] = []
    liability_keys: Sequence[str] | None = None

    model.eval()
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
            if isinstance(liability_targets, list):
                liability_keys = tuple(liability_targets[0].keys()) if liability_targets else liability_keys

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
                tokens = int(mask.sum().item())
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
                if liability_keys is None and isinstance(liability_targets, list) and liability_targets:
                    liability_keys = tuple(liability_targets[0].keys())

    metrics: Dict[str, float] = {}
    if mlm_tokens > 0 and "mlm" in tasks:
        metrics["mlm_perplexity"] = compute_perplexity(mlm_loss_total, mlm_tokens)

    confusion_matrix = None
    if (tp + fp + tn + fn) > 0 and "cls" in tasks:
        summary = classification_summary(tp, fp, tn, fn)
        metrics.update({f"cls_{key}": value for key, value in summary.items()})
        confusion_matrix = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    regression_details = None
    if reg_preds and "reg" in tasks and liability_keys:
        preds_tensor = torch.cat(reg_preds, dim=0)
        targets_tensor = torch.cat(reg_targets, dim=0)
        overall = regression_summary(preds_tensor, targets_tensor)
        per_key = regression_per_key(preds_tensor, targets_tensor, liability_keys)
        metrics.update({f"reg_{k}": v for k, v in overall.items()})
        regression_details = {"overall": overall, "per_key": per_key}

    return metrics, confusion_matrix, regression_details


def discover_strata(root: Path) -> Dict[str, Dict[str, Path]]:
    """Discover stratification buckets under the root directory."""
    mapping: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for dimension_dir in root.iterdir():
        if not dimension_dir.is_dir():
            continue
        dimension = dimension_dir.name
        for parquet_path in dimension_dir.glob("*.parquet"):
            bucket = parquet_path.stem
            mapping[dimension][bucket] = parquet_path
    return mapping


def evaluate_strata(
    model: torch.nn.Module,
    config: StratifiedEvalConfig,
) -> StratifiedEvaluationResult:
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strata = discover_strata(config.strata_root)
    if not strata:
        raise ValueError(f"No stratified Parquet files found in {config.strata_root}.")

    cache: Dict[Path, pd.DataFrame] = {}

    def load_frame(path: Path) -> pd.DataFrame:
        if config.cache_datasets and path in cache:
            return cache[path]
        frame = _read_parquet(path)
        if config.cache_datasets:
            cache[path] = frame
        return frame

    dimensions: Dict[str, List[StratumMetrics]] = {}
    all_frames: List[pd.DataFrame] = []

    for dimension, buckets in strata.items():
        metrics_list: List[StratumMetrics] = []
        for bucket_name, parquet_path in sorted(buckets.items()):
            frame = load_frame(parquet_path)
            dataloader = _build_dataloader(frame, config.batch_size, num_workers=config.num_workers)
            metrics, confusion, regression = _evaluate_batches(
                model,
                dataloader,
                device=device,
                tasks=config.tasks,
            )
            metrics_list.append(
                StratumMetrics(
                    name=f"{dimension}/{bucket_name}",
                    size=len(frame),
                    metrics=metrics,
                    confusion=confusion,
                    regression=regression,
                )
            )
            all_frames.append(frame)
        dimensions[dimension] = metrics_list

    baseline = None
    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["sequence", "chain"])
        dataloader = _build_dataloader(merged, config.batch_size, num_workers=config.num_workers)
        metrics, confusion, regression = _evaluate_batches(
            model,
            dataloader,
            device=device,
            tasks=config.tasks,
        )
        baseline = StratumMetrics(
            name="baseline",
            size=len(merged),
            metrics=metrics,
            confusion=confusion,
            regression=regression,
        )

    metadata = {
        "strata_root": str(config.strata_root),
        "tasks": list(config.tasks),
        "device": str(device),
    }

    return StratifiedEvaluationResult(baseline=baseline, dimensions=dimensions, metadata=metadata)


__all__ = [
    "StratifiedEvalConfig",
    "StratifiedEvaluationResult",
    "StratumMetrics",
    "discover_strata",
    "evaluate_strata",
]
