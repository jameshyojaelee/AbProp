"""Error analysis utilities for AbProp evaluations."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans

from abprop.utils import find_motifs


@dataclass
class SampleDetail:
    """Serializable structure describing a single evaluation example."""

    sequence: str
    chain: str
    species: str | None
    length: int
    mlm_loss: float | None = None
    mlm_tokens: int | None = None
    cls_accuracy: float | None = None
    cls_errors: int | None = None
    liability_pred: List[float] | None = None
    liability_true: List[float] | None = None


def build_sample_details(raw_examples: Iterable[Dict[str, object]]) -> List[SampleDetail]:
    """Convert raw example dictionaries into SampleDetail dataclasses."""

    details: List[SampleDetail] = []
    for entry in raw_examples:
        details.append(
            SampleDetail(
                sequence=entry.get("sequence", ""),
                chain=str(entry.get("chain", "")),
                species=entry.get("species"),
                length=int(entry.get("length", 0)),
                mlm_loss=float(entry.get("mlm_loss")) if entry.get("mlm_loss") is not None else None,
                mlm_tokens=int(entry.get("mlm_tokens")) if entry.get("mlm_tokens") is not None else None,
                cls_accuracy=float(entry.get("cls_accuracy")) if entry.get("cls_accuracy") is not None else None,
                cls_errors=int(entry.get("cls_errors")) if entry.get("cls_errors") is not None else None,
                liability_pred=list(entry.get("liability_pred", [])) if entry.get("liability_pred") is not None else None,
                liability_true=list(entry.get("liability_true", [])) if entry.get("liability_true") is not None else None,
            )
        )
    return details


def token_confusion(pairs: Iterable[Tuple[int, int]]) -> Dict[str, int]:
    """Aggregate token-level confusion counts."""

    counter = Counter()
    for predicted, target in pairs:
        if target not in (0, 1) or predicted not in (0, 1):
            continue
        if predicted == 1 and target == 1:
            counter["tp"] += 1
        elif predicted == 1 and target == 0:
            counter["fp"] += 1
        elif predicted == 0 and target == 0:
            counter["tn"] += 1
        elif predicted == 0 and target == 1:
            counter["fn"] += 1
    return dict(counter)


def liability_residuals(details: Sequence[SampleDetail], keys: Sequence[str]) -> np.ndarray:
    """Return matrix of residuals (prediction - target) per liability key."""

    residuals: List[List[float]] = []
    for item in details:
        if item.liability_pred is None or item.liability_true is None:
            continue
        if len(item.liability_pred) != len(keys):
            continue
        residuals.append([p - t for p, t in zip(item.liability_pred, item.liability_true)])
    if not residuals:
        return np.zeros((0, len(keys)))
    return np.asarray(residuals, dtype=np.float32)


def motif_residual_correlations(details: Sequence[SampleDetail], keys: Sequence[str]) -> Dict[str, float]:
    """Compute correlation between liability residuals and motif counts."""

    correlations: Dict[str, float] = {}
    residual_matrix = liability_residuals(details, keys)
    if residual_matrix.size == 0:
        return correlations

    motif_counts: List[Dict[str, int]] = [find_motifs(item.sequence) for item in details if item.sequence]
    if not motif_counts:
        return correlations

    motif_keys = sorted(motif_counts[0].keys())
    motif_array = np.asarray([[counts.get(key, 0) for key in motif_keys] for counts in motif_counts], dtype=np.float32)

    for idx, key in enumerate(keys):
        res = residual_matrix[:, idx]
        if np.allclose(res.std(), 0.0):
            correlations[key] = 0.0
            continue
        corr = np.corrcoef(res, motif_array.mean(axis=1))[0, 1]
        correlations[key] = float(np.nan_to_num(corr))
    return correlations


def worst_liability_examples(details: Sequence[SampleDetail], keys: Sequence[str], top_k: int = 20) -> List[Tuple[SampleDetail, float]]:
    """Return the worst liability examples ranked by mean absolute residual."""

    scored: List[Tuple[SampleDetail, float]] = []
    for item in details:
        if item.liability_pred is None or item.liability_true is None:
            continue
        residuals = [abs(p - t) for p, t in zip(item.liability_pred, item.liability_true)]
        if not residuals:
            continue
        scored.append((item, float(np.mean(residuals))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def cluster_sequences(details: Sequence[SampleDetail], num_clusters: int = 3) -> Dict[int, List[SampleDetail]]:
    """Cluster sequences using liability residual vectors."""

    residual_matrix = liability_residuals(details, keys=CANONICAL_LIABILITY_KEYS)
    if residual_matrix.shape[0] < num_clusters:
        return {}
    model = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    labels = model.fit_predict(residual_matrix)
    clusters: Dict[int, List[SampleDetail]] = {idx: [] for idx in range(num_clusters)}
    for item, label in zip(details, labels):
        clusters[int(label)].append(item)
    return clusters


def plot_confusion(confusion: Dict[str, int], output_path: Path) -> None:
    """Visualize confusion matrix and save to disk."""

    matrix = np.array(
        [
            [confusion.get("tn", 0), confusion.get("fp", 0)],
            [confusion.get("fn", 0), confusion.get("tp", 0)],
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_scatter(residuals: np.ndarray, keys: Sequence[str], output_path: Path) -> None:
    """Create scatter plots of residuals for each liability key."""

    if residuals.size == 0:
        return
    rows = int(np.ceil(len(keys) / 3))
    cols = min(3, len(keys))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, key in enumerate(keys):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        values = residuals[:, idx]
        ax.scatter(np.arange(len(values)), values, s=12, alpha=0.6)
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{key} residuals")
        ax.set_xlabel("Example")
        ax.set_ylabel("Pred - Target")
    for idx in range(len(keys), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


__all__ = [
    "SampleDetail",
    "build_sample_details",
    "token_confusion",
    "liability_residuals",
    "motif_residual_correlations",
    "worst_liability_examples",
    "cluster_sequences",
    "plot_confusion",
    "plot_residual_scatter",
    "save_json",
]
