"""Evaluation metric helpers for AbProp models."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch


def compute_perplexity(total_loss: float, token_count: int) -> float:
    """Return perplexity given total negative log-likelihood and token count."""
    if token_count <= 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_loss / token_count)))


def classification_summary(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1 for binary classification."""
    tp_f = float(tp)
    fp_f = float(fp)
    tn_f = float(tn)
    fn_f = float(fn)
    total = tp_f + fp_f + tn_f + fn_f
    accuracy = (tp_f + tn_f) / total if total else 0.0
    precision = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 0.0
    recall = tp_f / (tp_f + fn_f) if (tp_f + fn_f) else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _rank(data: torch.Tensor) -> torch.Tensor:
    """Compute ranks (1-indexed) for a 1D tensor. Ties receive average ranks."""
    sorted_vals, sorted_idx = torch.sort(data)
    ranks = torch.zeros_like(data, dtype=torch.float32)
    i = 0
    n = data.numel()
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # convert to 1-indexed
        ranks[sorted_idx[i:j]] = avg_rank
        i = j
    return ranks


def regression_summary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Return regression metrics (MSE, R^2, Spearman) for flattened predictions."""
    preds = predictions.view(-1).float()
    target = targets.view(-1).float()
    n = preds.numel()
    if n == 0:
        return {"mse": 0.0, "r2": 0.0, "spearman": 0.0}

    mse = torch.mean((preds - target) ** 2).item()

    target_mean = torch.mean(target)
    ss_res = torch.sum((preds - target) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    pred_ranks = _rank(preds)
    targ_ranks = _rank(target)
    pred_ranks_mean = pred_ranks.mean()
    targ_ranks_mean = targ_ranks.mean()
    numerator = torch.sum((pred_ranks - pred_ranks_mean) * (targ_ranks - targ_ranks_mean))
    denominator = torch.sqrt(
        torch.sum((pred_ranks - pred_ranks_mean) ** 2) * torch.sum((targ_ranks - targ_ranks_mean) ** 2)
    )
    spearman = numerator / (denominator + 1e-8)

    return {
        "mse": float(mse),
        "r2": float(r2.clamp(min=-1.0, max=1.0)),
        "spearman": float(spearman.clamp(min=-1.0, max=1.0)),
    }


def regression_per_key(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    keys: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Return regression metrics per liability key."""
    report: Dict[str, Dict[str, float]] = {}
    for idx, key in enumerate(keys):
        key_pred = predictions[:, idx]
        key_targ = targets[:, idx]
        report[key] = regression_summary(key_pred, key_targ)
    return report


__all__ = [
    "compute_perplexity",
    "classification_summary",
    "regression_summary",
    "regression_per_key",
]

