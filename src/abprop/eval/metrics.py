"""Evaluation metric helpers for AbProp models."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from scipy import stats


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
    "calibration_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "kl_divergence",
    "wasserstein_1d",
    "kendall_tau",
    "bootstrap_confidence_interval",
]


def calibration_curve(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> Dict[str, torch.Tensor]:
    """Return calibration statistics for binary probabilities.

    Args:
        probabilities: Tensor of predicted probabilities for the positive class.
        targets: Tensor of 0/1 ground truth labels.
        num_bins: Number of bins for calibration.

    Returns:
        Dictionary containing bin accuracies, confidences, counts, ECE, and MCE.
    """

    probs = probabilities.detach().flatten().float()
    labels = targets.detach().flatten().float()
    probs = probs.clamp(0.0, 1.0)
    labels = labels.clamp(0.0, 1.0)

    bins = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(probs, bins, right=True) - 1
    bin_indices = bin_indices.clamp(0, num_bins - 1)

    bin_confidence = torch.zeros(num_bins, dtype=torch.float32, device=probs.device)
    bin_accuracy = torch.zeros_like(bin_confidence)
    bin_counts = torch.zeros_like(bin_confidence)

    for b in range(num_bins):
        mask = bin_indices == b
        if mask.any():
            bin_probs = probs[mask]
            bin_labels = labels[mask]
            bin_confidence[b] = bin_probs.mean()
            bin_accuracy[b] = bin_labels.mean()
            bin_counts[b] = mask.sum().float()

    total = bin_counts.sum().clamp_min(1.0)
    abs_diff = (bin_accuracy - bin_confidence).abs()
    ece = (abs_diff * bin_counts / total).sum()
    mce = abs_diff.max()

    return {
        "bin_accuracy": bin_accuracy.cpu(),
        "bin_confidence": bin_confidence.cpu(),
        "bin_counts": bin_counts.cpu(),
        "ece": float(ece.cpu()),
        "mce": float(mce.cpu()),
        "bin_edges": bins.cpu(),
    }


def expected_calibration_error(probabilities: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> float:
    """Convenience wrapper returning the expected calibration error."""

    return calibration_curve(probabilities, targets, num_bins)["ece"]


def maximum_calibration_error(probabilities: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> float:
    """Return maximum calibration error across bins."""

    return calibration_curve(probabilities, targets, num_bins)["mce"]


def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Compute KL divergence KL(p || q) for discrete distributions."""

    p = p.float().view(-1)
    q = q.float().view(-1)
    p = p / p.sum().clamp_min(epsilon)
    q = q / q.sum().clamp_min(epsilon)
    divergence = (p * (torch.log(p + epsilon) - torch.log(q + epsilon))).sum()
    return float(divergence.cpu())


def wasserstein_1d(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute the 1D Wasserstein distance between two samples."""

    return float(stats.wasserstein_distance(p.view(-1).cpu().numpy(), q.view(-1).cpu().numpy()))


def kendall_tau(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Kendall's Tau rank correlation."""

    preds = predictions.view(-1).detach().cpu().numpy()
    targs = targets.view(-1).detach().cpu().numpy()
    if preds.size == 0:
        return 0.0
    tau, _ = stats.kendalltau(preds, targs)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def bootstrap_confidence_interval(
    values: torch.Tensor,
    statistic: Callable[[torch.Tensor], torch.Tensor | float],
    confidence: float = 0.95,
    num_bootstrap: int = 500,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a statistic over 1-D tensor."""

    rng = torch.Generator(device=values.device)
    rng.manual_seed(seed)
    values = values.flatten()
    n = values.numel()
    if n == 0:
        return (0.0, 0.0)

    stats_samples = []
    for _ in range(num_bootstrap):
        indices = torch.randint(0, n, (n,), generator=rng, device=values.device)
        sample = values[indices]
        stat_val = statistic(sample)
        stats_samples.append(float(stat_val))

    stats_array = torch.tensor(stats_samples)
    lower = float(torch.quantile(stats_array, (1 - confidence) / 2))
    upper = float(torch.quantile(stats_array, 1 - (1 - confidence) / 2))
    return lower, upper
