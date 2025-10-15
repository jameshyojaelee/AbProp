"""Uncertainty estimation utilities for AbProp evaluation and serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor


@dataclass
class SampleStatistics:
    """Container for stochastic predictions."""

    mean: Tensor
    variance: Tensor
    samples: Tensor


def stack_samples(sample_list: Sequence[Tensor]) -> Tensor:
    """Stack a list of tensors along a new sample dimension."""
    if not sample_list:
        raise ValueError("sample_list must contain at least one tensor.")
    return torch.stack(sample_list, dim=0)


def mean_variance(samples: Tensor, dim: int = 0) -> SampleStatistics:
    """
    Compute mean and variance along the sample dimension.

    Args:
        samples: Tensor of shape (num_samples, batch, ...)
        dim: Sample dimension (default: 0)

    Returns:
        SampleStatistics with mean, variance, and original samples.
    """
    mean = samples.mean(dim=dim)
    variance = samples.var(dim=dim, unbiased=False)
    return SampleStatistics(mean=mean, variance=variance, samples=samples)


def combine_ensemble(samples: Sequence[Tensor]) -> SampleStatistics:
    """Aggregate predictions from an ensemble of models."""
    stacked = stack_samples(samples)
    return mean_variance(stacked)


def sequence_perplexity_from_logits(
    logits: Tensor,
    labels: Tensor,
    pad_token_id: int = 0,
) -> Tensor:
    """
    Compute per-sequence perplexity given logits and labels.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab)
        labels: Tensor of shape (batch, seq_len)
        pad_token_id: Padding token index to ignore.

    Returns:
        Tensor of shape (batch,) with per-sequence perplexity values.
    """
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    per_token_loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).view(shifted_labels.size())
    mask = (shifted_labels != pad_token_id).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sequence_loss = (per_token_loss * mask).sum(dim=1) / denom
    return torch.exp(per_sequence_loss)


def regression_uncertainty_summary(
    mean: Tensor,
    variance: Tensor,
    targets: Tensor,
    sigma_levels: Sequence[float] = (1.0, 2.0, 3.0),
) -> Dict[str, object]:
    """
    Summarize regression uncertainty by comparing predictive std with errors.

    Returns:
        Dict with correlation metrics and empirical coverage.
    """
    std = variance.clamp_min(1e-12).sqrt()
    abs_error = (mean - targets).abs()
    flat_std = std.reshape(-1)
    flat_error = abs_error.reshape(-1)
    valid = torch.isfinite(flat_std) & torch.isfinite(flat_error)
    pearson = float("nan")
    spearman = float("nan")
    if valid.sum() > 1:
        pearson = float(_pearson_corr(flat_error[valid], flat_std[valid]))
        spearman = float(_spearman_corr(flat_error[valid], flat_std[valid]))

    coverage: Dict[str, float] = {}
    for sigma in sigma_levels:
        band = sigma * std
        within = ((targets >= mean - band) & (targets <= mean + band)).float()
        coverage[f"{sigma:.1f}σ"] = float(within.mean().item())

    return {
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "mean_std": float(std.mean().item()),
        "mean_abs_error": float(abs_error.mean().item()),
        "coverage": coverage,
    }


def expected_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    n_bins: int = 15,
    ignore_index: int = -100,
) -> float:
    """Compute expected calibration error for classification outputs."""
    confidences, predictions = probabilities.max(dim=-1)
    mask = labels != ignore_index
    if mask.sum() == 0:
        return float("nan")
    confidences = confidences[mask]
    accuracies = (predictions[mask] == labels[mask]).float()
    if confidences.numel() == 0:
        return float("nan")

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)
    for bin_idx in range(n_bins):
        lower = bin_boundaries[bin_idx]
        upper = bin_boundaries[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        if in_bin.any():
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = accuracies[in_bin].mean()
            proportion = in_bin.float().mean()
            ece += proportion * torch.abs(bin_confidence - bin_accuracy)
    return float(ece.item())


class TemperatureScaler(nn.Module):
    """Simple temperature scaling module for calibration."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))

    @property
    def temperature(self) -> Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        ignore_index: int = -100,
        max_iter: int = 50,
    ) -> float:
        """
        Fit the temperature parameter by minimizing cross-entropy on logits/labels.

        Returns:
            Fitted scalar temperature.
        """
        if logits.ndim != 2:
            raise ValueError("Temperature scaling expects 2D logits of shape (N, C).")
        mask = labels != ignore_index
        if mask.sum() == 0:
            return float(self.temperature.item())

        logits = logits[mask]
        labels = labels[mask]
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.log_temperature.data.clamp_(min=-5.0, max=5.0)
        return float(self.temperature.item())


def _pearson_corr(x: Tensor, y: Tensor) -> Tensor:
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = torch.sqrt((x_center**2).sum() * (y_center**2).sum()).clamp_min(1e-12)
    return (x_center * y_center).sum() / denom


def _spearman_corr(x: Tensor, y: Tensor) -> Tensor:
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    return _pearson_corr(x_rank, y_rank)


def _rankdata(values: Tensor) -> Tensor:
    """Compute simple ranks (1-based) without tie correction."""
    sorted_indices = torch.argsort(values)
    ranks = torch.zeros_like(values, dtype=torch.float32)
    ranks[sorted_indices] = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float32)
    return ranks


__all__ = [
    "SampleStatistics",
    "TemperatureScaler",
    "combine_ensemble",
    "expected_calibration_error",
    "mean_variance",
    "regression_uncertainty_summary",
    "sequence_perplexity_from_logits",
    "stack_samples",
]

