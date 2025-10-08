"""Evaluation helpers for AbProp models."""

from __future__ import annotations

from typing import Dict

import torch


def compute_perplexity(log_likelihood: torch.Tensor) -> float:
    """Convert average negative log-likelihood into perplexity."""
    return float(torch.exp(log_likelihood).item())


def classification_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute simple accuracy for classification targets."""
    preds = predictions.argmax(dim=-1)
    correct = (preds == targets).float()
    accuracy = correct.mean().item()
    return {"accuracy": accuracy}


__all__ = ["compute_perplexity", "classification_metrics"]

