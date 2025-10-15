"""Evaluation helpers for AbProp models."""

from .metrics import classification_summary, compute_perplexity, regression_per_key, regression_summary
from .uncertainty import (
    SampleStatistics,
    TemperatureScaler,
    combine_ensemble,
    expected_calibration_error,
    mean_variance,
    regression_uncertainty_summary,
    sequence_perplexity_from_logits,
    stack_samples,
)
from .stratified import (
    StratifiedEvalConfig,
    StratifiedEvaluationResult,
    StratumMetrics,
    discover_strata,
    evaluate_strata,
)

__all__ = [
    "compute_perplexity",
    "classification_summary",
    "regression_summary",
    "regression_per_key",
    "StratifiedEvalConfig",
    "StratifiedEvaluationResult",
    "StratumMetrics",
    "SampleStatistics",
    "TemperatureScaler",
    "combine_ensemble",
    "expected_calibration_error",
    "mean_variance",
    "regression_uncertainty_summary",
    "sequence_perplexity_from_logits",
    "stack_samples",
    "discover_strata",
    "evaluate_strata",
]
