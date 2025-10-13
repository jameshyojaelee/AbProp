"""Benchmark suite for comprehensive evaluation of AbProp models.

This module provides a modular benchmark infrastructure for evaluating antibody
property prediction models across multiple dimensions including:
- Perplexity on natural sequences
- CDR region classification
- Liability prediction
- Developability assessment
- Zero-shot generalization

Each benchmark implements a standard interface with load_data(), evaluate(),
and report() methods.

Usage:
    from abprop.benchmarks import get_registry
    from abprop.benchmarks.registry import BenchmarkConfig

    # Get the global registry
    registry = get_registry()

    # Create a benchmark
    config = BenchmarkConfig(data_path="data/processed/oas")
    benchmark = registry.create("perplexity", config)

    # Run the benchmark
    result = benchmark.run(model)
"""

from __future__ import annotations

from .registry import Benchmark, BenchmarkConfig, BenchmarkRegistry, BenchmarkResult, get_registry

# Import all benchmark implementations to trigger registration
from . import cdr_classification_benchmark
from . import developability_benchmark
from . import liability_benchmark
from . import perplexity_benchmark
from . import zero_shot_benchmark

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRegistry",
    "get_registry",
    "cdr_classification_benchmark",
    "developability_benchmark",
    "liability_benchmark",
    "perplexity_benchmark",
    "zero_shot_benchmark",
]
