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
"""

from __future__ import annotations

from .registry import Benchmark, BenchmarkRegistry, get_registry

__all__ = [
    "Benchmark",
    "BenchmarkRegistry",
    "get_registry",
]
