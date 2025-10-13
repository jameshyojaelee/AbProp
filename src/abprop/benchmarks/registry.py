"""Benchmark registry and base classes for AbProp evaluation suite."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
from torch.utils.data import DataLoader


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation.

    Attributes:
        data_path: Path to the benchmark dataset
        batch_size: Batch size for evaluation
        max_samples: Optional limit on number of samples to evaluate
        device: Device to run evaluation on
        output_dir: Directory to save results
        mlflow_tracking: Whether to log results to MLflow
    """
    data_path: Path
    batch_size: int = 32
    max_samples: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path = Path("outputs/benchmarks")
    mlflow_tracking: bool = True


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results.

    Attributes:
        benchmark_name: Name of the benchmark
        metrics: Dictionary of metric names to values
        plots: Dictionary of plot names to file paths
        metadata: Additional metadata about the evaluation
    """
    benchmark_name: str
    metrics: Dict[str, float]
    plots: Dict[str, Path]
    metadata: Dict[str, Any]


class Benchmark(ABC):
    """Base class for all AbProp benchmarks.

    All benchmark implementations must inherit from this class and implement
    the abstract methods: load_data(), evaluate(), and report().
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark with configuration.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.name = self.__class__.__name__.replace("Benchmark", "").lower()

    @abstractmethod
    def load_data(self) -> DataLoader:
        """Load and prepare the benchmark dataset.

        Returns:
            DataLoader for the benchmark dataset
        """
        pass

    @abstractmethod
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Run evaluation on the model.

        Args:
            model: Model to evaluate
            dataloader: DataLoader with benchmark data

        Returns:
            Dictionary of evaluation results
        """
        pass

    @abstractmethod
    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate benchmark report with metrics and visualizations.

        Args:
            results: Raw evaluation results from evaluate()

        Returns:
            BenchmarkResult with metrics, plots, and metadata
        """
        pass

    def run(self, model: torch.nn.Module) -> BenchmarkResult:
        """Run the complete benchmark pipeline.

        Args:
            model: Model to evaluate

        Returns:
            BenchmarkResult with metrics and visualizations
        """
        # Ensure output directory exists
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        dataloader = self.load_data()

        # Run evaluation
        results = self.evaluate(model, dataloader)

        # Generate report
        benchmark_result = self.report(results)

        return benchmark_result


class BenchmarkRegistry:
    """Registry for discovering and managing benchmarks.

    Allows registration and retrieval of benchmark implementations.
    """

    def __init__(self) -> None:
        """Initialize empty benchmark registry."""
        self._benchmarks: Dict[str, Type[Benchmark]] = {}

    def register(self, name: str, benchmark_cls: Type[Benchmark]) -> None:
        """Register a benchmark class.

        Args:
            name: Name to register the benchmark under
            benchmark_cls: Benchmark class to register
        """
        if not issubclass(benchmark_cls, Benchmark):
            raise TypeError(f"Benchmark class must inherit from Benchmark, got {benchmark_cls}")
        self._benchmarks[name] = benchmark_cls

    def get(self, name: str) -> Optional[Type[Benchmark]]:
        """Retrieve a benchmark class by name.

        Args:
            name: Name of the benchmark

        Returns:
            Benchmark class or None if not found
        """
        return self._benchmarks.get(name)

    def list_benchmarks(self) -> list[str]:
        """Get list of all registered benchmark names.

        Returns:
            List of benchmark names
        """
        return list(self._benchmarks.keys())

    def create(self, name: str, config: BenchmarkConfig) -> Benchmark:
        """Create a benchmark instance by name.

        Args:
            name: Name of the benchmark
            config: Configuration for the benchmark

        Returns:
            Instantiated benchmark

        Raises:
            ValueError: If benchmark name is not registered
        """
        benchmark_cls = self.get(name)
        if benchmark_cls is None:
            raise ValueError(
                f"Benchmark '{name}' not found. Available: {self.list_benchmarks()}"
            )
        return benchmark_cls(config)


# Global registry instance
_REGISTRY: Optional[BenchmarkRegistry] = None


def get_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry instance.

    Returns:
        Global BenchmarkRegistry singleton
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = BenchmarkRegistry()
    return _REGISTRY


def register_benchmark(name: str) -> Any:
    """Decorator to register a benchmark class.

    Args:
        name: Name to register the benchmark under

    Returns:
        Decorator function

    Example:
        @register_benchmark("perplexity")
        class PerplexityBenchmark(Benchmark):
            ...
    """
    def decorator(cls: Type[Benchmark]) -> Type[Benchmark]:
        registry = get_registry()
        registry.register(name, cls)
        return cls
    return decorator


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRegistry",
    "get_registry",
    "register_benchmark",
]
