#!/usr/bin/env python3
"""Example script demonstrating how to use the AbProp benchmark suite programmatically.

This script shows how to:
1. Load a model
2. Configure benchmarks
3. Run individual benchmarks
4. Access and process results
"""

from pathlib import Path

import torch

from abprop.benchmarks import get_registry
from abprop.benchmarks.registry import BenchmarkConfig
from abprop.models import AbPropModel, TransformerConfig


def load_model(checkpoint_path: Path) -> AbPropModel:
    """Load AbProp model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded model in eval mode
    """
    # Create default config (adjust as needed)
    config = TransformerConfig()

    # Instantiate model
    model = AbPropModel(config)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=False)

    model.eval()
    return model


def run_perplexity_benchmark(model: AbPropModel, data_path: Path):
    """Example: Run perplexity benchmark."""
    print("\n" + "=" * 80)
    print("Running Perplexity Benchmark")
    print("=" * 80)

    # Get registry
    registry = get_registry()

    # Configure benchmark
    config = BenchmarkConfig(
        data_path=data_path,
        batch_size=32,
        max_samples=100,  # Limit for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path("outputs/benchmark_demo"),
    )

    # Create benchmark
    benchmark = registry.create("perplexity", config)

    # Run benchmark
    result = benchmark.run(model)

    # Access results
    print("\nMetrics:")
    print(f"  Overall Perplexity: {result.metrics['overall_perplexity']:.2f}")

    if "perplexity_chain_H" in result.metrics:
        print(f"  Heavy Chain Perplexity: {result.metrics['perplexity_chain_H']:.2f}")
    if "perplexity_chain_L" in result.metrics:
        print(f"  Light Chain Perplexity: {result.metrics['perplexity_chain_L']:.2f}")

    print("\nPlots generated:")
    for plot_name, plot_path in result.plots.items():
        print(f"  {plot_name}: {plot_path}")

    print("\nMetadata:")
    print(f"  Number of sequences: {result.metadata['n_sequences']}")
    print(f"  Number of species: {result.metadata['n_species']}")

    return result


def run_liability_benchmark(model: AbPropModel, data_path: Path):
    """Example: Run liability benchmark."""
    print("\n" + "=" * 80)
    print("Running Liability Benchmark")
    print("=" * 80)

    registry = get_registry()

    config = BenchmarkConfig(
        data_path=data_path,
        batch_size=32,
        max_samples=100,  # Limit for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path("outputs/benchmark_demo"),
    )

    benchmark = registry.create("liability", config)
    result = benchmark.run(model)

    print("\nOverall Metrics:")
    print(f"  MSE: {result.metrics['overall_mse']:.4f}")
    print(f"  R²: {result.metrics['overall_r2']:.4f}")
    print(f"  Spearman: {result.metrics['overall_spearman']:.4f}")

    print("\nPer-Liability Metrics:")
    liability_keys = result.metadata["liability_keys"]
    for key in liability_keys[:3]:  # Show first 3
        if f"{key}_r2" in result.metrics:
            print(f"  {key} - R²: {result.metrics[f'{key}_r2']:.4f}")

    return result


def run_cdr_classification_benchmark(model: AbPropModel, data_path: Path):
    """Example: Run CDR classification benchmark."""
    print("\n" + "=" * 80)
    print("Running CDR Classification Benchmark")
    print("=" * 80)

    registry = get_registry()

    config = BenchmarkConfig(
        data_path=data_path,
        batch_size=32,
        max_samples=100,  # Limit for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path("outputs/benchmark_demo"),
    )

    benchmark = registry.create("cdr_classification", config)
    result = benchmark.run(model)

    print("\nClassification Metrics:")
    print(f"  Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"  Precision: {result.metrics['precision']:.4f}")
    print(f"  Recall: {result.metrics['recall']:.4f}")
    print(f"  F1: {result.metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  True Positives: {result.metrics['tp']}")
    print(f"  False Positives: {result.metrics['fp']}")
    print(f"  True Negatives: {result.metrics['tn']}")
    print(f"  False Negatives: {result.metrics['fn']}")

    return result


def main():
    """Main demo function."""
    # Example paths (adjust as needed)
    checkpoint_path = Path("outputs/checkpoints/model_epoch_10.pt")
    data_path = Path("data/processed/oas")

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path")
        return

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Please run ETL pipeline first")
        return

    print("Loading model...")
    model = load_model(checkpoint_path)
    print("Model loaded successfully")

    # Run individual benchmarks
    try:
        perplexity_result = run_perplexity_benchmark(model, data_path)
    except Exception as e:
        print(f"Perplexity benchmark failed: {e}")

    try:
        liability_result = run_liability_benchmark(model, data_path)
    except Exception as e:
        print(f"Liability benchmark failed: {e}")

    try:
        cdr_result = run_cdr_classification_benchmark(model, data_path)
    except Exception as e:
        print(f"CDR classification benchmark failed: {e}")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)
    print("\nResults saved to: outputs/benchmark_demo/")
    print("\nTo run all benchmarks, use:")
    print("  python scripts/run_benchmarks.py --checkpoint <path> --all")


if __name__ == "__main__":
    main()
