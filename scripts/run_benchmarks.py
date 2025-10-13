#!/usr/bin/env python3
"""Run AbProp benchmarks with parallel execution and comprehensive reporting.

Usage:
    python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --config configs/benchmarks.yaml
    python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks perplexity liability
    python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from abprop.benchmarks import get_registry
from abprop.benchmarks.registry import BenchmarkConfig, BenchmarkResult
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import load_yaml_config, mlflow_default_tags, mlflow_log_artifact, mlflow_log_metrics, mlflow_run


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark suite for AbProp models."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks.yaml"),
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Specific benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per benchmark (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run benchmarks in parallel (experimental)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report",
    )
    return parser


def instantiate_model(model_cfg: Dict, checkpoint_path: Path, device: str) -> AbPropModel:
    """Load model from checkpoint.

    Args:
        model_cfg: Model configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded AbProp model
    """
    # Build model config
    defaults = TransformerConfig()
    task_weights = model_cfg.get("task_weights", {})
    config = TransformerConfig(
        vocab_size=model_cfg.get("vocab_size", defaults.vocab_size),
        d_model=model_cfg.get("d_model", defaults.d_model),
        nhead=model_cfg.get("nhead", defaults.nhead),
        num_layers=model_cfg.get("num_layers", defaults.num_layers),
        dim_feedforward=model_cfg.get("dim_feedforward", defaults.dim_feedforward),
        dropout=model_cfg.get("dropout", defaults.dropout),
        max_position_embeddings=model_cfg.get("max_position_embeddings", defaults.max_position_embeddings),
        liability_keys=tuple(model_cfg.get("liability_keys", list(defaults.liability_keys))),
        mlm_weight=task_weights.get("mlm", defaults.mlm_weight),
        cls_weight=task_weights.get("cls", defaults.cls_weight),
        reg_weight=task_weights.get("reg", defaults.reg_weight),
    )

    # Instantiate model
    model = AbPropModel(config)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=False)

    model.to(device)
    model.eval()

    return model


def run_single_benchmark(
    benchmark_name: str,
    benchmark_config: BenchmarkConfig,
    model: AbPropModel,
) -> BenchmarkResult:
    """Run a single benchmark.

    Args:
        benchmark_name: Name of the benchmark
        benchmark_config: Configuration for the benchmark
        model: Model to evaluate

    Returns:
        BenchmarkResult with metrics and plots
    """
    print(f"\n{'=' * 80}")
    print(f"Running benchmark: {benchmark_name}")
    print(f"{'=' * 80}")

    registry = get_registry()
    benchmark = registry.create(benchmark_name, benchmark_config)

    start_time = time.time()
    result = benchmark.run(model)
    elapsed_time = time.time() - start_time

    print(f"\nBenchmark '{benchmark_name}' completed in {elapsed_time:.2f}s")
    print(f"Metrics: {list(result.metrics.keys())}")
    print(f"Plots: {list(result.plots.keys())}")

    return result


def generate_html_report(
    results: Dict[str, BenchmarkResult],
    output_path: Path,
) -> None:
    """Generate HTML report with all benchmark results.

    Args:
        results: Dictionary of benchmark results
        output_path: Path to save HTML report
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AbProp Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .header { text-align: center; margin-bottom: 40px; }
            .benchmark-section { background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .benchmark-title { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .metrics-table th { background-color: #3498db; color: white; padding: 10px; text-align: left; }
            .metrics-table td { padding: 10px; border-bottom: 1px solid #ddd; }
            .metrics-table tr:hover { background-color: #f5f5f5; }
            .plot-container { margin: 20px 0; }
            .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
            .metadata { background-color: #ecf0f1; padding: 10px; border-radius: 4px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AbProp Benchmark Report</h1>
            <p>Comprehensive evaluation of antibody property prediction model</p>
        </div>
    """

    for benchmark_name, result in results.items():
        html += f"""
        <div class="benchmark-section">
            <h2 class="benchmark-title">{benchmark_name.replace('_', ' ').title()}</h2>

            <h3>Metrics</h3>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
        """

        for metric_name, metric_value in result.metrics.items():
            if isinstance(metric_value, (float, int)):
                display_value = f"{metric_value:.4f}"
            else:
                display_value = str(metric_value)
            html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{display_value}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        """

        if result.plots:
            html += "<h3>Visualizations</h3>"
            for plot_name, plot_path in result.plots.items():
                html += f"""
                <div class="plot-container">
                    <h4>{plot_name.replace('_', ' ').title()}</h4>
                    <img src="{plot_path}" alt="{plot_name}">
                </div>
                """

        html += f"""
            <div class="metadata">
                <strong>Metadata:</strong> {json.dumps(result.metadata, indent=2)}
            </div>
        </div>
        """

    html += """
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report saved to: {output_path}")


def main(argv: list[str] | None = None) -> None:
    """Main entrypoint for benchmark runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load configurations
    if args.config.exists():
        benchmark_cfg = load_yaml_config(args.config)
    else:
        print(f"Warning: Benchmark config not found at {args.config}, using defaults")
        benchmark_cfg = {}

    model_cfg = load_yaml_config(args.model_config)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = instantiate_model(model_cfg, args.checkpoint, args.device)
    print("Model loaded successfully")

    # Register all benchmarks
    # Import to trigger registration
    from abprop.benchmarks import cdr_classification_benchmark  # noqa: F401
    from abprop.benchmarks import developability_benchmark  # noqa: F401
    from abprop.benchmarks import liability_benchmark  # noqa: F401
    from abprop.benchmarks import perplexity_benchmark  # noqa: F401
    from abprop.benchmarks import stratified_benchmark  # noqa: F401
    from abprop.benchmarks import zero_shot_benchmark  # noqa: F401

    registry = get_registry()
    available_benchmarks = registry.list_benchmarks()
    print(f"Available benchmarks: {available_benchmarks}")

    # Determine which benchmarks to run
    if args.all or args.benchmarks is None:
        benchmarks_to_run = available_benchmarks
    else:
        benchmarks_to_run = args.benchmarks
        # Validate
        invalid = [b for b in benchmarks_to_run if b not in available_benchmarks]
        if invalid:
            raise ValueError(f"Invalid benchmarks: {invalid}. Available: {available_benchmarks}")

    print(f"Running benchmarks: {benchmarks_to_run}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build benchmark configs
    benchmark_configs = {}
    for benchmark_name in benchmarks_to_run:
        # Get benchmark-specific config or use defaults
        bench_specific_cfg = benchmark_cfg.get(benchmark_name, {})
        data_path = Path(bench_specific_cfg.get("data_path", "data/processed/oas"))

        benchmark_configs[benchmark_name] = BenchmarkConfig(
            data_path=data_path,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device,
            output_dir=args.output_dir,
            mlflow_tracking=not args.no_mlflow,
        )

    # Run benchmarks
    results: Dict[str, BenchmarkResult] = {}

    if args.parallel and len(benchmarks_to_run) > 1:
        print("\nRunning benchmarks in parallel...")
        # Note: Parallel execution may have issues with GPU memory
        # This is experimental
        with ProcessPoolExecutor(max_workers=min(4, len(benchmarks_to_run))) as executor:
            futures = {
                executor.submit(
                    run_single_benchmark,
                    name,
                    benchmark_configs[name],
                    model,
                ): name
                for name in benchmarks_to_run
            }

            for future in as_completed(futures):
                benchmark_name = futures[future]
                try:
                    result = future.result()
                    results[benchmark_name] = result
                except Exception as exc:
                    print(f"Benchmark {benchmark_name} failed: {exc}")
    else:
        # Sequential execution
        for benchmark_name in benchmarks_to_run:
            try:
                result = run_single_benchmark(
                    benchmark_name,
                    benchmark_configs[benchmark_name],
                    model,
                )
                results[benchmark_name] = result
            except Exception as exc:
                print(f"Benchmark {benchmark_name} failed: {exc}")
                import traceback
                traceback.print_exc()

    # Save summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    summary = {}
    for benchmark_name, result in results.items():
        print(f"\n{benchmark_name}:")
        summary[benchmark_name] = result.metrics
        for metric_name, metric_value in list(result.metrics.items())[:5]:  # Show top 5
            print(f"  {metric_name}: {metric_value}")

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Generate HTML report
    if args.html_report:
        html_path = args.output_dir / "report.html"
        generate_html_report(results, html_path)

    # MLflow logging
    if not args.no_mlflow:
        print("\nLogging results to MLflow...")
        with mlflow_run("abprop-benchmarks", tags=mlflow_default_tags()):
            # Log all metrics
            for benchmark_name, result in results.items():
                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow_log_metrics({f"{benchmark_name}/{metric_name}": metric_value})

            # Log summary
            mlflow_log_artifact(summary_path, artifact_path="benchmarks")

            # Log plots
            for benchmark_name, result in results.items():
                for plot_name, plot_path in result.plots.items():
                    mlflow_log_artifact(plot_path, artifact_path=f"benchmarks/{benchmark_name}")

            if args.html_report and html_path.exists():
                mlflow_log_artifact(html_path, artifact_path="benchmarks")

    print("\n✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
