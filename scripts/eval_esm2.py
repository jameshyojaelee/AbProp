#!/usr/bin/env python3
"""Evaluate an ESM-2 baseline across the AbProp benchmark suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from contextlib import nullcontext

from abprop.baselines import ESM2Baseline, ESM2Config
from abprop.benchmarks import get_registry
from abprop.benchmarks.registry import BenchmarkConfig, BenchmarkResult
from abprop.utils import load_yaml_config, mlflow_default_tags, mlflow_log_artifact, mlflow_log_dict, mlflow_log_metrics, mlflow_run
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ESM-2 baseline on AbProp benchmarks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.yaml"))
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/esm2_benchmarks"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--html-report", action="store_true")
    parser.add_argument("--runtime-tracking", action="store_true")
    parser.add_argument("--memory-tracking", action="store_true")
    return parser


def generate_html_report(results: Dict[str, BenchmarkResult], output_path: Path) -> None:
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESM-2 Baseline Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .header { text-align: center; margin-bottom: 40px; }
            .benchmark-section { background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .benchmark-title { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .metrics-table th { background-color: #3498db; color: white; padding: 10px; text-align: left; }
            .metrics-table td { padding: 10px; border-bottom: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ESM-2 Baseline Benchmark Report</h1>
            <p>Automated evaluation summary for ESM-2 probes.</p>
        </div>
    """

    for name, result in results.items():
        html += f"""
        <div class="benchmark-section">
            <h2 class="benchmark-title">{name.title()}</h2>
            <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th></tr>
        """
        for metric, value in result.metrics.items():
            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        html += "</table>"
        if result.metadata:
            html += "<h3>Metadata</h3><pre>" + json.dumps(result.metadata, indent=2) + "</pre>"
        if result.plots:
            html += "<h3>Artifacts</h3><ul>"
            for plot_name, path in result.plots.items():
                html += f"<li>{plot_name}: {path}</li>"
            html += "</ul>"
        html += "</div>"

    html += """
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def load_model(config: Dict[str, object], device: torch.device) -> ESM2Baseline:
    model_cfg = config.get("model", {})
    esm_config = ESM2Config(
        model_name=model_cfg.get("model_name", "esm2_t33_650M_UR50D"),
        repr_layer=int(model_cfg.get("repr_layer", 33)),
        probe_dropout=float(model_cfg.get("probe_dropout", 0.1)),
        liability_keys=tuple(model_cfg.get("liability_keys", CANONICAL_LIABILITY_KEYS)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        half_precision=bool(model_cfg.get("half_precision", False)),
        tied_token_head=bool(model_cfg.get("tied_token_head", True)),
        tasks=tuple(model_cfg.get("tasks", ("mlm", "cls", "reg"))),
        sequence_pooling=str(model_cfg.get("sequence_pooling", "mean")),
    )
    model = ESM2Baseline(esm_config).to(device)
    return model


def build_benchmark_config(
    cfg: Dict[str, object],
    benchmark_name: str,
    batch_size_override: Optional[int],
    max_samples_override: Optional[int],
    device: torch.device,
) -> BenchmarkConfig:
    benchmark_cfg = cfg.get(benchmark_name, {})
    data_path = Path(benchmark_cfg.get("data_path", "./data/processed/oas"))
    batch_size = batch_size_override or benchmark_cfg.get("batch_size", cfg.get("batch_size", 32))
    max_samples = max_samples_override or benchmark_cfg.get("max_samples")
    return BenchmarkConfig(
        data_path=data_path,
        batch_size=int(batch_size),
        max_samples=None if max_samples in {None, "null"} else int(max_samples),
        device=str(device),
        output_dir=Path(cfg.get("output_dir", "outputs/esm2_benchmarks")),
        mlflow_tracking=bool(cfg.get("mlflow_tracking", True)),
    )


def run_benchmark(
    benchmark_name: str,
    benchmark_config: BenchmarkConfig,
    model: ESM2Baseline,
    *,
    track_runtime: bool,
    track_memory: bool,
) -> BenchmarkResult:
    registry = get_registry()
    benchmark = registry.create(benchmark_name, benchmark_config)

    start = time.time()
    if track_memory and torch.cuda.is_available() and benchmark_config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(benchmark_config.device))
    result = benchmark.run(model)
    elapsed = time.time() - start

    metadata = dict(result.metadata)
    if track_runtime:
        metadata["runtime_seconds"] = elapsed
    if track_memory and torch.cuda.is_available() and benchmark_config.device.startswith("cuda"):
        peak_memory = torch.cuda.max_memory_allocated(torch.device(benchmark_config.device))
        metadata["peak_memory_mb"] = peak_memory / (1024**2)
    return BenchmarkResult(
        benchmark_name=result.benchmark_name,
        metrics=result.metrics,
        plots=result.plots,
        metadata=metadata,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config)
    baseline_cfg = cfg.get("baselines", {}).get("esm2", {})
    eval_cfg = baseline_cfg.get("evaluation", {})

    output_dir = args.output_dir or Path(eval_cfg.get("output_dir", "outputs/esm2_benchmarks"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or eval_cfg.get("device")
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = load_model(baseline_cfg, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    probe_state = checkpoint.get("probe_state", checkpoint)
    model.load_probe_state_dict(probe_state)
    model.eval()

    registry = get_registry()
    available = registry.list_benchmarks()
    if args.all or not args.benchmarks:
        benchmarks = available
    else:
        missing = [name for name in args.benchmarks if name not in available]
        if missing:
            raise ValueError(f"Unknown benchmarks {missing}. Available: {available}")
        benchmarks = args.benchmarks

    results: Dict[str, BenchmarkResult] = {}
    with mlflow_run("esm2-benchmark", tags=mlflow_default_tags()) if not args.no_mlflow else nullcontext():
        for benchmark_name in benchmarks:
            bench_config = build_benchmark_config(
                cfg,
                benchmark_name,
                batch_size_override=args.batch_size or eval_cfg.get("batch_size"),
                max_samples_override=args.max_samples or eval_cfg.get("max_samples"),
                device=device,
            )

            print(f"\n=== Running {benchmark_name} ===")
            result = run_benchmark(
                benchmark_name,
                bench_config,
                model,
                track_runtime=args.runtime_tracking or eval_cfg.get("runtime_tracking", False),
                track_memory=args.memory_tracking or eval_cfg.get("memory_tracking", False),
            )
            results[benchmark_name] = result

            if not args.no_mlflow:
                mlflow_log_metrics({f"{benchmark_name}_{k}": v for k, v in result.metrics.items()})
                for plot_name, path in result.plots.items():
                    if Path(path).exists():
                        mlflow_log_artifact(Path(path), artifact_path=benchmark_name)
                mlflow_log_dict(result.metadata, f"{benchmark_name}/metadata.json")

    summary = {
        name: {
            "metrics": result.metrics,
            "metadata": result.metadata,
        }
        for name, result in results.items()
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.html_report or eval_cfg.get("html_report", False):
        report_path = output_dir / "report.html"
        generate_html_report(results, report_path)
        if not args.no_mlflow:
            mlflow_log_artifact(report_path, artifact_path="reports")

    print("\nBenchmark evaluation complete.")
    for name, result in results.items():
        best_metrics = ", ".join(f"{metric}={value:.4f}" for metric, value in result.metrics.items())
        print(f"- {name}: {best_metrics}")


if __name__ == "__main__":
    main()
