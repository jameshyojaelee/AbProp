"""Benchmark that evaluates model performance across stratified difficulty buckets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch

from abprop.eval import StratifiedEvalConfig, evaluate_strata

from .registry import Benchmark, BenchmarkConfig, BenchmarkResult, get_registry


class StratifiedDifficultyBenchmark(Benchmark):
    """Run stratified difficulty evaluation and surface performance cliffs."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)
        self.strata_root = Path(config.data_path)
        if not self.strata_root.exists():
            raise FileNotFoundError(f"Stratified dataset directory not found: {self.strata_root}")

    # These abstract methods are required by the Benchmark base class but are not
    # used directly because the evaluation happens inside `run()`. They are
    # provided solely to satisfy the interface.
    def load_data(self) -> None:  # type: ignore[override]
        return None

    def evaluate(self, model: torch.nn.Module, dataloader) -> Dict[str, object]:  # type: ignore[override]
        return {}

    def report(self, results: Dict[str, object]) -> BenchmarkResult:  # type: ignore[override]
        return BenchmarkResult(self.name, {}, {}, {})

    def run(self, model: torch.nn.Module) -> BenchmarkResult:
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_config = StratifiedEvalConfig(
            strata_root=self.strata_root,
            batch_size=self.config.batch_size,
            device=torch.device(self.config.device),
            num_workers=0,
        )
        result = evaluate_strata(model, eval_config)
        summary = result.to_dict()

        json_path = output_dir / "stratified_metrics.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        metrics = _flatten_metrics(summary)
        metadata = {
            "baseline": summary.get("baseline"),
            "strata_root": str(self.strata_root),
        }

        plots: Dict[str, Path] = {"summary": json_path}
        plots.update(_maybe_generate_plots(json_path, output_dir))

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )


def _flatten_metrics(summary: Dict[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    baseline_metrics = summary.get("baseline", {}).get("metrics", {}) if isinstance(summary.get("baseline"), dict) else {}

    dimensions = summary.get("dimensions", {})
    for dimension, entries in dimensions.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            bucket = entry.get("name", "")
            if "/" in bucket:
                bucket = bucket.split("/", 1)[-1]
            sub_metrics = entry.get("metrics", {})
            if not isinstance(sub_metrics, dict):
                continue
            for metric_name, value in sub_metrics.items():
                key = f"{dimension}:{bucket}:{metric_name}"
                metrics[key] = float(value)
                if metric_name in baseline_metrics:
                    metrics[f"{key}_delta"] = float(value - baseline_metrics[metric_name])
    return metrics


def _maybe_generate_plots(json_path: Path, output_dir: Path) -> Dict[str, Path]:
    plots: Dict[str, Path] = {}
    plot_script = Path(__file__).resolve().parents[3] / "scripts" / "plot_difficulty_performance.py"
    if not plot_script.exists():
        return plots

    plots_dir = output_dir / "plots"
    cmd = [
        sys.executable,
        str(plot_script),
        "--results",
        str(json_path),
        "--output-dir",
        str(plots_dir),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for image_path in plots_dir.glob("*.png"):
            plots[image_path.stem] = image_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return plots


get_registry().register("stratified_difficulty", StratifiedDifficultyBenchmark)


__all__ = ["StratifiedDifficultyBenchmark"]
