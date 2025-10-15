#!/usr/bin/env python3
"""Check benchmark results for regressions relative to a baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True, help="Path to the latest benchmark JSON.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to the baseline benchmark JSON.")
    parser.add_argument("--metrics", nargs="*", help="Subset of benchmarks to compare (default: all).")
    parser.add_argument("--max-drop", type=float, default=0.02, help="Allowed relative drop (fraction).")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Per-metric threshold overrides (format metric=value).",
    )
    parser.add_argument("--mlflow-uri", type=str, help="Optional MLflow tracking URI.")
    parser.add_argument("--mlflow-experiment", type=str, default="abprop-benchmarks")
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_thresholds(default: float, overrides: list[str]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid threshold spec: {item}")
        key, value = item.split("=", 1)
        thresholds[key] = float(value)
    thresholds["__default__"] = default
    return thresholds


def compare(
    new_data: Dict[str, object],
    ref_data: Dict[str, object],
    metrics: Tuple[str, ...],
    thresholds: Dict[str, float],
) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    new_bench = new_data.get("benchmarks", {})
    ref_bench = ref_data.get("benchmarks", {})
    failures = False
    summary: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        new_entry = new_bench.get(metric)
        ref_entry = ref_bench.get(metric)
        if not new_entry or not ref_entry:
            continue
        new_value = float(new_entry.get("value"))
        ref_value = float(ref_entry.get("value"))
        higher_is_better = bool(new_entry.get("higher_is_better", ref_entry.get("higher_is_better", False)))
        change = new_value - ref_value
        relative = (change / ref_value) if ref_value != 0 else change
        threshold = thresholds.get(metric, thresholds["__default__"])
        summary[metric] = {
            "baseline": ref_value,
            "candidate": new_value,
            "delta": change,
            "relative": relative,
            "higher_is_better": 1.0 if higher_is_better else 0.0,
            "threshold": threshold,
        }
        regression = False
        if higher_is_better:
            regression = new_value + abs(ref_value) * threshold < ref_value
        else:
            regression = new_value - abs(ref_value) * threshold > ref_value
        if regression:
            failures = True
            summary[metric]["status"] = -1
        else:
            summary[metric]["status"] = 1
    return failures, summary


def log_mlflow(args: argparse.Namespace, summary: Dict[str, Dict[str, float]]) -> None:
    if not args.mlflow_uri:
        return
    try:
        import mlflow
    except ImportError:  # pragma: no cover - optional dependency
        print("[WARN] mlflow not installed; skipping logging", file=sys.stderr)
        return
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="benchmark_regression_check"):
        for metric, values in summary.items():
            mlflow.log_metrics({
                f"{metric}_candidate": values["candidate"],
                f"{metric}_baseline": values["baseline"],
                f"{metric}_delta": values["delta"],
            })


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    print("Metric\tBaseline\tCandidate\tDelta\tRelative\tStatus")
    for metric, values in summary.items():
        status = "PASS" if values.get("status", 0) > 0 else "FAIL"
        print(
            f"{metric}\t{values['baseline']:.4f}\t{values['candidate']:.4f}\t"
            f"{values['delta']:.4f}\t{values['relative']:.4f}\t{status}"
        )


def main() -> None:
    args = parse_args()
    new_data = load_results(args.new)
    ref_data = load_results(args.reference)
    available_metrics = tuple(new_data.get("benchmarks", {}).keys())
    metrics = tuple(args.metrics) if args.metrics else available_metrics
    thresholds = compute_thresholds(args.max_drop, args.threshold)
    failures, summary = compare(new_data, ref_data, metrics, thresholds)
    print_summary(summary)
    log_mlflow(args, summary)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

