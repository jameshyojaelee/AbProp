#!/usr/bin/env python3
"""Visualise stratified evaluation metrics across difficulty buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DIMENSION_ORDER = {
    "length": ["short", "medium", "long"],
    "complexity": ["low_entropy", "high_entropy", "unusual_composition"],
    "liability": ["low_liability", "medium_liability", "high_liability"],
    "germline": ["common_germline", "rare_germline", "novel_germline"],
    "species": ["human", "mouse", "other_species"],
}

HEATMAP_METRICS = ["mlm_perplexity", "cls_f1", "reg_r2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot difficulty-aware performance diagnostics.")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="JSON file produced by stratified evaluation (StratifiedEvaluationResult.to_dict()).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mlm_perplexity",
        help="Metric to plot for line charts (default: mlm_perplexity).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots/difficulty"),
        help="Directory to save generated plots.",
    )
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_dataframe(results: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    baseline_metrics = results.get("baseline", {}).get("metrics", {})
    for dimension, entries in results.get("dimensions", {}).items():
        for entry in entries:
            bucket_name = entry["name"].split("/", 1)[-1]
            metrics = entry.get("metrics", {})
            row = {
                "dimension": dimension,
                "bucket": bucket_name,
                "size": entry.get("size", 0),
            }
            for metric_name, value in metrics.items():
                row[metric_name] = value
                if metric_name in baseline_metrics:
                    row[f"{metric_name}_delta"] = value - baseline_metrics[metric_name]
            rows.append(row)
    return pd.DataFrame(rows)


def ensure_order(dimension: str, categories: Sequence[str]) -> List[str]:
    preferred = DEFAULT_DIMENSION_ORDER.get(dimension, [])
    ordered = [cat for cat in preferred if cat in categories]
    for cat in categories:
        if cat not in ordered:
            ordered.append(cat)
    return ordered


def plot_curves(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    for dimension in df["dimension"].unique():
        subset = df[df["dimension"] == dimension].copy()
        categories = ensure_order(dimension, subset["bucket"].tolist())
        subset["bucket"] = pd.Categorical(subset["bucket"], categories=categories, ordered=True)
        subset = subset.sort_values("bucket")

        plt.figure(figsize=(6, 4))
        plt.plot(subset["bucket"], subset[metric], marker="o")
        plt.title(f"{metric} vs difficulty ({dimension})")
        plt.xlabel("Difficulty bucket")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_path = output_dir / f"{dimension}_{metric}_curve.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_heatmap(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    pivot = df.pivot_table(
        index="dimension",
        columns="bucket",
        values=metric,
        aggfunc="mean",
    )
    plt.figure(figsize=(7, max(3, len(pivot) * 0.8)))
    im = plt.imshow(pivot, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=metric)
    plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index)
    plt.title(f"Heatmap of {metric}")
    plt.tight_layout()

    out_path = output_dir / f"heatmap_{metric}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_bars(df: pd.DataFrame, output_dir: Path) -> None:
    if "cls_accuracy" not in df.columns:
        return
    df = df.copy()
    df["cls_error_rate"] = 1.0 - df["cls_accuracy"]

    for dimension in df["dimension"].unique():
        subset = df[df["dimension"] == dimension].copy()
        categories = ensure_order(dimension, subset["bucket"].tolist())
        subset["bucket"] = pd.Categorical(subset["bucket"], categories=categories, ordered=True)
        subset = subset.sort_values("bucket")

        plt.figure(figsize=(6, 4))
        plt.bar(subset["bucket"], subset["cls_error_rate"], color="#d35400")
        plt.title(f"Classification error rate ({dimension})")
        plt.xlabel("Difficulty bucket")
        plt.ylabel("Error rate")
        plt.ylim(0, min(1.0, subset["cls_error_rate"].max() * 1.2))
        plt.tight_layout()

        out_path = output_dir / f"{dimension}_cls_error.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    data = load_results(args.results)
    df = build_dataframe(data)
    if df.empty:
        raise RuntimeError("No metrics found in the provided stratified evaluation file.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metric not in df.columns:
        raise KeyError(f"Metric '{args.metric}' not present in results. Available metrics: {sorted(df.columns)}")

    plot_curves(df, args.metric, output_dir)

    for metric in HEATMAP_METRICS:
        if metric in df.columns:
            plot_heatmap(df, metric, output_dir)

    plot_error_bars(df, output_dir)
    print(f"[INFO] Plots saved under {output_dir}")


if __name__ == "__main__":
    main()

