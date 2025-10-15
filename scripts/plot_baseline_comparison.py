#!/usr/bin/env python3
"""Plot baseline comparison charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_metric(entries: List[Dict[str, object]], metric: str) -> pd.DataFrame:
    rows = []
    for entry in entries:
        baseline = entry.get("baseline", {})
        metrics = baseline.get("metrics", {})
        name = baseline.get("name", "unknown")
        if metric in metrics:
            rows.append({"baseline": name, "value": metrics[metric]})
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, title: str, ylabel: str, output_path: Path) -> None:
    if df.empty:
        return
    df = df.sort_values("value")
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df["baseline"], df["value"], color="#4C72B0")
    for bar, value in zip(bars, df["value"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot simple baseline comparisons.")
    parser.add_argument("--results", type=Path, required=True, help="Path to baseline_results.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines/plots"))
    parser.add_argument("--metric", type=str, default="perplexity")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args(argv)

    results = load_results(args.results)
    split_data = results.get(args.split, {})

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if "perplexity" in split_data:
        df = extract_metric(split_data["perplexity"], "perplexity")
        plot_bar(df, "Perplexity Baselines", "Perplexity", output_dir / f"{args.split}_perplexity.png")

    if "cdr_classification" in split_data:
        df = extract_metric(split_data["cdr_classification"], "accuracy")
        plot_bar(df, "CDR Classification Baselines", "Accuracy", output_dir / f"{args.split}_cdr_accuracy.png")

    if "liability" in split_data:
        df = extract_metric(split_data["liability"], "mse")
        plot_bar(df, "Liability Regression Baselines", "MSE", output_dir / f"{args.split}_liability_mse.png")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()

