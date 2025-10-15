#!/usr/bin/env python3
"""Aggregate ablation results and generate summary reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datetime import datetime


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_runs(root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for plan_dir in sorted(root.glob("*")):
        if not plan_dir.is_dir():
            continue
        for metadata_path in plan_dir.rglob("metadata.json"):
            metadata = load_json(metadata_path)
            summary = load_json(metadata_path.with_name("summary.json"))
            record: Dict[str, Any] = {
                "plan": metadata.get("plan", plan_dir.name),
                "experiment": metadata.get("experiment", metadata_path.parent.name),
                "status": metadata.get("status", "unknown"),
                "duration_seconds": metadata.get("duration_seconds"),
                "returncode": metadata.get("returncode"),
                "run_root": metadata.get("run_root", str(metadata_path.parent)),
            }
            tags = metadata.get("tags", {})
            if isinstance(tags, dict):
                for key, value in tags.items():
                    record[f"tag_{key}"] = value
            summary_data = summary if summary else metadata.get("summary", {})
            if isinstance(summary_data, dict):
                for key, value in summary_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            record[f"{key}_{sub_key}"] = sub_value
                    else:
                        record[key] = value
            records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    if "metrics_loss" in df.columns and "final_step" not in df.columns:
        df.rename(columns={"metrics_loss": "loss_latest"}, inplace=True)
    return df


def pareto_front(df: pd.DataFrame, metric: str, time_col: str = "duration_seconds") -> pd.DataFrame:
    if metric not in df or time_col not in df:
        return pd.DataFrame()
    subset = df[["plan", "experiment", metric, time_col, "run_root", "status"]].copy()
    subset = subset[subset["status"] == "completed"]
    subset = subset.dropna(subset=[metric, time_col])
    subset = subset.sort_values(metric, ascending=True)
    pareto_rows = []
    best_time = float("inf")
    for _, row in subset.iterrows():
        current_time = row[time_col]
        if current_time <= best_time:
            pareto_rows.append(row)
            best_time = current_time
    return pd.DataFrame(pareto_rows)


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metric_col = metric
    if metric_col not in df.columns:
        return
    plotted = df[df["status"] == "completed"].copy()
    plotted = plotted.sort_values(metric_col)
    if plotted.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(plotted["experiment"], plotted[metric_col], color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_col)
    plt.title(f"Ablation Summary ({metric_col})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse ablation results.")
    parser.add_argument("--results", type=Path, required=True, help="Root directory produced by run_ablations.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Path to write the aggregated CSV summary.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Path to write the aggregated JSON summary.")
    parser.add_argument("--plot", type=Path, default=None, help="Path to save a bar plot of the chosen metric.")
    parser.add_argument("--metric", type=str, default="metrics_loss", help="Metric column to focus on for plots/Pareto.")
    args = parser.parse_args(argv)

    results_dir = args.results.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    df = collect_runs(results_dir)
    if df.empty:
        print(f"No ablation runs found under {results_dir}.")
        return

    df.sort_values(["plan", "experiment"], inplace=True)

    pareto_df = pareto_front(df, args.metric)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results_dir": str(results_dir),
        "num_runs": int(df.shape[0]),
        "metric": args.metric,
        "records": df.to_dict(orient="records"),
        "pareto": pareto_df.to_dict(orient="records") if not pareto_df.empty else [],
    }

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.summary_csv, index=False)
        print(f"Wrote CSV summary to {args.summary_csv}")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote JSON summary to {args.summary_json}")

    if args.plot:
        plot_metric(df, args.metric, args.plot)
        print(f"Saved plot to {args.plot}")

    print("Pareto-optimal experiments:")
    if pareto_df.empty:
        print("  (No completed runs with the requested metric.)")
    else:
        for _, row in pareto_df.iterrows():
            print(
                f"  {row['plan']} :: {row['experiment']} "
                f"| {args.metric}={row[args.metric]:.4f} "
                f"| duration={row['duration_seconds']:.1f}s"
            )


if __name__ == "__main__":
    main()
