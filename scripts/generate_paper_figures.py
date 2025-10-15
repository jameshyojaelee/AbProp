#!/usr/bin/env python3
"""Generate publication-ready figures for AbProp papers or decks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from abprop.viz.publication_figures import (
    apply_publication_style,
    plot_ablation_summary,
    plot_attention_gallery,
    plot_benchmark_comparison,
    plot_embedding_panel,
    plot_error_histogram,
    plot_training_curves,
    save_figure,
)

logger = logging.getLogger("generate_paper_figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=Path, default=Path("configs/publication.mplstyle"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures/publication"))
    parser.add_argument("--figures", nargs="+", default=["all"], help="Figures to generate (all, training, benchmarks, attention, embeddings, errors, ablations)")
    parser.add_argument("--training-metrics", type=Path, help="CSV with training history (epoch, train_loss, val_loss, ...)")
    parser.add_argument("--benchmark-results", type=Path, help="CSV or JSON with benchmark metrics (model, benchmark, metric).")
    parser.add_argument("--attention-dir", type=Path, help="Directory with attention PNGs (aggregated outputs).")
    parser.add_argument("--embedding-csv", type=Path, help="CSV containing reduced embeddings (columns: x, y, label).")
    parser.add_argument("--embedding-x", type=str, default="umap_0")
    parser.add_argument("--embedding-y", type=str, default="umap_1")
    parser.add_argument("--embedding-color", type=str, default="source")
    parser.add_argument("--error-json", type=Path, help="JSON file with residuals array.")
    parser.add_argument("--ablation-csv", type=Path, help="CSV with ablation results (ablation, metric).")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats (pdf/png/svg).")
    return parser.parse_args()


def load_training_history(path: Path) -> pd.DataFrame:
    if path and path.is_file():
        logger.info("Loading training history from %s", path)
        return pd.read_csv(path)
    logger.warning("Training metrics not provided; generating synthetic curve for layout testing.")
    epochs = list(range(1, 11))
    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": [2.0 / (e ** 0.4) for e in epochs],
            "val_loss": [2.2 / (e ** 0.35) for e in epochs],
        }
    )


def load_benchmark_results(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        logger.warning("Benchmark results missing; creating placeholder table.")
        return pd.DataFrame(
            {
                "model": ["AbProp", "Baseline"],
                "benchmark": ["Perplexity", "Perplexity"],
                "metric": [1.8, 2.4],
            }
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        return pd.DataFrame(payload)
    return pd.read_csv(path)


def load_embeddings(path: Path) -> pd.DataFrame:
    if path and path.is_file():
        return pd.read_csv(path)
    logger.warning("Embedding CSV not found; generating synthetic blob.")
    data = {
        "umap_0": [0.1, 0.2, 0.8, 1.0, -0.4, -0.5],
        "umap_1": [0.0, 0.1, 0.9, 0.8, -0.3, -0.2],
        "source": ["AbProp", "AbProp", "AbProp", "Baseline", "Baseline", "Baseline"],
    }
    return pd.DataFrame(data)


def load_residuals(path: Path) -> List[float]:
    if path and path.is_file():
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("residuals", [])
        return [float(x) for x in payload]
    logger.warning("Error JSON missing; using Gaussian samples for placeholder.")
    return list(np.random.normal(loc=0.0, scale=0.2, size=200))


def load_ablation(path: Path) -> pd.DataFrame:
    if path and path.is_file():
        return pd.read_csv(path)
    logger.warning("Ablation CSV missing; fabricating mock results.")
    return pd.DataFrame(
        {
            "ablation": ["Full", "No CDR loss", "No Liabilities"],
            "metric": [0.91, 0.84, 0.79],
        }
    )


def collect_attention_images(root: Path, limit: int = 6) -> List[Path]:
    paths: List[Path] = []
    if root and root.is_dir():
        for image in sorted(root.glob("*.png"))[:limit]:
            paths.append(image)
    if not paths:
        logger.warning("No attention images found in %s", root)
    return paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_publication_style(args.style)

    figures = set(args.figures)
    if "all" in figures:
        figures = {"training", "benchmarks", "attention", "embeddings", "errors", "ablations"}

    if "training" in figures:
        history = load_training_history(args.training_metrics)
        fig = plot_training_curves(history, metric_columns=[col for col in history.columns if col != "epoch"])
        save_figure(fig, args.output, "training_curves", formats=args.formats)

    if "benchmarks" in figures:
        results = load_benchmark_results(args.benchmark_results)
        fig = plot_benchmark_comparison(results, metric="metric")
        save_figure(fig, args.output, "benchmark_comparison", formats=args.formats)

    if "attention" in figures:
        images = collect_attention_images(args.attention_dir) if args.attention_dir else []
        if images:
            fig = plot_attention_gallery(images)
            save_figure(fig, args.output, "attention_gallery", formats=args.formats)

    if "embeddings" in figures:
        embedding_df = load_embeddings(args.embedding_csv)
        fig = plot_embedding_panel(
            embedding_df,
            x=args.embedding_x,
            y=args.embedding_y,
            color=args.embedding_color,
        )
        save_figure(fig, args.output, "embeddings", formats=args.formats)

    if "errors" in figures:
        residuals = load_residuals(args.error_json)
        if residuals:
            fig = plot_error_histogram(residuals)
            save_figure(fig, args.output, "error_distribution", formats=args.formats)

    if "ablations" in figures:
        ablation_df = load_ablation(args.ablation_csv)
        fig = plot_ablation_summary(ablation_df, metric="metric")
        save_figure(fig, args.output, "ablations", formats=args.formats)


if __name__ == "__main__":
    main()
