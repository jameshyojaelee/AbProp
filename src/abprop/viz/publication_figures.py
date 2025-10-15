"""Reusable figure generators for publications."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_publication_style(style_path: Path | str) -> None:
    plt.style.use(str(style_path))


def plot_training_curves(
    history: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    title: str = "Training Curves",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = history.index if history.index.name else history["epoch"].values
    for metric in metric_columns:
        if metric not in history.columns:
            continue
        ax.plot(epochs, history[metric], label=metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def plot_benchmark_comparison(
    results: pd.DataFrame,
    *,
    metric: str,
    hue: str = "benchmark",
    title: str = "Benchmark Comparison",
) -> plt.Figure:
    pivot = results.pivot(index="model", columns=hue, values=metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title=hue, frameon=False)
    plt.xticks(rotation=45, ha="right")
    return fig


def plot_attention_gallery(image_paths: Sequence[Path], *, title: str = "Attention Maps") -> plt.Figure:
    count = len(image_paths)
    cols = min(count, 3)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(image_path.stem)
    for idx in range(count, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_axis_off()
    fig.suptitle(title)
    return fig


def plot_embedding_panel(
    scatter: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str = "Embedding Projection",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    categories = scatter[color].fillna("unknown")
    palette = {cat: col for cat, col in zip(categories.unique(), plt.cm.tab20.colors)}
    for cat, group in scatter.groupby(categories):
        ax.scatter(group[x], group[y], label=cat, s=12, alpha=0.85)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_error_histogram(
    residuals: Sequence[float],
    *,
    title: str = "Error Distribution",
    bins: int = 40,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=bins, color="#546e7a", alpha=0.85)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return fig


def plot_ablation_summary(
    results: pd.DataFrame,
    *,
    metric: str,
    title: str = "Ablation Summary",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ordered = results.sort_values(metric, ascending=False)
    ax.barh(ordered["ablation"], ordered[metric], color="#8d6e63")
    ax.set_xlabel(metric)
    ax.set_ylabel("Ablation")
    ax.set_title(title)
    ax.invert_yaxis()
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, name: str, formats: Iterable[str] = ("pdf", "png")) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        saved.append(path)
    return saved


__all__ = [
    "apply_publication_style",
    "plot_training_curves",
    "plot_benchmark_comparison",
    "plot_attention_gallery",
    "plot_embedding_panel",
    "plot_error_histogram",
    "plot_ablation_summary",
    "save_figure",
]

