#!/usr/bin/env python3
"""Visualize AbProp embedding spaces with dimensionality reduction and metrics."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection side-effects
from torch.utils.data import DataLoader

from abprop.data import OASDataset, build_collate_fn
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import load_yaml_config
from abprop.viz.embeddings import (
    EmbeddingResult,
    bucketize_liabilities,
    compute_silhouette,
    extract_embeddings,
    load_esm_embeddings,
    nearest_neighbor_accuracy,
    reduce_embeddings,
)

logger = logging.getLogger("visualize_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="Model checkpoint paths.")
    parser.add_argument("--labels", type=str, nargs="*", help="Optional labels for checkpoints (same order).")
    parser.add_argument("--parquet", type=Path, required=True, help="Directory containing parquet splits.")
    parser.add_argument("--splits", type=str, nargs="+", default=["val"], help="Dataset splits to visualize.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional token cap per batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (0 avoids fork issues).")
    parser.add_argument("--device", type=str, default="cpu", help="Device for forward passes (cpu, cuda, cuda:0).")
    parser.add_argument("--model-config", type=Path, help="YAML overrides for TransformerConfig.")
    parser.add_argument(
        "--reducers",
        type=str,
        nargs="+",
        default=["umap"],
        help="Dimensionality reduction methods (umap, pca, tsne).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[2],
        help="Target embedding dimensionalities (2 or 3).",
    )
    parser.add_argument(
        "--color-fields",
        type=str,
        nargs="+",
        default=["species", "chain", "germline_v"],
        help="Metadata fields for coloring/metrics.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "last", "max"],
        help="Pooling strategy for sequence embeddings.",
    )
    parser.add_argument("--esm2", type=Path, help="Optional npz archive containing ESM embeddings.")
    parser.add_argument("--interactive", action="store_true", help="Write interactive HTML via plotly.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reducers.")
    return parser.parse_args()


def ensure_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Plotly required for interactive scatter plots. Install with `pip install plotly`.") from exc
    return go


@dataclass
class SourceEmbeddings:
    label: str
    result: EmbeddingResult


def load_model(checkpoint_path: Path, config_path: Optional[Path], device: torch.device) -> AbPropModel:
    state = torch.load(checkpoint_path, map_location=device)
    cfg: Dict[str, object] = {}
    if config_path is not None:
        cfg = load_yaml_config(config_path)
    config_dict: Dict[str, object] = {}
    for key in ("model_config", "config", "transformer_config"):
        payload = state.get(key)
        if isinstance(payload, dict):
            config_dict = payload
            break
    if not config_dict and cfg:
        config_dict = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    config = TransformerConfig(**config_dict) if config_dict else TransformerConfig()
    model = AbPropModel(config).to(device)
    model_state = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        logger.warning("Missing keys for %s: %s", checkpoint_path, missing)
    if unexpected:
        logger.warning("Unexpected keys for %s: %s", checkpoint_path, unexpected)
    model.eval()
    return model


def build_dataloaders(
    parquet_dir: Path,
    splits: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    max_tokens: Optional[int],
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        dataset = OASDataset(parquet_dir, split=split)
        collate = build_collate_fn(generate_mlm=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )
        loaders[split] = dataloader
    return loaders


def concat_results(results: Sequence[EmbeddingResult]) -> EmbeddingResult:
    if not results:
        return EmbeddingResult(
            pooled=np.empty((0, 0)),
            token=[],
            attention_mask=[],
            metadata=pd.DataFrame(),
            sequences=[],
        )
    pooled = np.concatenate([res.pooled for res in results if res.pooled.size], axis=0)
    tokens: List[np.ndarray] = []
    attention: List[np.ndarray] = []
    metadata = pd.concat([res.metadata for res in results], ignore_index=True)
    sequences: List[str] = []
    for res in results:
        tokens.extend(res.token)
        attention.extend(res.attention_mask)
        sequences.extend(res.sequences)
    return EmbeddingResult(
        pooled=pooled,
        token=tokens,
        attention_mask=attention,
        metadata=metadata,
        sequences=sequences,
    )


def get_source_label(checkpoint: Path, labels: Optional[Sequence[str]], index: int) -> str:
    if labels and index < len(labels):
        return labels[index]
    return checkpoint.stem or f"checkpoint_{index}"


def scatter_palette(categories: Sequence[object]) -> Tuple[List[Tuple[float, float, float, float]], Dict[object, Tuple]]:
    unique = pd.Series(categories).fillna("unknown").unique()
    cmap = plt.get_cmap("tab20", len(unique))
    mapping = {category: cmap(idx) for idx, category in enumerate(unique)}
    colors = [mapping.get(category, (0.6, 0.6, 0.6, 1.0)) for category in categories]
    return colors, mapping


def save_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, output_path: Path, title: str) -> None:
    colors, mapping = scatter_palette(df[color_col])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df[x_col], df[y_col], c=colors, s=12, alpha=0.85, edgecolors="none")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=str(label), markerfacecolor=color, markersize=6)
               for label, color in mapping.items()]
    ax.legend(handles=handles, title=color_col, loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_scatter_3d(df: pd.DataFrame, cols: Tuple[str, str, str], color_col: str, output_path: Path, title: str) -> None:
    colors, mapping = scatter_palette(df[color_col])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], c=colors, s=12, alpha=0.8, depthshade=True)
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    ax.set_title(title)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=str(label), markerfacecolor=color, markersize=6)
               for label, color in mapping.items()]
    ax.legend(handles=handles, title=color_col, loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_density_heatmap(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    heat, xedges, yedges = np.histogram2d(df[x_col], df[y_col], bins=60)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heat.T,
        origin="lower",
        cmap="magma",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_interactive_scatter(
    df: pd.DataFrame,
    columns: Sequence[str],
    color_col: str,
    title: str,
    output_path: Path,
) -> None:
    go = ensure_plotly()
    color_values = df[color_col].fillna("unknown").astype(str)
    if len(columns) == 2:
        fig = go.Figure(
            data=go.Scattergl(
                x=df[columns[0]],
                y=df[columns[1]],
                mode="markers",
                marker=dict(color=color_values, size=6, opacity=0.8),
                text=df[color_col],
            )
        )
    else:
        fig = go.Figure(
            data=go.Scatter3d(
                x=df[columns[0]],
                y=df[columns[1]],
                z=df[columns[2]],
                mode="markers",
                marker=dict(size=5, color=color_values, opacity=0.8),
                text=df[color_col],
            )
        )
    fig.update_layout(title=title, legend_title=color_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        args.parquet,
        args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
    )

    sources: List[SourceEmbeddings] = []
    for idx, checkpoint in enumerate(args.checkpoints):
        label = get_source_label(checkpoint, args.labels, idx)
        logger.info("Extracting embeddings for %s (%s)", label, checkpoint)
        model = load_model(checkpoint, args.model_config, device)
        split_results: List[EmbeddingResult] = []
        for split, dataloader in loaders.items():
            logger.info("  Processing split=%s", split)
            result = extract_embeddings(
                model,
                dataloader,
                device=device,
                pooling=args.pooling,
            )
            if not result.metadata.empty:
                result.metadata["split"] = split
            split_results.append(result)
        merged = concat_results(split_results)
        merged.metadata["source"] = label
        merged.metadata["checkpoint"] = str(checkpoint)
        enriched_meta = bucketize_liabilities(merged.metadata)
        merged = EmbeddingResult(
            pooled=merged.pooled,
            token=merged.token,
            attention_mask=merged.attention_mask,
            metadata=enriched_meta,
            sequences=merged.sequences,
        )
        sources.append(SourceEmbeddings(label=label, result=merged))

    if args.esm2:
        logger.info("Loading ESM embeddings from %s", args.esm2)
        esm_payload = load_esm_embeddings(args.esm2)
        meta_records = []
        meta_raw = esm_payload["metadata"]
        if isinstance(meta_raw, np.ndarray):
            for row in meta_raw:
                if isinstance(row, dict):
                    meta_records.append(row)
                else:
                    meta_records.append({})
        else:
            meta_records = [{} for _ in range(len(esm_payload["embeddings"]))]
        metadata = pd.DataFrame(meta_records)
        metadata["source"] = "esm2"
        metadata["checkpoint"] = "esm2"
        metadata["sequence"] = esm_payload["sequences"]
        result = EmbeddingResult(
            pooled=esm_payload["embeddings"],
            token=[],
            attention_mask=[],
            metadata=bucketize_liabilities(metadata),
            sequences=list(esm_payload["sequences"]),
        )
        sources.append(SourceEmbeddings(label="esm2", result=result))

    metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for reducer in args.reducers:
        for dim in args.dimensions:
            if dim not in (2, 3):
                logger.warning("Skipping unsupported dimension %d", dim)
                continue
            frames: List[pd.DataFrame] = []
            logger.info("Running %s reduction to %dD", reducer, dim)
            for source in sources:
                if source.result.pooled.size == 0:
                    logger.warning("No pooled embeddings for %s; skipping.", source.label)
                    continue
                coords = reduce_embeddings(
                    source.result.pooled,
                    method=reducer,
                    n_components=dim,
                    random_state=args.random_state,
                )
                frame = source.result.metadata.copy()
                columns = [f"{reducer}_{idx}" for idx in range(dim)]
                for idx_dim, column in enumerate(columns):
                    frame[column] = coords[:, idx_dim]
                frame["sequence"] = source.result.sequences
                frames.append(frame)

                metrics.setdefault(source.label, {}).setdefault(reducer, {})
                for field in args.color_fields:
                    if field not in frame.columns:
                        continue
                    labels = frame[field].fillna("unknown").tolist()
                    silhouette = compute_silhouette(coords, labels)
                    nn_acc = nearest_neighbor_accuracy(coords, labels)
                    metrics[source.label][reducer][field] = {
                        "silhouette": silhouette if silhouette is not None else float("nan"),
                        "nearest_neighbor_accuracy": nn_acc if nn_acc is not None else float("nan"),
                    }

                subdir = output_dir / f"{reducer}_{dim}d" / source.label
                subdir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(subdir / "embedded_points.csv", index=False)
                if dim == 2:
                    x_col, y_col = columns[0], columns[1]
                    for field in args.color_fields:
                        if field not in frame.columns:
                            continue
                        save_scatter_2d(frame, x_col, y_col, field, subdir / f"scatter_{field}.png",
                                        title=f"{source.label} · {reducer.upper()} ({field})")
                        if args.interactive:
                            try:
                                save_interactive_scatter(
                                    frame,
                                    (x_col, y_col),
                                    field,
                                    title=f"{source.label} · {reducer.upper()} ({field})",
                                    output_path=subdir / f"scatter_{field}.html",
                                )
                            except RuntimeError as exc:
                                logger.warning("%s", exc)
                    save_density_heatmap(
                        frame,
                        x_col,
                        y_col,
                        subdir / "density.png",
                        title=f"{source.label} · {reducer.upper()} Density",
                    )
                else:
                    cols = (columns[0], columns[1], columns[2])
                    for field in args.color_fields:
                        if field not in frame.columns:
                            continue
                        save_scatter_3d(
                            frame,
                            cols,
                            field,
                            subdir / f"scatter_{field}.png",
                            title=f"{source.label} · {reducer.upper()} 3D ({field})",
                        )
                        if args.interactive:
                            try:
                                save_interactive_scatter(
                                    frame,
                                    cols,
                                    field,
                                    title=f"{source.label} · {reducer.upper()} 3D ({field})",
                                    output_path=subdir / f"scatter_{field}.html",
                                )
                            except RuntimeError as exc:
                                logger.warning("%s", exc)

            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined_dir = output_dir / f"{reducer}_{dim}d" / "comparison"
                combined_dir.mkdir(parents=True, exist_ok=True)
                combined.to_csv(combined_dir / "embedded_points.csv", index=False)
                if dim == 2:
                    x_col, y_col = f"{reducer}_0", f"{reducer}_1"
                    save_scatter_2d(
                        combined,
                        x_col,
                        y_col,
                        "source",
                        combined_dir / "scatter_source.png",
                        title=f"{reducer.upper()} Comparison by Source",
                    )
                    if args.interactive:
                        try:
                            save_interactive_scatter(
                                combined,
                                (x_col, y_col),
                                "source",
                                title=f"{reducer.upper()} Comparison by Source",
                                output_path=combined_dir / "scatter_source.html",
                            )
                        except RuntimeError as exc:
                            logger.warning("%s", exc)
                else:
                    cols = (f"{reducer}_0", f"{reducer}_1", f"{reducer}_2")
                    save_scatter_3d(
                        combined,
                        cols,
                        "source",
                        combined_dir / "scatter_source.png",
                        title=f"{reducer.upper()} 3D Comparison by Source",
                    )
                    if args.interactive:
                        try:
                            save_interactive_scatter(
                                combined,
                                cols,
                                "source",
                                title=f"{reducer.upper()} 3D Comparison by Source",
                                output_path=combined_dir / "scatter_source.html",
                            )
                        except RuntimeError as exc:
                            logger.warning("%s", exc)

    metrics_path = output_dir / "embedding_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
