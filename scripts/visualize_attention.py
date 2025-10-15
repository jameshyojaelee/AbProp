#!/usr/bin/env python3
"""CLI for generating attention visualizations from AbProp checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from abprop.models import AbPropModel, TransformerConfig
from abprop.tokenizers.aa import ID_TO_TOKEN, collate_batch
from abprop.utils import load_yaml_config
from abprop.viz.attention import (
    AttentionCache,
    attention_rollout,
    aggregate_heads,
    normalize_attention,
    plot_attention_heatmap,
    plot_head_grid,
    prepare_regions,
    summarize_attention_to_csv,
    to_numpy_attention,
)

logger = logging.getLogger("visualize_attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pt file.")
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Path to FASTA/text file or raw amino-acid sequence.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for figures.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference (cpu, cuda).")
    parser.add_argument("--model-config", type=Path, help="Optional YAML config for TransformerConfig overrides.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML heatmaps (requires plotly).",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="Limit the number of heads visualized in the grid plots.",
    )
    parser.add_argument(
        "--cdr",
        type=str,
        default="",
        help="Comma-separated 1-based residue ranges for CDRs (e.g. '30-35,50-65,95-105').",
    )
    parser.add_argument(
        "--liabilities",
        type=str,
        default="",
        help="Comma-separated 1-based residue ranges for liability motifs.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="example",
        help="Label used to organize outputs (e.g. 'success', 'failure').",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching attention tensors (defaults to <output>/cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip reading/writing cached attention tensors.",
    )
    return parser.parse_args()


def load_sequence(sequence_arg: str) -> str:
    path = Path(sequence_arg)
    if path.exists():
        content: List[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith(">"):
                    continue
                content.append(stripped)
        seq = "".join(content).upper()
        if not seq:
            raise ValueError(f"No residues found in {path}.")
        return seq
    return sequence_arg.strip().upper()


def build_model(checkpoint_path: Path, model_config_path: Optional[Path], device: torch.device) -> AbPropModel:
    state = torch.load(checkpoint_path, map_location=device)
    cfg: Dict[str, object] = {}
    if model_config_path is not None:
        cfg = load_yaml_config(model_config_path)

    config_data: Dict[str, object] = {}
    for key in ("model_config", "config", "transformer_config"):
        maybe_cfg = state.get(key)
        if isinstance(maybe_cfg, dict):
            config_data = maybe_cfg
            break
    if not config_data and cfg:
        config_data = cfg.get("model", cfg) if isinstance(cfg, dict) else {}

    transformer_cfg = TransformerConfig(**config_data) if config_data else TransformerConfig()

    model = AbPropModel(transformer_cfg).to(device)
    model_state = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    model.eval()
    return model


def parse_ranges(value: str, num_tokens: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    if not value:
        return ranges
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start_val = int(start_str)
            end_val = int(end_str)
        else:
            start_val = end_val = int(item)
        if end_val < start_val:
            start_val, end_val = end_val, start_val
        start_idx = max(0, min(num_tokens - 1, start_val))
        end_idx = min(num_tokens, end_val + 1)
        ranges.append((start_idx, end_idx))
    return ranges


def ensure_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Plotly is required for interactive output. Install with `pip install plotly`."
        ) from exc
    return go


def save_interactive_heatmap(
    matrix: np.ndarray,
    tokens: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    go = ensure_plotly()
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(tokens),
            y=list(tokens),
            colorscale="Inferno",
            zsmooth="best",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        xaxis=dict(tickmode="array", tickvals=list(range(len(tokens))), ticktext=list(tokens), tickangle=45),
        yaxis=dict(tickmode="array", tickvals=list(range(len(tokens))), ticktext=list(tokens)),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    output_root = args.output / args.label
    output_root.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or (args.output / "cache")
    cache = AttentionCache(cache_dir)

    sequence = load_sequence(args.sequence)
    logger.info("Loaded sequence of length %d residues.", len(sequence))

    model = build_model(args.checkpoint, args.model_config, device)

    cached = None
    if not args.no_cache:
        cached = cache.load(args.checkpoint, sequence)

    attentions: Sequence[torch.Tensor]
    token_ids: Sequence[int]
    if cached:
        logger.info("Loaded attention tensors from cache %s.", cache.resolve(args.checkpoint, sequence))
        attentions = cached["attentions"]
        token_ids = cached.get("metadata", {}).get("tokens", [])
    else:
        batch = collate_batch([sequence], add_special=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask,
                tasks=(),
                return_attentions=True,
            )

        attentions = list(outputs.get("attentions", []))
        if not attentions:
            raise RuntimeError("Model did not return attention weights. Ensure encoder supports attention outputs.")
        attentions = [tensor.detach().cpu() for tensor in attentions]
        token_ids = input_ids[0].detach().cpu().tolist()

        if not args.no_cache:
            cache.store(
                args.checkpoint,
                sequence,
                attentions,
                metadata={"tokens": token_ids},
            )

    if not token_ids:
        # Fall back to reconstructing tokens if missing from cache metadata.
        batch = collate_batch([sequence], add_special=True)
        token_ids = batch["input_ids"][0].tolist()

    tokens = [ID_TO_TOKEN.get(int(idx), "<unk>") for idx in token_ids]
    num_tokens = len(tokens)

    cdr_ranges = parse_ranges(args.cdr, num_tokens)
    liability_ranges = parse_ranges(args.liabilities, num_tokens)
    regions = prepare_regions(cdr_ranges, liability_ranges)

    agg_dir = output_root / "aggregated"
    head_dir = output_root / "heads"
    agg_dir.mkdir(exist_ok=True)
    head_dir.mkdir(exist_ok=True)

    rollout = attention_rollout(attentions, add_residual=True)
    rollout_matrix = rollout[0].detach().cpu().numpy()
    rollout_fig = plot_attention_heatmap(
        rollout_matrix,
        tokens,
        title="Attention Rollout",
        regions=regions,
        cmap="viridis",
    )
    rollout_path = agg_dir / "rollout.png"
    rollout_fig.savefig(rollout_path, dpi=220)
    plt_close(rollout_fig)
    logger.info("Saved rollout heatmap to %s", rollout_path)

    if args.interactive:
        try:
            save_interactive_heatmap(rollout_matrix, tokens, "Attention Rollout", agg_dir / "rollout.html")
        except RuntimeError as exc:
            logger.warning("%s", exc)

    for layer_idx, attn in enumerate(attentions):
        aggregated = aggregate_heads(attn, reduction="mean")
        aggregated = normalize_attention(aggregated)
        matrix = to_numpy_attention(aggregated[0])

        title = f"Layer {layer_idx + 1} · Mean Attention"
        fig = plot_attention_heatmap(matrix, tokens, title=title, regions=regions)
        path = agg_dir / f"layer_{layer_idx + 1:02d}_mean.png"
        fig.savefig(path, dpi=220)
        plt_close(fig)
        logger.info("Saved %s", path)

        if args.interactive:
            try:
                save_interactive_heatmap(matrix, tokens, title, agg_dir / f"layer_{layer_idx + 1:02d}_mean.html")
            except RuntimeError as exc:
                logger.warning("%s", exc)

        head_fig = plot_head_grid(attn.cpu(), tokens, layer_index=layer_idx, regions=regions, max_heads=args.max_heads)
        head_path = head_dir / f"layer_{layer_idx + 1:02d}_heads.png"
        head_fig.savefig(head_path, dpi=190)
        plt_close(head_fig)
        logger.info("Saved %s", head_path)

    summary_csv = output_root / "attention_summary.csv"
    summarize_attention_to_csv(attentions, tokens, output_csv=summary_csv)
    logger.info("Wrote summary statistics to %s", summary_csv)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "sequence_length": len(sequence),
        "num_layers": len(attentions),
        "tokens": tokens,
        "ranges": {
            "cdr": cdr_ranges,
            "liabilities": liability_ranges,
        },
    }
    metadata_path = output_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved metadata to %s", metadata_path)


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    main()
