"""Attention visualization helpers for AbProp models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from abprop.tokenizers.aa import decode


@dataclass(frozen=True)
class HighlightRegion:
    """Region to highlight on attention maps."""

    label: str
    start: int
    end: int
    kind: str = "cdr"
    color: Optional[str] = None

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end


class AttentionCache:
    """File-backed cache for attention tensors keyed by checkpoint + sequence."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_key(checkpoint_path: Path, sequence: str) -> str:
        digest = hashlib.sha1()
        digest.update(str(checkpoint_path.resolve()).encode("utf-8"))
        digest.update(sequence.encode("utf-8"))
        return digest.hexdigest()[:16]

    def resolve(self, checkpoint_path: Path, sequence: str) -> Path:
        key = self._hash_key(checkpoint_path, sequence)
        return self.output_dir / f"attention_{key}.pt"

    def load(
        self,
        checkpoint_path: Path,
        sequence: str,
    ) -> Optional[Dict[str, object]]:
        cache_path = self.resolve(checkpoint_path, sequence)
        if not cache_path.exists():
            return None
        return torch.load(cache_path, map_location="cpu")

    def store(
        self,
        checkpoint_path: Path,
        sequence: str,
        attentions: Sequence[torch.Tensor],
        metadata: Optional[Dict[str, object]] = None,
    ) -> Path:
        cache_path = self.resolve(checkpoint_path, sequence)
        payload = {
            "attentions": [tensor.cpu() for tensor in attentions],
            "metadata": metadata or {},
        }
        torch.save(payload, cache_path)
        return cache_path


def to_numpy_attention(attn: torch.Tensor) -> np.ndarray:
    """Convert an attention tensor to numpy format."""
    return attn.detach().float().cpu().numpy()


def aggregate_heads(attn: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Aggregate head dimension of attention weights."""
    if attn.dim() != 4:
        raise ValueError(f"Expected attention tensor of shape (B, H, T, S). Got {tuple(attn.shape)}.")
    if reduction == "mean":
        return attn.mean(dim=1)
    if reduction == "max":
        values, _ = attn.max(dim=1)
        return values
    if reduction == "sum":
        return attn.sum(dim=1)
    raise ValueError(f"Unsupported reduction '{reduction}'.")


def normalize_attention(attn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize attention weights over the source dimension."""
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attn / denom


def attention_rollout(
    attentions: Sequence[torch.Tensor],
    *,
    add_residual: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute attention rollout across layers."""
    if not attentions:
        raise ValueError("attention_rollout expects at least one attention tensor.")
    rollout: Optional[torch.Tensor] = None
    for layer_attn in attentions:
        layer = aggregate_heads(layer_attn, reduction=reduction)
        if add_residual:
            identity = torch.eye(layer.size(-1), device=layer.device, dtype=layer.dtype)
            layer = layer + identity.unsqueeze(0)
        layer = normalize_attention(layer)
        rollout = layer if rollout is None else torch.bmm(layer, rollout)
    assert rollout is not None
    return rollout


def decode_tokens(input_ids: Sequence[int], strip_special: bool = True) -> List[str]:
    """Decode token ids into amino acid characters for axis labels."""
    sequence = decode(input_ids, strip_special=strip_special)
    return list(sequence)


def default_palette(kind: str) -> str:
    if kind == "cdr":
        return "#ff8a65"
    if kind == "liability":
        return "#4fc3f7"
    return "#c5e1a5"


def _apply_region_highlights(
    ax: plt.Axes,
    regions: Sequence[HighlightRegion],
    *,
    length: int,
    axis: str,
) -> None:
    for region in regions:
        color = region.color or default_palette(region.kind)
        start, end = region.start, region.end
        if axis == "x":
            ax.add_patch(
                patches.Rectangle(
                    (start - 0.5, -0.5),
                    end - start,
                    length,
                    linewidth=0.0,
                    edgecolor=None,
                    facecolor=color,
                    alpha=0.08,
                )
            )
            ax.text(
                (start + end - 1) / 2,
                -1.2,
                region.label,
                ha="center",
                va="top",
                fontsize=8,
                color=color,
            )
        else:
            ax.add_patch(
                patches.Rectangle(
                    (-0.5, start - 0.5),
                    length,
                    end - start,
                    linewidth=0.0,
                    edgecolor=None,
                    facecolor=color,
                    alpha=0.08,
                )
            )


def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: Sequence[str],
    *,
    title: str,
    regions: Optional[Sequence[HighlightRegion]] = None,
    cmap: str = "inferno",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot a single attention heatmap."""
    num_tokens = len(tokens)
    figure = plt.figure(figsize=figsize or (max(6, num_tokens * 0.35), max(4, num_tokens * 0.35)))
    ax = figure.add_subplot(111)
    im = ax.imshow(attention, cmap=cmap, vmin=0.0, vmax=float(np.max(attention) + 1e-8))
    ax.set_xticks(range(num_tokens))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if regions:
        _apply_region_highlights(ax, regions, length=num_tokens, axis="x")
        _apply_region_highlights(ax, regions, length=num_tokens, axis="y")

    figure.tight_layout()
    return figure


def plot_head_grid(
    attention: torch.Tensor,
    tokens: Sequence[str],
    *,
    layer_index: int,
    regions: Optional[Sequence[HighlightRegion]] = None,
    max_heads: Optional[int] = None,
) -> plt.Figure:
    """Visualize all heads within a layer."""
    if attention.dim() != 4:
        raise ValueError("Expected attention tensor of shape (B, H, T, S).")
    batch_size, num_heads, tgt_len, src_len = attention.shape
    if batch_size != 1:
        raise ValueError("plot_head_grid currently expects batch size of 1.")
    if tgt_len != src_len:
        raise ValueError("plot_head_grid expects square attention matrices.")
    if max_heads is not None:
        num_heads = min(num_heads, max_heads)
        attention = attention[:, :num_heads]

    cols = min(4, num_heads)
    rows = int(np.ceil(num_heads / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 3.5 * rows),
        squeeze=False,
    )
    tokens_list = list(tokens)

    for head_idx in range(num_heads):
        row = head_idx // cols
        col = head_idx % cols
        ax = axes[row][col]
        head_matrix = to_numpy_attention(attention[0, head_idx])
        im = ax.imshow(head_matrix, cmap="magma", vmin=0.0, vmax=float(np.max(head_matrix) + 1e-8))
        ax.set_xticks(range(len(tokens_list)))
        ax.set_xticklabels(tokens_list, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tokens_list)))
        ax.set_yticklabels(tokens_list, fontsize=7)
        ax.set_title(f"Layer {layer_index + 1} · Head {head_idx + 1}", fontsize=10)
        if regions:
            _apply_region_highlights(ax, regions, length=len(tokens_list), axis="x")
            _apply_region_highlights(ax, regions, length=len(tokens_list), axis="y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots.
    for idx in range(num_heads, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig


def summarize_attention_to_csv(
    attentions: Sequence[torch.Tensor],
    tokens: Sequence[str],
    *,
    output_csv: Path,
    reduction: str = "mean",
) -> None:
    """Write aggregated attention statistics for downstream analysis."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stacked = []
    for layer_idx, attn in enumerate(attentions):
        agg = aggregate_heads(attn, reduction=reduction)
        agg = normalize_attention(agg)
        data = agg[0].detach().cpu().numpy()
        stacked.append(
            np.stack(
                [
                    np.full(len(tokens), layer_idx, dtype=np.int32),
                    np.arange(len(tokens), dtype=np.int32),
                    data.diagonal(),
                ],
                axis=1,
            )
        )
    concatenated = np.concatenate(stacked, axis=0)
    header = "layer,position,self_attention"
    np.savetxt(output_csv, concatenated, delimiter=",", header=header, comments="")


def prepare_regions(
    cdr_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    liability_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[HighlightRegion]:
    """Build HighlightRegion instances from raw tuples."""
    regions: List[HighlightRegion] = []
    if cdr_ranges:
        for idx, (start, end) in enumerate(cdr_ranges, start=1):
            regions.append(HighlightRegion(label=f"CDR{idx}", start=start, end=end, kind="cdr"))
    if liability_ranges:
        for idx, (start, end) in enumerate(liability_ranges, start=1):
            regions.append(
                HighlightRegion(label=f"Liability {idx}", start=start, end=end, kind="liability")
            )
    return regions


__all__ = [
    "AttentionCache",
    "HighlightRegion",
    "aggregate_heads",
    "attention_rollout",
    "normalize_attention",
    "plot_attention_heatmap",
    "plot_head_grid",
    "summarize_attention_to_csv",
    "prepare_regions",
]
