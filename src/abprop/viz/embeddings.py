"""Embedding extraction and analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS


@dataclass
class EmbeddingResult:
    """Container for model embeddings and associated metadata."""

    pooled: np.ndarray
    token: List[np.ndarray]
    attention_mask: List[np.ndarray]
    metadata: pd.DataFrame
    sequences: List[str]


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    if strategy == "mean":
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-6)
        return summed / counts
    if strategy == "cls":
        return hidden_states[:, 0]
    if strategy == "last":
        indices = attention_mask.sum(dim=1) - 1
        gathered = []
        for row, idx in zip(hidden_states, indices):
            gathered.append(row[idx])
        return torch.stack(gathered, dim=0)
    if strategy == "max":
        masked = hidden_states.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
        values, _ = masked.max(dim=1)
        return values
    raise ValueError(f"Unknown pooling strategy '{strategy}'.")


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    pooling: str = "mean",
    return_token: bool = True,
) -> EmbeddingResult:
    """Extract token and pooled embeddings for a dataset."""
    model.eval()
    pooled_batches: List[np.ndarray] = []
    token_embeddings: List[np.ndarray] = []
    attention_masks: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []
    sequences: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids,
                attention_mask,
                tasks=(),
            )
            hidden = outputs["hidden_states"]
            pooled = _pool_hidden_states(hidden, attention_mask, pooling)
            pooled_batches.append(pooled.detach().cpu().numpy())

            mask_cpu = attention_mask.detach().cpu().numpy().astype(bool)
            attention_masks.extend(mask_cpu)
            sequences.extend(batch["sequences"])

            if return_token:
                hidden_cpu = hidden.detach().cpu().numpy()
                for emb, mask in zip(hidden_cpu, mask_cpu):
                    length = int(mask.sum())
                    token_embeddings.append(emb[:length])

            batch_metadata = _collect_metadata(batch)
            metadata_rows.extend(batch_metadata)

    metadata_frame = pd.DataFrame(metadata_rows)
    pooled_array = np.concatenate(pooled_batches, axis=0) if pooled_batches else np.empty((0, 0))
    return EmbeddingResult(
        pooled=pooled_array,
        token=token_embeddings,
        attention_mask=attention_masks,
        metadata=metadata_frame,
        sequences=sequences,
    )


def _collect_metadata(batch: Dict[str, object]) -> List[Dict[str, object]]:
    size = len(batch["sequences"])
    meta: List[Dict[str, object]] = []
    chains = batch.get("chains", ["?"] * size)
    species = batch.get("species", [None] * size)
    germline_v = batch.get("germline_v", [None] * size)
    germline_j = batch.get("germline_j", [None] * size)
    liabilities = batch.get("liability_ln", [{}] * size)
    for idx in range(size):
        entry = {
            "chain": chains[idx],
            "species": species[idx] or "unknown",
            "germline_v": germline_v[idx] or "unknown",
            "germline_j": germline_j[idx] or "unknown",
        }
        liability_entry = liabilities[idx] or {}
        for key in CANONICAL_LIABILITY_KEYS:
            entry[f"liability_{key}"] = float(liability_entry.get(key, np.nan))
        meta.append(entry)
    return meta


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Apply dimensionality reduction to embedding vectors."""
    if embeddings.size == 0:
        return embeddings

    method = method.lower()
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("scikit-learn is required for PCA reduction.") from exc
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("scikit-learn is required for t-SNE reduction.") from exc
        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            **kwargs,
        )
        return tsne.fit_transform(embeddings)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install umap-learn for UMAP reductions.") from exc
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        return reducer.fit_transform(embeddings)

    raise ValueError(f"Unsupported reduction method '{method}'.")


def compute_silhouette(embedding: np.ndarray, labels: Sequence[object]) -> Optional[float]:
    """Compute the silhouette score for given labels."""
    if embedding.size == 0:
        return None
    unique = {label for label in labels if label is not None}
    if len(unique) < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for silhouette scoring.") from exc
    return float(silhouette_score(embedding, labels))


def nearest_neighbor_accuracy(
    embedding: np.ndarray,
    labels: Sequence[object],
    *,
    metric: str = "euclidean",
) -> Optional[float]:
    """Compute leave-one-out nearest-neighbor accuracy for categorical labels."""
    if embedding.size == 0:
        return None
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for nearest neighbor metrics.") from exc
    neighbors = NearestNeighbors(n_neighbors=2, metric=metric)
    neighbors.fit(embedding)
    distances, indices = neighbors.kneighbors(embedding)
    correct = 0
    total = 0
    labels_list = list(labels)
    for row_idx, row in enumerate(indices):
        if len(row) < 2:
            continue
        nn_index = row[1]
        total += 1
        if labels_list[row_idx] == labels_list[nn_index]:
            correct += 1
    if total == 0:
        return None
    return correct / total


def bucketize_liabilities(
    metadata: pd.DataFrame,
    *,
    quantiles: Tuple[float, float] = (0.33, 0.66),
) -> pd.DataFrame:
    """Add categorical liability buckets (low/med/high) to metadata."""
    enriched = metadata.copy()
    for key in CANONICAL_LIABILITY_KEYS:
        column = f"liability_{key}"
        if column not in enriched.columns:
            continue
        series = enriched[column].astype(float)
        if series.isna().all():
            continue
        lower = series.quantile(quantiles[0])
        upper = series.quantile(quantiles[1])
        def categorize(value: float) -> str:
            if np.isnan(value):
                return "missing"
            if value <= lower:
                return "low"
            if value >= upper:
                return "high"
            return "medium"
        enriched[f"{column}_bucket"] = series.apply(categorize)
    return enriched


def load_esm_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load precomputed ESM embeddings stored as a NumPy .npz archive."""
    archive = np.load(path)
    embeddings = archive.get("embeddings")
    if embeddings is None:
        raise ValueError("Archive must contain an 'embeddings' array.")
    sequences = archive.get("sequences")
    if sequences is None:
        raise ValueError("Archive must contain a 'sequences' array aligning with embeddings.")
    metadata = archive.get("metadata")
    if metadata is None:
        metadata = np.array([{}] * len(embeddings), dtype=object)
    return {
        "embeddings": embeddings,
        "sequences": sequences,
        "metadata": metadata,
    }


__all__ = [
    "EmbeddingResult",
    "bucketize_liabilities",
    "compute_silhouette",
    "extract_embeddings",
    "load_esm_embeddings",
    "nearest_neighbor_accuracy",
    "reduce_embeddings",
]

