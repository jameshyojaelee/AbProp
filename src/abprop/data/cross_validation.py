"""Clonotype-aware cross-validation utilities for robust performance estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from torch.utils.data import Dataset

from abprop.data.dataset import OASDataset


@dataclass
class CVFold:
    """Represents a single cross-validation fold with train/val/test indices."""

    fold_idx: int
    train_indices: List[int]
    val_indices: List[int]
    test_indices: Optional[List[int]] = None


class ClonotypeAwareKFold:
    """
    K-Fold cross-validation splitter that ensures clonotypes don't leak across folds.

    This splitter groups sequences by clonotype (chain + CDR3 or chain + sequence)
    and ensures that all sequences from the same clonotype end up in the same fold.
    Additionally, it stratifies by species and chain to maintain balanced distributions.

    Args:
        n_splits: Number of folds (default: 5)
        shuffle: Whether to shuffle data before splitting (default: True)
        seed: Random seed for reproducibility (default: 42)
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 42) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(
        self,
        df: pd.DataFrame,
        clonotype_col: str = "clonotype_key",
        stratify_cols: Optional[Sequence[str]] = None,
    ) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """
        Generate train/validation indices for k-fold CV.

        Args:
            df: DataFrame with sequences
            clonotype_col: Column name containing clonotype keys
            stratify_cols: Columns to stratify by (default: ["species", "chain"])

        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        if stratify_cols is None:
            stratify_cols = ["species", "chain"]

        # Assign clonotype if not present
        if clonotype_col not in df.columns:
            df = self._assign_clonotype(df)

        # Shuffle if requested
        if self.shuffle:
            df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        # Group by stratification columns and split within each group
        all_train_indices: List[int] = []
        all_val_indices_per_fold: List[List[int]] = [[] for _ in range(self.n_splits)]

        for _, group in df.groupby(list(stratify_cols), dropna=False):
            fold_assignments = self._assign_clonotypes_to_folds(group, clonotype_col)

            for fold_idx in range(self.n_splits):
                val_mask = fold_assignments == fold_idx
                val_indices = group.index[val_mask].tolist()
                all_val_indices_per_fold[fold_idx].extend(val_indices)

        # Generate train/val splits for each fold
        all_indices = set(df.index)
        for fold_idx in range(self.n_splits):
            val_indices = all_val_indices_per_fold[fold_idx]
            train_indices = sorted(all_indices - set(val_indices))
            yield train_indices, val_indices

    def _assign_clonotype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign clonotype keys based on chain and CDR3 (or sequence)."""
        df = df.copy()
        if "cdr3" in df.columns and df["cdr3"].notna().any():
            df["clonotype_key"] = df["chain"].astype(str) + "|" + df["cdr3"].fillna("NA")
        else:
            df["clonotype_key"] = df["chain"].astype(str) + "|" + df["sequence"]
        return df

    def _assign_clonotypes_to_folds(
        self, group: pd.DataFrame, clonotype_col: str
    ) -> pd.Series:
        """Assign each clonotype in a group to a fold."""
        unique_clonotypes = group[clonotype_col].unique()
        n_clonotypes = len(unique_clonotypes)

        # Calculate target size for each fold
        fold_size = n_clonotypes // self.n_splits
        remainder = n_clonotypes % self.n_splits

        # Assign clonotypes to folds
        clonotype_to_fold: Dict[str, int] = {}
        clonotype_idx = 0

        for fold_idx in range(self.n_splits):
            # Add one extra clonotype to first 'remainder' folds
            current_fold_size = fold_size + (1 if fold_idx < remainder else 0)

            for _ in range(current_fold_size):
                if clonotype_idx < n_clonotypes:
                    clonotype_to_fold[unique_clonotypes[clonotype_idx]] = fold_idx
                    clonotype_idx += 1

        # Map each row to its fold
        return group[clonotype_col].map(clonotype_to_fold).fillna(-1).astype(int)


class GroupKFoldDataset(Dataset):
    """
    Wrapper around OASDataset that provides k-fold cross-validation splits.

    This dataset wrapper allows you to train on specific folds while ensuring
    clonotype-aware splitting and stratification.

    Args:
        parquet_dir: Path to parquet dataset directory
        fold_idx: Which fold to use as validation (0 to n_splits-1)
        n_splits: Total number of folds (default: 5)
        split_type: Either "train" or "val" to select which data to load
        columns: Columns to load from parquet
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        parquet_dir: Path | str,
        fold_idx: int,
        n_splits: int = 5,
        split_type: str = "train",
        columns: Optional[Sequence[str]] = None,
        seed: int = 42,
    ) -> None:
        if split_type not in {"train", "val"}:
            raise ValueError(f"split_type must be 'train' or 'val', got {split_type}")
        if not (0 <= fold_idx < n_splits):
            raise ValueError(f"fold_idx must be in [0, {n_splits}), got {fold_idx}")

        self.parquet_dir = Path(parquet_dir)
        self.fold_idx = fold_idx
        self.n_splits = n_splits
        self.split_type = split_type
        self.seed = seed

        # Load the full training split from the original dataset
        # (We'll split this into CV folds)
        df = pd.read_parquet(
            self.parquet_dir,
            filters=[("split", "=", "train")],
        )

        if df.empty:
            raise ValueError(f"No training data found in {self.parquet_dir}")

        # Perform clonotype-aware k-fold splitting
        splitter = ClonotypeAwareKFold(n_splits=n_splits, shuffle=True, seed=seed)
        splits = list(splitter.split(df))
        train_indices, val_indices = splits[fold_idx]

        # Select appropriate indices based on split_type
        if split_type == "train":
            selected_indices = train_indices
        else:  # val
            selected_indices = val_indices

        self.data = df.iloc[selected_indices].reset_index(drop=True)

        # Parse JSON fields
        if "liability_ln" in self.data.columns:
            from abprop.data.dataset import _maybe_to_dict
            self.data["liability_ln"] = self.data["liability_ln"].apply(_maybe_to_dict)

        if "cdr_mask" in self.data.columns:
            from abprop.data.dataset import _maybe_to_list
            self.data["cdr_mask"] = self.data["cdr_mask"].apply(_maybe_to_list)

        self.lengths = self.data["length"].astype(int).tolist()
        self.has_cdr = "cdr_mask" in self.data.columns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.data.iloc[idx]
        item: Dict[str, object] = {
            "sequence": row["sequence"],
            "chain": row["chain"],
            "liability_ln": row["liability_ln"],
            "length": int(row["length"]),
        }
        if self.has_cdr:
            cdr_mask = row["cdr_mask"]
            item["cdr_mask"] = cdr_mask
        return item


def generate_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> List[CVFold]:
    """
    Generate k-fold cross-validation splits with optional held-out test set.

    Args:
        df: DataFrame containing full dataset
        n_splits: Number of CV folds (default: 5)
        test_ratio: Proportion of data to hold out for final testing (default: 0.0)
        seed: Random seed for reproducibility

    Returns:
        List of CVFold objects, one per fold
    """
    df = df.copy()

    # Hold out test set if requested
    test_indices: Optional[List[int]] = None
    if test_ratio > 0:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = int(len(df) * test_ratio)
        test_indices = df.index[:n_test].tolist()
        df = df.iloc[n_test:].reset_index(drop=True)

    # Perform k-fold splitting on remaining data
    splitter = ClonotypeAwareKFold(n_splits=n_splits, shuffle=True, seed=seed)
    folds: List[CVFold] = []

    for fold_idx, (train_indices, val_indices) in enumerate(splitter.split(df)):
        fold = CVFold(
            fold_idx=fold_idx,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
        folds.append(fold)

    return folds


def stratified_cv_summary(df: pd.DataFrame, folds: List[CVFold]) -> pd.DataFrame:
    """
    Generate summary statistics for CV folds showing stratification quality.

    Args:
        df: Original DataFrame
        folds: List of CV folds

    Returns:
        DataFrame with fold statistics
    """
    summary_data: List[Dict[str, object]] = []

    for fold in folds:
        train_data = df.iloc[fold.train_indices]
        val_data = df.iloc[fold.val_indices]

        summary_data.append({
            "fold": fold.fold_idx,
            "split": "train",
            "n_sequences": len(train_data),
            "n_heavy": (train_data["chain"] == "H").sum(),
            "n_light": (train_data["chain"] == "L").sum(),
            "n_species": train_data["species"].nunique(),
        })

        summary_data.append({
            "fold": fold.fold_idx,
            "split": "val",
            "n_sequences": len(val_data),
            "n_heavy": (val_data["chain"] == "H").sum(),
            "n_light": (val_data["chain"] == "L").sum(),
            "n_species": val_data["species"].nunique(),
        })

    return pd.DataFrame(summary_data)


__all__ = [
    "ClonotypeAwareKFold",
    "GroupKFoldDataset",
    "CVFold",
    "generate_cv_folds",
    "stratified_cv_summary",
]
