"""Validation helpers for AbProp Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

EXPECTED_COLUMNS: Sequence[str] = (
    "sequence",
    "chain",
    "species",
    "germline_v",
    "germline_j",
    "is_productive",
    "cdr3",
    "length",
    "liability_counts",
    "liability_ln",
    "split",
)

EXPECTED_SPLITS = {"train", "val", "test"}


def _ensure_dict(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise TypeError(f"Expected dict-compatible value, received {type(value)}")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not set(df["split"].unique()).issubset(EXPECTED_SPLITS):
        raise ValueError("Unexpected split labels present in dataset.")

    for column in ["sequence", "chain", "species", "germline_v", "germline_j"]:
        if not pd.api.types.is_string_dtype(df[column]):
            raise TypeError(f"Column {column} must have string dtype.")

    if not pd.api.types.is_bool_dtype(df["is_productive"]):
        raise TypeError("Column is_productive must be boolean.")

    if not pd.api.types.is_integer_dtype(df["length"]):
        raise TypeError("Column length must be integer typed.")

    df["liability_counts"] = df["liability_counts"].apply(_ensure_dict)
    df["liability_ln"] = df["liability_ln"].apply(_ensure_dict)

    for _, row in df.iterrows():
        counts = row["liability_counts"]
        norm = row["liability_ln"]
        length = row["length"]
        for key, value in counts.items():
            if key not in norm:
                raise ValueError(f"Liability key {key} missing from normalized dict.")
            expected = value / length if length else 0.0
            if not _close(norm[key], expected):
                raise ValueError(f"Liability normalization mismatch for key {key}.")

    return df


def validate_parquet_dataset(path: Path | str) -> pd.DataFrame:
    """Load a Parquet dataset directory/file and validate."""
    df = pd.read_parquet(path)
    return validate_dataframe(df)


def _close(lhs: float, rhs: float, tol: float = 1e-6) -> bool:
    return abs(lhs - rhs) <= tol


__all__ = ["validate_parquet_dataset", "validate_dataframe"]

