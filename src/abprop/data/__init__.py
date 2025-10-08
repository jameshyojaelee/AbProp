"""Data access utilities for the AbProp project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .dataset import BucketBatchSampler, OASDataset, build_collate_fn
from .etl import ETLConfig, run_etl
from .schema import validate_parquet_dataset

@dataclass
class SequenceRecord:
    """Lightweight representation of an antibody sequence entry."""

    sequence: str
    chain: str
    species: str
    germline_v: str
    germline_j: str


def load_parquet_dataset(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Load and concatenate Parquet files containing antibody sequences."""
    dataframes: List[pd.DataFrame] = []
    for path in paths:
        dataframes.append(pd.read_parquet(path))
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


__all__ = [
    "SequenceRecord",
    "load_parquet_dataset",
    "run_etl",
    "ETLConfig",
    "validate_parquet_dataset",
    "OASDataset",
    "BucketBatchSampler",
    "build_collate_fn",
]
