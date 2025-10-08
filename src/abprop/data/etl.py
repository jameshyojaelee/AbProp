"""OAS ETL utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from abprop.utils import find_motifs, normalize_by_length

RAW_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sequence": ("sequence", "sequence_alignment_aa", "sequence_aa", "seq"),
    "chain": ("chain", "chain_type", "chain_id"),
    "species": ("species", "species_common", "organism"),
    "germline_v": ("v_gene", "germline_v", "v_call"),
    "germline_j": ("j_gene", "germline_j", "j_call"),
    "is_productive": ("is_productive", "productive", "productive_status"),
    "cdr3": ("cdr3", "cdr3_aa"),
}


@dataclass
class ETLConfig:
    input_path: Path
    output_dir: Path
    splits: Tuple[float, float, float]
    seed: int = 42
    partition_cols: Tuple[str, ...] = ("species", "chain", "split")


def read_raw_oas(path: Path) -> pd.DataFrame:
    """Read raw OAS export from TSV or CSV."""
    if not path.exists():
        raise FileNotFoundError(f"OAS file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        sep = "\t"
    elif suffix in {".csv"}:
        sep = ","
    else:
        sep = None

    if sep is None:
        return pd.read_csv(path)
    return pd.read_csv(path, sep=sep)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns into the canonical schema and coerce types."""
    rename_map: Dict[str, str] = {}
    for canonical, aliases in RAW_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
        else:
            if canonical in {"cdr3"}:
                continue
            raise KeyError(f"Missing required column for '{canonical}'. Expected one of {aliases}.")

    df = df.rename(columns=rename_map).copy()

    for column in ["sequence", "chain", "species", "germline_v", "germline_j"]:
        df[column] = df[column].astype(str).str.strip()

    df["sequence"] = df["sequence"].str.upper()
    df["chain"] = df["chain"].str.upper().str[0]
    df["species"] = df["species"].str.lower()
    df["germline_v"] = df["germline_v"].str.upper()
    df["germline_j"] = df["germline_j"].str.upper()

    df = df[df["sequence"].str.len() > 0]
    df = df[df["chain"].isin({"H", "L"})]

    if "is_productive" in df.columns:
        df["is_productive"] = df["is_productive"].apply(_to_bool)
    else:
        df["is_productive"] = True

    if "cdr3" not in df.columns:
        df["cdr3"] = None
    else:
        df["cdr3"] = df["cdr3"].fillna("").astype(str).str.upper()
        df.loc[df["cdr3"] == "", "cdr3"] = pd.NA

    return df


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    return value_str in {"true", "t", "yes", "y", "1", "productive"}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features such as length and liabilities."""
    df = df.copy()
    df["length"] = df["sequence"].str.len()
    liabilities: List[Dict[str, int]] = []
    liabilities_norm: List[Dict[str, float]] = []

    for seq, length in zip(df["sequence"], df["length"]):
        counts = find_motifs(seq)
        liabilities.append(counts)
        liabilities_norm.append(normalize_by_length(counts, length))

    df["liability_counts"] = liabilities
    df["liability_ln"] = liabilities_norm
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate entries by sequence+chain and drop non-productive rows."""
    df = df[df["is_productive"]].copy()
    df["sequence_chain_key"] = df["sequence"] + "|" + df["chain"]
    df = df.drop_duplicates(subset=["sequence_chain_key"])
    df = df.drop(columns=["sequence_chain_key"])
    return df


def assign_clonotype(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate rows with a clonotype key that combines chain and CDR3 / sequence."""
    df = df.copy()
    if df["cdr3"].notna().any():
        df["clonotype_key"] = df["chain"].astype(str) + "|" + df["cdr3"].fillna("NA")
    else:
        df["clonotype_key"] = df["chain"].astype(str) + "|" + df["sequence"]
    return df


def stratified_split(
    df: pd.DataFrame, ratios: Tuple[float, float, float], seed: int
) -> pd.DataFrame:
    """Stratify by species and germline while keeping clonotypes together."""
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-3):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}.")
    df = assign_clonotype(df)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    group_frames: List[pd.DataFrame] = []
    for (_, _), group in df.groupby(["species", "germline_v"], dropna=False):
        keys = group["clonotype_key"].unique().tolist()
        total = len(keys)
        if total == 0:
            continue
        counts = _split_counts(total, ratios)
        n_train, n_val, n_test = counts

        train_keys = keys[:n_train]
        val_keys = keys[n_train : n_train + n_val]
        test_keys = keys[n_train + n_val :]

        group = group.copy()
        group["split"] = "train"
        if val_keys:
            group.loc[group["clonotype_key"].isin(val_keys), "split"] = "val"
        if test_keys:
            group.loc[group["clonotype_key"].isin(test_keys), "split"] = "test"
        group_frames.append(group)

    result = pd.concat(group_frames, ignore_index=True)
    result = result.drop(columns=["clonotype_key"])
    return result


def write_parquet(df: pd.DataFrame, output_dir: Path, partition_cols: Sequence[str]) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory {output_dir} already exists and is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    export_df["liability_counts"] = export_df["liability_counts"].apply(json.dumps)
    export_df["liability_ln"] = export_df["liability_ln"].apply(json.dumps)
    export_df.to_parquet(
        output_dir,
        engine="pyarrow",
        partition_cols=list(partition_cols),
        index=False,
    )


def run_etl(config: ETLConfig) -> pd.DataFrame:
    """End-to-end ETL pipeline producing parquet dataset and returning dataframe."""
    df = read_raw_oas(config.input_path)
    df = normalize_columns(df)
    df = deduplicate(df)
    df = add_features(df)
    df = stratified_split(df, config.splits, config.seed)
    write_parquet(df, config.output_dir, config.partition_cols)
    return df


def _split_counts(total: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    raw_counts = [total * ratio for ratio in ratios]
    base = [int(math.floor(value)) for value in raw_counts]
    remainder = total - sum(base)
    fractions = sorted(
        ((value - math.floor(value), idx) for idx, value in enumerate(raw_counts)),
        reverse=True,
    )
    for i in range(remainder):
        base[fractions[i][1]] += 1
    # Ensure non-negative and does not exceed total
    for idx in range(len(base)):
        if base[idx] < 0:
            base[idx] = 0
    # Adjust final sum if needed
    diff = sum(base) - total
    while diff > 0:
        for idx in range(len(base)-1, -1, -1):
            if base[idx] > 0 and diff > 0:
                base[idx] -= 1
                diff -= 1
            if diff == 0:
                break
    return base[0], base[1], base[2]


__all__ = ["ETLConfig", "run_etl", "read_raw_oas", "normalize_columns"]
