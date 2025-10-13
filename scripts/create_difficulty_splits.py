#!/usr/bin/env python3
"""Create difficulty-stratified evaluation splits for AbProp test data.

This utility loads the processed Parquet dataset, identifies the requested
evaluation split (default: ``test``), derives multiple difficulty dimensions,
and materialises balanced Parquet subsets for each difficulty bucket. The goal
is to expose systematic weaknesses that aggregate metrics can hide.

Dimensions:
    - Sequence length (short / medium / long)
    - Sequence complexity (low entropy / high entropy / unusual composition)
    - Liability burden (low / medium / high)
    - Germline frequency (common / rare / novel)
    - Species exposure (human / mouse / other)

Example:
    python scripts/create_difficulty_splits.py \
        --input data/processed/oas_real_full \
        --output data/processed/stratified_test \
        --split test
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

AMINO_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
RARE_AMINO_SYMBOLS = set("BJOUXZ")
DEFAULT_COMMON_GERMLINES = {
    "IGHV3-23",
    "IGHV1-69",
    "IGHV3-30",
    "IGHV3-30-3",
    "IGHV3-11",
    "IGHV3-7",
    "IGHV1-2",
    "IGHLV3-48",
    "IGKV1-39",
    "IGKV3-20",
    "IGKV3-15",
    "IGKV1-5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create difficulty-stratified Parquet splits.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Processed Parquet dataset directory or file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stratified_test"),
        help="Destination directory for stratified Parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to stratify (default: test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per bucket (lower values will drop the bucket).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum samples per bucket (if provided, buckets are capped).",
    )
    return parser.parse_args()


def load_split(input_path: Path, split: str) -> pd.DataFrame:
    """Load the requested dataset split into a DataFrame."""
    df = pd.read_parquet(input_path)
    if "split" not in df.columns:
        raise ValueError("Expected 'split' column in Parquet dataset.")
    subset = df[df["split"].str.lower() == split.lower()].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split '{split}'.")
    return subset.reset_index(drop=True)


def shannon_entropy(sequence: str) -> float:
    """Compute normalized Shannon entropy for the sequence."""
    sequence = (sequence or "").upper()
    if not sequence:
        return 0.0
    counts = Counter(res for res in sequence if res in AMINO_ALPHABET)
    if not counts:
        return 0.0
    probs = np.array([count / len(sequence) for count in counts.values()], dtype=np.float64)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy / math.log(len(AMINO_ALPHABET)))


def classify_length(length: int) -> str:
    if length < 100:
        return "short"
    if length <= 150:
        return "medium"
    return "long"


def classify_complexity(sequence: str) -> str:
    sequence = (sequence or "").upper()
    if not sequence:
        return "low_entropy"

    length = len(sequence)
    non_canonical_frac = sum(res not in AMINO_ALPHABET for res in sequence) / max(length, 1)
    cysteine_frac = sequence.count("C") / max(length, 1)
    proline_frac = sequence.count("P") / max(length, 1)

    if (
        non_canonical_frac >= 0.05
        or cysteine_frac >= 0.18
        or proline_frac >= 0.25
        or any(res in RARE_AMINO_SYMBOLS for res in sequence)
    ):
        return "unusual_composition"

    entropy = shannon_entropy(sequence)
    if entropy < 0.7:
        return "low_entropy"
    return "high_entropy"


def parse_liabilities(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise TypeError(f"Unable to interpret liability entry of type {type(value)}")


def classify_liability(liab_dict: Dict[str, float]) -> Tuple[str, float]:
    score = float(sum(float(v) for v in liab_dict.values()))
    if score < 0.01:
        return "low_liability", score
    if score < 0.03:
        return "medium_liability", score
    return "high_liability", score


def canonicalise_germline(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float) and math.isnan(raw):
        return ""
    text = str(raw).strip().upper()
    if not text or text in {"NA", "NAN", "NONE", "UNASSIGNED"}:
        return ""
    if "*" in text:
        text = text.split("*", 1)[0]
    return text


def classify_germline(
    germline: str,
    freq_map: Dict[str, int],
    common_set: set[str],
    rare_threshold: int,
) -> str:
    if not germline:
        return "novel_germline"
    if germline in common_set:
        return "common_germline"
    count = freq_map.get(germline, 0)
    if count >= rare_threshold:
        return "rare_germline"
    return "novel_germline"


def classify_species(species: str) -> str:
    species_norm = (species or "").strip().lower()
    if species_norm in {"human", "homo sapiens"}:
        return "human"
    if species_norm in {"mouse", "mus musculus"}:
        return "mouse"
    return "other_species"


def derive_germline_sets(freq_map: Dict[str, int]) -> Tuple[set[str], int]:
    sorted_items = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
    # Keep the top-N frequent germlines as "common"
    top_n = {germline for germline, _ in sorted_items[:10]}
    # Augment with known common germlines
    common = set(DEFAULT_COMMON_GERMLINES)
    common.update(top_n)
    rare_threshold = max(5, int(0.01 * sum(freq_map.values())))
    return common, rare_threshold


def balanced_sample(df: pd.DataFrame, column: str, min_samples: int, max_samples: int | None, seed: int) -> Dict[str, pd.DataFrame]:
    """Sample a balanced subset for each value in column."""
    buckets: Dict[str, pd.DataFrame] = {}
    rng = random.Random(seed)

    grouped = df.groupby(column)
    counts = {name: len(group) for name, group in grouped}
    if any(count < min_samples for count in counts.values()):
        return {}
    positive_counts = [count for count in counts.values() if count > 0]
    if not positive_counts:
        return {}
    target = min(positive_counts)
    target = max(min_samples, min(target, max_samples or target))

    for name, group in grouped:
        if len(group) <= target:
            buckets[name] = group.copy()
        else:
            indices = group.index.tolist()
            rng.shuffle(indices)
            selected = indices[:target]
            buckets[name] = group.loc[selected].copy()
    return buckets


def write_buckets(buckets: Dict[str, pd.DataFrame], out_dir: Path, dimension: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for bucket_name, frame in buckets.items():
        bucket_dir = out_dir / dimension
        bucket_dir.mkdir(parents=True, exist_ok=True)
        out_path = bucket_dir / f"{bucket_name}.parquet"
        export = frame.copy()
        export["liability_ln"] = export["liability_ln"].apply(json.dumps)
        if "cdr_mask" in export.columns:
            export["cdr_mask"] = export["cdr_mask"].apply(lambda value: json.dumps(value) if value is not None else None)
        export.to_parquet(out_path, index=False)
        counts[bucket_name] = len(frame)
    return counts


def main() -> None:
    args = parse_args()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    df = load_split(args.input, args.split)

    # Compute derived columns
    df["length_bucket"] = df["length"].apply(lambda x: classify_length(int(x)))
    df["complexity_bucket"] = df["sequence"].apply(classify_complexity)
    df["species_bucket"] = df["species"].apply(classify_species)

    liabilities = df["liability_ln"].apply(parse_liabilities)
    liab_categories, liab_scores = zip(*(classify_liability(entry) for entry in liabilities))
    df["liability_bucket"] = liab_categories
    df["liability_score"] = liab_scores

    df["germline_root"] = df["germline_v"].apply(canonicalise_germline)
    germline_counts = df["germline_root"].value_counts().to_dict()
    common_germlines, rare_threshold = derive_germline_sets(germline_counts)
    df["germline_bucket"] = df["germline_root"].apply(
        lambda g: classify_germline(g, germline_counts, common_germlines, rare_threshold)
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    dimensions = {
        "length": "length_bucket",
        "complexity": "complexity_bucket",
        "liability": "liability_bucket",
        "germline": "germline_bucket",
        "species": "species_bucket",
    }

    for dimension, column in dimensions.items():
        buckets = balanced_sample(df, column, args.min_samples, args.max_samples, rng_seed)
        if not buckets:
            print(f"[WARN] Skipping dimension '{dimension}': insufficient data after filtering.")
            continue
        summary[dimension] = write_buckets(buckets, output_dir, dimension)
        print(f"[INFO] Wrote {len(buckets)} buckets for dimension '{dimension}': {summary[dimension]}")

    # Persist summary metadata
    if summary:
        summary_path = output_dir / "stratification_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[INFO] Stratification summary saved to {summary_path}.")
    else:
        raise RuntimeError("No stratified buckets were generated – check dataset coverage and thresholds.")


if __name__ == "__main__":
    main()
