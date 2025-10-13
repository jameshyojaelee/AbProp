#!/usr/bin/env python3
"""
Process real antibody sequences through the ETL pipeline.

This script takes the fetched real antibody data and runs it through
the full AbProp ETL pipeline to create the processed parquet dataset.

Usage:
    python scripts/process_real_data_etl.py --input data/raw/sabdab_sequences.csv --output data/processed/oas_real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abprop.data.etl import ETLConfig, run_etl


def prepare_input_for_etl(input_csv: Path, output_tsv: Path) -> None:
    """Convert fetched sequences to ETL-compatible format."""
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} sequences")

    # Check what columns we have
    print(f"Columns: {list(df.columns)}")

    # Ensure required columns exist
    required = ["sequence", "chain", "species"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add defaults for optional columns if missing
    if "germline_v" not in df.columns:
        df["germline_v"] = "UNKNOWN"
    if "germline_j" not in df.columns:
        df["germline_j"] = "UNKNOWN"
    if "is_productive" not in df.columns:
        df["is_productive"] = True
    if "cdr3" not in df.columns:
        df["cdr3"] = None

    # Select columns in order expected by ETL
    output_df = df[[
        "sequence",
        "chain",
        "species",
        "germline_v",
        "germline_j",
        "is_productive",
        "cdr3",
    ]]

    # Clean up data
    output_df = output_df[output_df["sequence"].str.len() > 0]
    output_df["chain"] = output_df["chain"].str.upper().str[0]
    output_df = output_df[output_df["chain"].isin(["H", "L"])]

    print(f"After cleaning: {len(output_df)} sequences")
    print(f"Species distribution:\n{output_df['species'].value_counts()}")
    print(f"Chain distribution:\n{output_df['chain'].value_counts()}")

    # Save as TSV for ETL
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved ETL input to {output_tsv}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Process real antibody data through ETL")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file with fetched sequences",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed parquet files",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )

    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Prepare intermediate TSV file
    interim_tsv = Path("data/interim") / f"{args.input.stem}_for_etl.tsv"
    prepare_input_for_etl(args.input, interim_tsv)

    # Run ETL pipeline
    print(f"\nRunning ETL pipeline...")
    print(f"  Input: {interim_tsv}")
    print(f"  Output: {args.output}")
    print(f"  Splits: {args.splits}")
    print(f"  Seed: {args.seed}")

    config = ETLConfig(
        input_path=interim_tsv,
        output_dir=args.output,
        splits=tuple(args.splits),
        seed=args.seed,
    )

    try:
        df = run_etl(config)
        print(f"\nETL completed successfully!")
        print(f"Total sequences processed: {len(df)}")
        print(f"\nSplit sizes:")
        print(df["split"].value_counts())
        print(f"\nOutput saved to: {args.output}")
        print(f"\nYou can now use this dataset for training/evaluation:")
        print(f"  - Training: {args.output}/species=*/chain=*/split=train/")
        print(f"  - Validation: {args.output}/species=*/chain=*/split=val/")
        print(f"  - Test: {args.output}/species=*/chain=*/split=test/")

    except Exception as e:
        print(f"ERROR: ETL failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
