#!/usr/bin/env python
"""CLI entrypoint for AbProp ETL pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from abprop.data import ETLConfig, run_etl, validate_parquet_dataset
from abprop.utils import DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, transform, and load antibody sequence data into Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw OAS file (TSV/CSV).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_DIR / "processed" / "oas",
        help="Output directory for partitioned Parquet dataset.",
    )
    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=(0.9, 0.05, 0.05),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Ratios for train/val/test splits (must sum to 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling stratified split shuffling.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="If set, validate the resulting Parquet dataset schema.",
    )
    return parser.parse_args()


def run_etl(args: argparse.Namespace) -> None:
    config = ETLConfig(
        input_path=args.input,
        output_dir=args.out,
        splits=(args.splits[0], args.splits[1], args.splits[2]),
        seed=args.seed,
    )
    dataframe = run_etl(config)
    print(
        f"Processed {len(dataframe)} sequences -> {config.output_dir} "
        f"(train={sum(dataframe['split']=='train')}, "
        f"val={sum(dataframe['split']=='val')}, "
        f"test={sum(dataframe['split']=='test')})"
    )
    if args.validate:
        validate_parquet_dataset(config.output_dir)
        print("Validation successful.")


def main() -> None:
    args = parse_args()
    run_etl(args)


if __name__ == "__main__":
    main()
