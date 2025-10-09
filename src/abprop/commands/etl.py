"""ETL command entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from abprop.data import ETLConfig, run_etl, validate_parquet_dataset
from abprop.utils import DEFAULT_DATA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, transform, and load antibody sequence data into Parquet."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw OAS TSV/CSV path.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the resulting Parquet dataset schema after ETL.",
    )
    return parser


def execute(args: argparse.Namespace) -> None:
    config = ETLConfig(
        input_path=args.input,
        output_dir=args.out,
        splits=tuple(args.splits),
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    execute(args)


__all__ = ["main", "execute", "build_parser"]

