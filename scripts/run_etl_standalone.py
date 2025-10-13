#!/usr/bin/env python3
"""Standalone ETL runner that doesn't require package installation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abprop.data.etl import ETLConfig, run_etl

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run AbProp ETL pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input TSV/CSV file")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for parquet files")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/val/test split ratios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = ETLConfig(
        input_path=args.input,
        output_dir=args.out,
        splits=tuple(args.splits),
        seed=args.seed,
    )

    print(f"Running ETL pipeline:")
    print(f"  Input: {config.input_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Splits: {config.splits}")
    print(f"  Seed: {config.seed}")
    print()

    df = run_etl(config)

    print(f"\n✓ ETL complete!")
    print(f"  Total sequences: {len(df)}")
    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val: {(df['split'] == 'val').sum()}")
    print(f"  Test: {(df['split'] == 'test').sum()}")
    print(f"\nOutput written to: {config.output_dir}")

if __name__ == "__main__":
    main()
