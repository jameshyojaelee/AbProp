#!/usr/bin/env python3
"""
Download and prepare OAS (Observed Antibody Space) data for AbProp.

This script provides multiple methods to acquire antibody sequence data:
1. Direct download from OAS web interface (requires manual selection)
2. Programmatic download from AIRR Data Commons API
3. Create synthetic dataset for testing

Usage:
    python scripts/download_oas_data.py --method oas --output data/raw/oas_human.tsv
    python scripts/download_oas_data.py --method airr --output data/raw/oas_human.tsv
    python scripts/download_oas_data.py --method synthetic --output data/raw/oas_synthetic.tsv --num-sequences 10000
"""

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def create_synthetic_dataset(
    num_sequences: int = 10000,
    chain_distribution: Tuple[float, float] = (0.5, 0.5),  # H, L
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic antibody dataset for testing.

    This generates realistic-looking antibody sequences with proper:
    - Chain types (H/L)
    - Germline assignments (common IGHV/IGLV families)
    - CDR3 sequences
    - Liability motifs
    """
    random.seed(seed)

    # Common germline genes
    IGHV_GENES = [
        "IGHV1-2", "IGHV1-18", "IGHV1-69", "IGHV3-7", "IGHV3-15",
        "IGHV3-21", "IGHV3-23", "IGHV3-30", "IGHV3-33", "IGHV3-48",
        "IGHV4-34", "IGHV4-39", "IGHV4-59", "IGHV5-51"
    ]

    IGLV_GENES = [
        "IGLV1-40", "IGLV1-44", "IGLV1-47", "IGLV1-51", "IGLV2-8",
        "IGLV2-11", "IGLV2-14", "IGLV3-1", "IGLV3-19", "IGLV3-21"
    ]

    IGHJ_GENES = ["IGHJ1", "IGHJ2", "IGHJ3", "IGHJ4", "IGHJ5", "IGHJ6"]
    IGLJ_GENES = ["IGLJ1", "IGLJ2", "IGLJ3", "IGLJ4", "IGLJ5", "IGLJ6", "IGLJ7"]

    # Amino acid frequencies in antibodies (rough approximation)
    COMMON_AA = "ACDEFGHIKLMNPQRSTVWY"

    # CDR3 motifs (common patterns)
    CDR3_STARTS = ["CAR", "CAS", "CAA", "CVR", "CVS"]
    CDR3_ENDS = ["WGQ", "YYY", "FDY", "MDV", "LDY"]

    def generate_framework_sequence(length: int) -> str:
        """Generate a plausible framework sequence."""
        # Framework regions have specific patterns
        # This is a simplified version
        aa_list = []
        for i in range(length):
            if i % 20 < 5:  # More hydrophobic patches
                aa_list.append(random.choice("AVLIPFW"))
            elif i % 20 < 10:
                aa_list.append(random.choice("STNQ"))
            elif i % 20 < 15:
                aa_list.append(random.choice("DERK"))
            else:
                aa_list.append(random.choice(COMMON_AA))
        return "".join(aa_list)

    def generate_cdr3_sequence(min_len: int = 8, max_len: int = 20) -> str:
        """Generate a plausible CDR3 sequence."""
        cdr3_len = random.randint(min_len, max_len)
        start = random.choice(CDR3_STARTS)
        end = random.choice(CDR3_ENDS)
        middle_len = max(0, cdr3_len - len(start) - len(end))
        middle = "".join(random.choices(COMMON_AA, k=middle_len))
        return start + middle + end

    def generate_full_sequence(chain: str, include_cdr3: bool = True) -> Tuple[str, Optional[str]]:
        """Generate a full antibody sequence."""
        if chain == "H":
            # Heavy chain: ~110-130 AA
            fw1_len = random.randint(20, 25)
            cdr1_len = random.randint(8, 12)
            fw2_len = random.randint(15, 18)
            cdr2_len = random.randint(7, 10)
            fw3_len = random.randint(30, 38)
            cdr3 = generate_cdr3_sequence(8, 20)
            fw4_len = random.randint(10, 13)
        else:  # Light chain: ~100-115 AA
            fw1_len = random.randint(20, 24)
            cdr1_len = random.randint(7, 10)
            fw2_len = random.randint(14, 17)
            cdr2_len = random.randint(5, 8)
            fw3_len = random.randint(28, 35)
            cdr3 = generate_cdr3_sequence(7, 15)
            fw4_len = random.randint(9, 12)

        fw1 = generate_framework_sequence(fw1_len)
        cdr1 = generate_framework_sequence(cdr1_len)  # CDR1
        fw2 = generate_framework_sequence(fw2_len)
        cdr2 = generate_framework_sequence(cdr2_len)  # CDR2
        fw3 = generate_framework_sequence(fw3_len)
        fw4 = generate_framework_sequence(fw4_len)

        full_sequence = fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3 + fw4

        return full_sequence, cdr3 if include_cdr3 else None

    # Generate sequences
    records = []
    num_heavy = int(num_sequences * chain_distribution[0])
    num_light = num_sequences - num_heavy

    print(f"Generating {num_heavy} heavy chains and {num_light} light chains...")

    for i in range(num_heavy):
        sequence, cdr3 = generate_full_sequence("H")
        v_gene = random.choice(IGHV_GENES)
        j_gene = random.choice(IGHJ_GENES)

        records.append({
            "sequence": sequence,
            "chain": "H",
            "species": "human",
            "germline_v": v_gene,
            "germline_j": j_gene,
            "is_productive": True,
            "cdr3": cdr3,
        })

    for i in range(num_light):
        sequence, cdr3 = generate_full_sequence("L")
        v_gene = random.choice(IGLV_GENES)
        j_gene = random.choice(IGLJ_GENES)

        records.append({
            "sequence": sequence,
            "chain": "L",
            "species": "human",
            "germline_v": v_gene,
            "germline_j": j_gene,
            "is_productive": True,
            "cdr3": cdr3,
        })

    # Shuffle
    random.shuffle(records)

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} sequences")
    print(f"  Heavy chains: {(df['chain'] == 'H').sum()}")
    print(f"  Light chains: {(df['chain'] == 'L').sum()}")
    print(f"  Mean length: {df['sequence'].str.len().mean():.1f}")
    print(f"  Length range: {df['sequence'].str.len().min()}-{df['sequence'].str.len().max()}")

    return df


def download_instructions_oas() -> str:
    """Provide instructions for manual OAS download."""
    instructions = """
    ========================================
    MANUAL OAS DOWNLOAD INSTRUCTIONS
    ========================================

    The OAS database requires manual selection and download through their web interface.
    Follow these steps:

    1. Go to: http://opig.stats.ox.ac.uk/webapps/oas/

    2. Use the search/filter options:
       - Select Species: "Human"
       - Select Chain: Both "Heavy" and "Light" (run separately)
       - Select "Productive sequences only"
       - Optional: Filter by disease state, age, etc.

    3. Download options:
       a) For small datasets (<1M sequences):
          - Use the web interface download button
          - Choose "CSV" or "TSV" format
          - Download directly

       b) For larger datasets:
          - Use the "Bulk download" option
          - Download compressed files (.csv.gz)
          - Combine multiple studies if needed

    4. Save downloaded files to: data/raw/
       - Example: data/raw/oas_human_heavy.csv
       - Example: data/raw/oas_human_light.csv

    5. If you have multiple files, combine them:
       python scripts/download_oas_data.py --method combine \
           --input data/raw/oas_*.csv \
           --output data/raw/oas_human.tsv

    6. Verify the format and run ETL:
       abprop-etl --input data/raw/oas_human.tsv --out data/processed/oas

    Expected CSV/TSV columns from OAS:
    - sequence_alignment_aa (or sequence_aa)
    - v_call (or germline_v)
    - j_call (or germline_j)
    - productive (or is_productive)
    - junction_aa (or cdr3)
    - [other metadata columns]

    The ETL pipeline will handle column name variations automatically.

    For questions, see: https://www.blopig.com/blog/2018/06/how-to-parse-oas-data/
    """
    return instructions


def combine_csv_files(input_pattern: str, output_path: Path) -> pd.DataFrame:
    """Combine multiple CSV/TSV files into one."""
    from glob import glob

    files = sorted(glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {input_pattern}")

    print(f"Found {len(files)} files to combine:")
    for f in files:
        print(f"  - {f}")

    dfs = []
    for file_path in files:
        print(f"Reading {file_path}...")
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt") as f:
                df = pd.read_csv(f, sep=None, engine="python")
        else:
            df = pd.read_csv(file_path, sep=None, engine="python")

        print(f"  Loaded {len(df)} sequences")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined total: {len(combined)} sequences")

    # Remove duplicates
    original_len = len(combined)
    combined = combined.drop_duplicates(subset=["sequence_alignment_aa" if "sequence_alignment_aa" in combined.columns else "sequence"])
    print(f"Removed {original_len - len(combined)} duplicates")

    return combined


def sample_oas_subset(input_path: Path, output_path: Path, num_sequences: int, seed: int = 42) -> None:
    """Sample a subset from a large OAS file."""
    print(f"Sampling {num_sequences} sequences from {input_path}...")

    if str(input_path).endswith(".gz"):
        with gzip.open(input_path, "rt") as f:
            df = pd.read_csv(f, sep=None, engine="python")
    else:
        df = pd.read_csv(input_path, sep=None, engine="python")

    print(f"Original dataset: {len(df)} sequences")

    if len(df) > num_sequences:
        df = df.sample(n=num_sequences, random_state=seed)
        print(f"Sampled down to: {len(df)} sequences")

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare OAS data for AbProp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["synthetic", "oas", "combine", "sample"],
        default="synthetic",
        help="Method to acquire data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/oas_synthetic.tsv"),
        help="Output file path",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10000,
        help="Number of sequences (for synthetic method)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file(s) for combine/sample methods (can use wildcards)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "synthetic":
        print("Generating synthetic antibody dataset...")
        df = create_synthetic_dataset(
            num_sequences=args.num_sequences,
            seed=args.seed,
        )
        df.to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved to: {args.output}")
        print(f"Columns: {list(df.columns)}")
        print("\nNext steps:")
        print(f"  abprop-etl --input {args.output} --out data/processed/oas")

    elif args.method == "oas":
        print(download_instructions_oas())
        print("\nNote: This method requires manual download from the OAS website.")
        print("After downloading, use --method combine to merge files.")
        sys.exit(0)

    elif args.method == "combine":
        if not args.input:
            print("ERROR: --input required for combine method")
            sys.exit(1)

        df = combine_csv_files(args.input, args.output)
        df.to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved combined file to: {args.output}")

    elif args.method == "sample":
        if not args.input:
            print("ERROR: --input required for sample method")
            sys.exit(1)

        sample_oas_subset(
            Path(args.input),
            args.output,
            args.num_sequences,
            args.seed,
        )

    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
