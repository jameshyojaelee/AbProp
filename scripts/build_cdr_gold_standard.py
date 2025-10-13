#!/usr/bin/env python3
"""
Build CDR identification gold standard dataset for AbProp.

This script creates a high-quality benchmark dataset for evaluating
CDR vs framework classification with multiple definition schemes.

Usage:
    python scripts/build_cdr_gold_standard.py \
        --output data/processed/cdr_gold_standard \
        --num-sequences 1000 \
        --download-sabdab
"""

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore")


# SAbDab API endpoints
SABDAB_SUMMARY_URL = "http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
SABDAB_SEARCH_URL = "http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/"


# CDR definitions according to different schemes
# Positions are given in the Chothia numbering scheme
# (most commonly used for structural analysis)

CHOTHIA_CDR_DEFINITIONS = {
    "H": {  # Heavy chain
        "CDR1": (26, 32),  # Chothia H1: 26-32
        "CDR2": (52, 56),  # Chothia H2: 52-56
        "CDR3": (95, 102), # Chothia H3: 95-102 (variable length)
    },
    "L": {  # Light chain (kappa/lambda)
        "CDR1": (24, 34),  # Chothia L1: 24-34
        "CDR2": (50, 56),  # Chothia L2: 50-56
        "CDR3": (89, 97),  # Chothia L3: 89-97
    }
}

KABAT_CDR_DEFINITIONS = {
    "H": {
        "CDR1": (31, 35),  # Kabat H1: 31-35 (up to 35B in insertions)
        "CDR2": (50, 65),  # Kabat H2: 50-65
        "CDR3": (95, 102), # Kabat H3: 95-102
    },
    "L": {
        "CDR1": (24, 34),  # Kabat L1: 24-34
        "CDR2": (50, 56),  # Kabat L2: 50-56
        "CDR3": (89, 97),  # Kabat L3: 89-97
    }
}

IMGT_CDR_DEFINITIONS = {
    "H": {
        "CDR1": (27, 38),  # IMGT H1: 27-38 (IMGT positions)
        "CDR2": (56, 65),  # IMGT H2: 56-65
        "CDR3": (105, 117), # IMGT H3: 105-117 (variable)
    },
    "L": {
        "CDR1": (27, 38),  # IMGT L1: 27-38
        "CDR2": (56, 65),  # IMGT L2: 56-65
        "CDR3": (105, 117), # IMGT L3: 105-117
    }
}


def try_import_anarci():
    """Try to import ANARCI, return None if not available."""
    try:
        import anarci
        return anarci
    except ImportError:
        return None


def download_sabdab_summary(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Download SAbDab summary file with all antibody structures.

    Args:
        cache_path: Optional path to cache the downloaded file

    Returns:
        DataFrame with SAbDab summary data
    """
    # Check cache
    if cache_path and cache_path.exists():
        print(f"Loading cached SAbDab summary from {cache_path}")
        return pd.read_csv(cache_path)

    print("Downloading SAbDab summary...")
    print(f"  URL: {SABDAB_SUMMARY_URL}")

    try:
        response = requests.get(SABDAB_SUMMARY_URL, timeout=60)
        response.raise_for_status()

        # Save to temp file
        temp_file = Path("sabdab_summary_temp.tsv")
        with open(temp_file, "wb") as f:
            f.write(response.content)

        # Read TSV
        df = pd.read_csv(temp_file, sep="\t")
        temp_file.unlink()

        print(f"  Downloaded {len(df)} antibody structures")

        # Cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"  Cached to {cache_path}")

        return df

    except Exception as e:
        print(f"ERROR: Failed to download SAbDab: {e}")
        print("\nCreating synthetic CDR gold standard for testing...")
        return create_synthetic_cdr_dataset()


def create_synthetic_cdr_dataset() -> pd.DataFrame:
    """
    Create a synthetic CDR dataset for testing when SAbDab is unavailable.

    Uses realistic framework and CDR patterns.
    """
    print("Generating synthetic CDR gold standard...")

    records = []

    # Heavy chain examples
    for i in range(50):
        # Typical heavy chain structure
        fw1 = "QVQLVQSGAEVKKPGSSVKVSCKAS"  # ~25 AA
        cdr1 = "GYTFTSYYM"  # ~7-10 AA
        fw2 = "HWVRQAPGQGLEW"  # ~13-17 AA
        cdr2 = "MGWINT"  # ~7 AA
        fw3 = "YTGESVKGRFTISADTS"  # Variable
        cdr3 = f"CAR{'DGLTY'[i%5]}{('A'*((i%5)+2))}YYYGMDV"  # Variable CDR3
        fw4 = "WGQGTLVTVSS"  # ~11 AA

        sequence = fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3 + fw4

        # Create CDR mask
        cdr_mask = [0] * len(fw1)
        cdr_mask += [1] * len(cdr1)
        cdr_mask += [0] * len(fw2)
        cdr_mask += [1] * len(cdr2)
        cdr_mask += [0] * len(fw3)
        cdr_mask += [1] * len(cdr3)
        cdr_mask += [0] * len(fw4)

        records.append({
            "pdb_id": f"SYNTH_H{i:03d}",
            "chain": "H",
            "sequence": sequence,
            "length": len(sequence),
            "resolution": 2.0 + (i % 10) / 10,
            "cdr_mask_chothia": json.dumps(cdr_mask),
            "cdr_mask_kabat": json.dumps(cdr_mask),  # Same for synthetic
            "cdr_mask_imgt": json.dumps(cdr_mask),   # Same for synthetic
        })

    # Light chain examples
    for i in range(50):
        fw1 = "DIQMTQSPSSLSASVGDRVTITC"  # ~23 AA
        cdr1 = "RASQSIS"  # ~7-11 AA
        fw2 = "SYLAWYQQKPG"  # ~11-17 AA
        cdr2 = "AASSLQ"  # ~7 AA
        fw3 = "SGVPSRFSGSG"  # Variable
        cdr3 = f"QQ{'SYTS'[i%4]}{('T'*((i%3)+2))}PY"  # Variable CDR3
        fw4 = "TFGQGTKVEIK"  # ~11 AA

        sequence = fw1 + cdr1 + fw2 + cdr2 + fw3 + cdr3 + fw4

        # Create CDR mask
        cdr_mask = [0] * len(fw1)
        cdr_mask += [1] * len(cdr1)
        cdr_mask += [0] * len(fw2)
        cdr_mask += [1] * len(cdr2)
        cdr_mask += [0] * len(fw3)
        cdr_mask += [1] * len(cdr3)
        cdr_mask += [0] * len(fw4)

        records.append({
            "pdb_id": f"SYNTH_L{i:03d}",
            "chain": "L",
            "sequence": sequence,
            "length": len(sequence),
            "resolution": 2.0 + (i % 10) / 10,
            "cdr_mask_chothia": json.dumps(cdr_mask),
            "cdr_mask_kabat": json.dumps(cdr_mask),  # Same for synthetic
            "cdr_mask_imgt": json.dumps(cdr_mask),   # Same for synthetic
        })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df)} synthetic antibody sequences")
    print(f"    Heavy chains: {(df['chain'] == 'H').sum()}")
    print(f"    Light chains: {(df['chain'] == 'L').sum()}")

    return df


def parse_sabdab_summary(df: pd.DataFrame, max_sequences: int = 1000) -> pd.DataFrame:
    """
    Parse and filter SAbDab summary data.

    Args:
        df: SAbDab summary DataFrame
        max_sequences: Maximum number of sequences to keep

    Returns:
        Filtered and cleaned DataFrame
    """
    print(f"\nParsing SAbDab summary...")
    print(f"  Total structures: {len(df)}")

    # Filter for high-quality structures
    if "resolution" in df.columns:
        df = df[df["resolution"] <= 3.0]  # Max 3.0 Å resolution
        print(f"  After resolution filter (<= 3.0 Å): {len(df)}")

    # Filter for X-ray structures (more reliable)
    if "method" in df.columns:
        df = df[df["method"].str.contains("X-RAY", case=False, na=False)]
        print(f"  After method filter (X-RAY): {len(df)}")

    # Remove NMR and models
    if "experimental_method" in df.columns:
        df = df[~df["experimental_method"].str.contains("NMR|MODEL", case=False, na=False)]

    # Ensure we have sequence information
    required_cols = ["pdb", "Hchain", "Lchain"]
    for col in required_cols:
        if col not in df.columns:
            print(f"WARNING: Missing column '{col}'")

    # Sample if we have too many
    if len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=42)
        print(f"  Sampled down to: {len(df)}")

    return df


def number_sequence_simple(sequence: str, chain: str) -> List[Tuple[int, str]]:
    """
    Simple sequence numbering (fallback when ANARCI unavailable).

    Uses a naive approach: assign sequential positions assuming
    standard antibody structure.

    Args:
        sequence: Amino acid sequence
        chain: H or L

    Returns:
        List of (position, residue) tuples
    """
    numbered = []
    position = 1

    for i, residue in enumerate(sequence):
        numbered.append((position, residue))
        position += 1

    return numbered


def number_sequence_with_anarci(sequence: str, chain: str, anarci_module) -> Optional[List[Tuple[int, str]]]:
    """
    Number sequence using ANARCI.

    Args:
        sequence: Amino acid sequence
        chain: H or L
        anarci_module: Imported anarci module

    Returns:
        List of (position, residue) tuples or None if numbering fails
    """
    try:
        # Run ANARCI numbering (using Chothia scheme by default)
        results = anarci_module.anarci(
            [("seq", sequence)],
            scheme="chothia",
            output=False
        )

        if not results or not results[0]:
            return None

        numbering, alignment_details, hit_tables = results
        if not numbering or not numbering[0]:
            return None

        numbered_seq = numbering[0][0][0]  # First sequence, first chain, first scheme
        if numbered_seq is None:
            return None

        # Convert to list of (position, residue) tuples
        result = []
        for (pos, insert), residue in numbered_seq:
            if residue != '-':  # Skip gaps
                result.append((pos, residue))

        return result

    except Exception as e:
        print(f"WARNING: ANARCI numbering failed: {e}")
        return None


def create_cdr_masks(
    numbered_sequence: List[Tuple[int, str]],
    chain: str,
    schemes: List[str] = ["chothia", "kabat", "imgt"]
) -> Dict[str, List[int]]:
    """
    Create CDR masks for different numbering schemes.

    Args:
        numbered_sequence: List of (position, residue) tuples
        chain: H or L
        schemes: List of schemes to generate

    Returns:
        Dictionary mapping scheme name to CDR mask
    """
    masks = {}

    scheme_defs = {
        "chothia": CHOTHIA_CDR_DEFINITIONS,
        "kabat": KABAT_CDR_DEFINITIONS,
        "imgt": IMGT_CDR_DEFINITIONS,
    }

    for scheme in schemes:
        if scheme not in scheme_defs:
            continue

        cdr_def = scheme_defs[scheme].get(chain, {})
        mask = []

        for pos, residue in numbered_sequence:
            is_cdr = False

            # Check if position falls in any CDR
            for cdr_name, (start, end) in cdr_def.items():
                if start <= pos <= end:
                    is_cdr = True
                    break

            mask.append(1 if is_cdr else 0)

        masks[scheme] = mask

    return masks


def extract_sequences_from_sabdab(df: pd.DataFrame) -> List[Dict]:
    """
    Extract sequences and metadata from SAbDab data.

    Args:
        df: Parsed SAbDab DataFrame

    Returns:
        List of sequence dictionaries
    """
    print("\nExtracting sequences from SAbDab...")

    anarci = try_import_anarci()
    if anarci:
        print("  Using ANARCI for accurate numbering")
    else:
        print("  WARNING: ANARCI not available, using fallback numbering")
        print("  Install ANARCI for accurate CDR boundaries: pip install anarci")

    sequences = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing structures"):
        pdb_id = row.get("pdb", f"UNK{idx}")

        # Extract heavy chain if available
        if "Hchain" in row.columns and pd.notna(row["Hchain"]):
            h_seq = row["Hchain"]
            if isinstance(h_seq, str) and len(h_seq) > 0:
                # Number the sequence
                if anarci:
                    numbered = number_sequence_with_anarci(h_seq, "H", anarci)
                else:
                    numbered = number_sequence_simple(h_seq, "H")

                if numbered:
                    # Create CDR masks for different schemes
                    masks = create_cdr_masks(numbered, "H", ["chothia", "kabat", "imgt"])

                    sequences.append({
                        "pdb_id": pdb_id,
                        "chain": "H",
                        "sequence": h_seq,
                        "length": len(h_seq),
                        "resolution": row.get("resolution", None),
                        "cdr_mask_chothia": json.dumps(masks.get("chothia", [])),
                        "cdr_mask_kabat": json.dumps(masks.get("kabat", [])),
                        "cdr_mask_imgt": json.dumps(masks.get("imgt", [])),
                    })

        # Extract light chain if available
        if "Lchain" in row.columns and pd.notna(row["Lchain"]):
            l_seq = row["Lchain"]
            if isinstance(l_seq, str) and len(l_seq) > 0:
                if anarci:
                    numbered = number_sequence_with_anarci(l_seq, "L", anarci)
                else:
                    numbered = number_sequence_simple(l_seq, "L")

                if numbered:
                    masks = create_cdr_masks(numbered, "L", ["chothia", "kabat", "imgt"])

                    sequences.append({
                        "pdb_id": pdb_id,
                        "chain": "L",
                        "sequence": l_seq,
                        "length": len(l_seq),
                        "resolution": row.get("resolution", None),
                        "cdr_mask_chothia": json.dumps(masks.get("chothia", [])),
                        "cdr_mask_kabat": json.dumps(masks.get("kabat", [])),
                        "cdr_mask_imgt": json.dumps(masks.get("imgt", [])),
                    })

    print(f"  Extracted {len(sequences)} sequences")
    return sequences


def create_gold_standard_dataset(
    sequences: List[Dict],
    output_dir: Path,
) -> None:
    """
    Create the final gold standard dataset.

    Args:
        sequences: List of sequence dictionaries
        output_dir: Output directory
    """
    print(f"\nCreating gold standard dataset in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(sequences)

    # Add length if not present
    if "length" not in df.columns:
        df["length"] = df["sequence"].str.len()

    # Add resolution if not present
    if "resolution" not in df.columns:
        df["resolution"] = None

    print(f"  Total sequences: {len(df)}")
    print(f"  Heavy chains: {(df['chain'] == 'H').sum()}")
    print(f"  Light chains: {(df['chain'] == 'L').sum()}")

    # Save full dataset
    df.to_parquet(
        output_dir / "cdr_gold_standard_full.parquet",
        engine="pyarrow",
        index=False,
    )

    # Save partitioned by chain
    df.to_parquet(
        output_dir / "cdr_gold_standard.parquet",
        engine="pyarrow",
        partition_cols=["chain"],
        index=False,
    )

    # Save summary CSV
    summary_cols = ["pdb_id", "chain", "sequence"]
    if "length" in df.columns:
        summary_cols.append("length")
    if "resolution" in df.columns:
        summary_cols.append("resolution")
    summary_df = df[summary_cols].copy()
    summary_df.to_csv(output_dir / "cdr_summary.csv", index=False)

    print(f"\n✓ Gold standard dataset created:")
    print(f"  Output files:")
    print(f"    - {output_dir / 'cdr_gold_standard_full.parquet'} (full)")
    print(f"    - {output_dir / 'cdr_gold_standard.parquet'} (partitioned)")
    print(f"    - {output_dir / 'cdr_summary.csv'} (summary)")


def main():
    parser = argparse.ArgumentParser(
        description="Build CDR identification gold standard dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cdr_gold_standard"),
        help="Output directory",
    )
    parser.add_argument(
        "--download-sabdab",
        action="store_true",
        help="Download fresh data from SAbDab",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/raw/sabdab_summary_cache.csv"),
        help="Cache file for SAbDab summary",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="Maximum number of sequences to include",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic dataset (skip SAbDab download)",
    )

    args = parser.parse_args()

    print("="*70)
    print("CDR Identification Gold Standard Dataset")
    print("="*70)

    # Check for ANARCI
    anarci = try_import_anarci()
    if not anarci:
        print("\nWARNING: ANARCI not installed")
        print("For accurate CDR boundary detection, install ANARCI:")
        print("  pip install anarci")
        print("Continuing with fallback numbering...\n")

    # Download or create data
    if args.synthetic:
        df = create_synthetic_cdr_dataset()
        sequences = df.to_dict('records')
    else:
        if args.download_sabdab or not args.cache.exists():
            df = download_sabdab_summary(cache_path=args.cache)
        else:
            df = download_sabdab_summary(cache_path=args.cache)

        df = parse_sabdab_summary(df, max_sequences=args.num_sequences)
        sequences = extract_sequences_from_sabdab(df)

    # Create gold standard dataset
    create_gold_standard_dataset(sequences, args.output)

    print("\n" + "="*70)
    print("✓ Gold standard creation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
