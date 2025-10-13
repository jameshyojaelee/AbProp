#!/usr/bin/env python3
"""
Fetch antibody sequence data from AIRR Data Commons API.

This script queries the AIRR Data Commons API to download real antibody sequences.
AIRR Data Commons provides standardized access to multiple repositories.

Usage:
    python scripts/fetch_airr_data.py --species human --sequences 50000 --output data/raw/airr_human.tsv
    python scripts/fetch_airr_data.py --repertoire-id <id> --output data/raw/airr_repertoire.tsv
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


# AIRR Data Commons API endpoint
AIRR_API_BASE = "https://airr-api.ireceptor.org/airr/v1"


def query_repertoires(
    species: Optional[str] = None,
    study_id: Optional[str] = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Query available repertoires from AIRR Data Commons.

    Args:
        species: Filter by species (e.g., "human", "mouse")
        study_id: Filter by specific study
        limit: Maximum number of repertoires to return

    Returns:
        List of repertoire metadata dictionaries
    """
    url = f"{AIRR_API_BASE}/repertoire"

    # Build query
    query = {"filters": {}}

    if species:
        query["filters"]["subject.species.label"] = {"value": species.lower()}

    if study_id:
        query["filters"]["study.study_id"] = {"value": study_id}

    query["size"] = limit

    print(f"Querying repertoires from AIRR Data Commons...")
    print(f"  API: {url}")
    print(f"  Filters: {query['filters']}")

    try:
        response = requests.post(url, json=query, timeout=30)
        response.raise_for_status()
        data = response.json()

        repertoires = data.get("Repertoire", [])
        print(f"  Found {len(repertoires)} repertoires")

        return repertoires

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to query repertoires: {e}")
        return []


def query_rearrangements(
    repertoire_id: str,
    limit: int = 1000,
    productive_only: bool = True,
    locus: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query rearrangement sequences for a specific repertoire.

    Args:
        repertoire_id: Repertoire identifier
        limit: Number of sequences to fetch
        productive_only: Only return productive sequences
        locus: Filter by locus (e.g., "IGH", "IGL", "IGK")

    Returns:
        DataFrame with sequence data
    """
    url = f"{AIRR_API_BASE}/rearrangement"

    # Build query
    query = {
        "filters": {
            "repertoire_id": {"value": repertoire_id}
        },
        "size": limit,
        "fields": [
            "sequence_aa",
            "sequence",
            "v_call",
            "d_call",
            "j_call",
            "productive",
            "locus",
            "junction_aa",
            "cdr3_aa",
            "repertoire_id",
        ]
    }

    if productive_only:
        query["filters"]["productive"] = {"value": True}

    if locus:
        query["filters"]["locus"] = {"value": locus}

    try:
        response = requests.post(url, json=query, timeout=60)
        response.raise_for_status()
        data = response.json()

        rearrangements = data.get("Rearrangement", [])

        if not rearrangements:
            return pd.DataFrame()

        df = pd.DataFrame(rearrangements)
        return df

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Failed to fetch sequences: {e}")
        return pd.DataFrame()


def fetch_sequences_multirepet(
    species: str,
    target_sequences: int,
    productive_only: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch sequences from multiple repertoires until target is reached.

    Args:
        species: Species to filter by
        target_sequences: Target number of sequences
        productive_only: Only productive sequences
        output_path: If provided, save intermediate results

    Returns:
        Combined DataFrame
    """
    print(f"\nFetching {target_sequences} {species} antibody sequences from AIRR Data Commons...")

    # First, get available repertoires
    repertoires = query_repertoires(species=species, limit=100)

    if not repertoires:
        print("ERROR: No repertoires found. The AIRR API may be unavailable.")
        print("Consider using the synthetic data method instead:")
        print("  python scripts/download_oas_data.py --method synthetic --num-sequences 10000")
        return pd.DataFrame()

    print(f"\nFound {len(repertoires)} {species} repertoires")
    print("Fetching sequences from each repertoire...")

    all_sequences = []
    sequences_per_repertoire = max(100, target_sequences // len(repertoires))

    for i, rep in enumerate(tqdm(repertoires, desc="Repertoires")):
        rep_id = rep.get("repertoire_id")
        if not rep_id:
            continue

        # Fetch sequences for this repertoire
        df = query_rearrangements(
            repertoire_id=rep_id,
            limit=sequences_per_repertoire,
            productive_only=productive_only,
        )

        if not df.empty:
            all_sequences.append(df)
            total_so_far = sum(len(d) for d in all_sequences)
            print(f"  Repertoire {i+1}/{len(repertoires)}: {len(df)} sequences (total: {total_so_far})")

            if total_so_far >= target_sequences:
                print(f"\n✓ Reached target of {target_sequences} sequences")
                break

        # Be nice to the API
        time.sleep(0.5)

    if not all_sequences:
        print("\nWARNING: No sequences retrieved from AIRR API")
        return pd.DataFrame()

    # Combine all sequences
    combined = pd.concat(all_sequences, ignore_index=True)
    print(f"\nTotal sequences retrieved: {len(combined)}")

    # Sample down if we have too many
    if len(combined) > target_sequences:
        combined = combined.sample(n=target_sequences, random_state=42)
        print(f"Sampled down to: {len(combined)}")

    return combined


def format_for_abprop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format AIRR data to match AbProp ETL expectations.

    Expected columns:
    - sequence: amino acid sequence
    - chain: H or L
    - species: species name
    - germline_v: V gene
    - germline_j: J gene
    - is_productive: boolean
    - cdr3: CDR3 sequence
    """
    # Map AIRR fields to AbProp fields
    formatted = pd.DataFrame()

    # Sequence
    if "sequence_aa" in df.columns:
        formatted["sequence"] = df["sequence_aa"]
    elif "sequence" in df.columns:
        formatted["sequence"] = df["sequence"]
    else:
        raise ValueError("No sequence column found")

    # Determine chain from locus
    def extract_chain(locus):
        if pd.isna(locus):
            return "H"  # default
        locus = str(locus).upper()
        if "IGH" in locus:
            return "H"
        elif "IGL" in locus or "IGK" in locus:
            return "L"
        return "H"

    if "locus" in df.columns:
        formatted["chain"] = df["locus"].apply(extract_chain)
    else:
        formatted["chain"] = "H"  # default to heavy

    # Species (will be standardized by ETL)
    formatted["species"] = "human"  # AIRR query already filtered by species

    # Germline genes
    if "v_call" in df.columns:
        formatted["germline_v"] = df["v_call"].fillna("UNKNOWN")
    else:
        formatted["germline_v"] = "UNKNOWN"

    if "j_call" in df.columns:
        formatted["germline_j"] = df["j_call"].fillna("UNKNOWN")
    else:
        formatted["germline_j"] = "UNKNOWN"

    # Productive status
    if "productive" in df.columns:
        formatted["is_productive"] = df["productive"].fillna(True)
    else:
        formatted["is_productive"] = True

    # CDR3
    if "cdr3_aa" in df.columns:
        formatted["cdr3"] = df["cdr3_aa"].fillna("")
    elif "junction_aa" in df.columns:
        formatted["cdr3"] = df["junction_aa"].fillna("")
    else:
        formatted["cdr3"] = ""

    # Remove rows with empty sequences
    formatted = formatted[formatted["sequence"].notna()]
    formatted = formatted[formatted["sequence"].str.len() > 0]

    # Remove non-productive if specified
    formatted = formatted[formatted["is_productive"] == True]

    print(f"\nFormatted {len(formatted)} sequences for AbProp")
    print(f"  Heavy chains: {(formatted['chain'] == 'H').sum()}")
    print(f"  Light chains: {(formatted['chain'] == 'L').sum()}")

    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Fetch antibody sequences from AIRR Data Commons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--species",
        default="human",
        help="Species to fetch (e.g., human, mouse)",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=50000,
        help="Target number of sequences to fetch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/airr_human.tsv"),
        help="Output file path",
    )
    parser.add_argument(
        "--productive-only",
        action="store_true",
        default=True,
        help="Only fetch productive sequences",
    )
    parser.add_argument(
        "--list-repertoires",
        action="store_true",
        help="List available repertoires and exit",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.list_repertoires:
        print("Querying available repertoires...")
        repertoires = query_repertoires(species=args.species, limit=50)

        if repertoires:
            print(f"\nFound {len(repertoires)} {args.species} repertoires:\n")
            for i, rep in enumerate(repertoires[:20], 1):
                rep_id = rep.get("repertoire_id", "N/A")
                study = rep.get("study", {})
                study_title = study.get("study_title", "N/A")
                print(f"{i}. {rep_id}")
                print(f"   Study: {study_title}")
                print()

            if len(repertoires) > 20:
                print(f"... and {len(repertoires) - 20} more")
        else:
            print("No repertoires found or API unavailable.")

        return

    # Fetch sequences
    try:
        df = fetch_sequences_multirepet(
            species=args.species,
            target_sequences=args.sequences,
            productive_only=args.productive_only,
            output_path=args.output,
        )

        if df.empty:
            print("\n" + "="*60)
            print("FALLBACK: AIRR API did not return data.")
            print("Generating synthetic dataset instead...")
            print("="*60 + "\n")

            # Fallback to synthetic data
            import sys
            import subprocess

            result = subprocess.run([
                sys.executable,
                "scripts/download_oas_data.py",
                "--method", "synthetic",
                "--num-sequences", str(args.sequences),
                "--output", str(args.output),
            ])

            if result.returncode == 0:
                print(f"\n✓ Created synthetic dataset at {args.output}")
            else:
                print("\nERROR: Failed to create synthetic dataset")
                sys.exit(1)

            return

        # Format for AbProp
        formatted = format_for_abprop(df)

        # Save
        formatted.to_csv(args.output, sep="\t", index=False)
        print(f"\n✓ Saved {len(formatted)} sequences to: {args.output}")
        print(f"\nColumns: {list(formatted.columns)}")
        print("\nNext steps:")
        print(f"  abprop-etl --input {args.output} --out data/processed/oas")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFalling back to synthetic data generation...")
        import sys
        import subprocess

        result = subprocess.run([
            sys.executable,
            "scripts/download_oas_data.py",
            "--method", "synthetic",
            "--num-sequences", str(args.sequences),
            "--output", str(args.output),
        ])

        if result.returncode != 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
