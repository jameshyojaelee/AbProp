#!/usr/bin/env python3
"""
Fetch real antibody sequence data from multiple sources.

This script attempts to download real antibody sequences from:
1. SAbDab metadata (already downloaded)
2. PDB REST API for actual sequences
3. OAS bulk data (if available)
4. UniProt for therapeutic antibodies

Usage:
    python scripts/fetch_real_antibody_data.py --source sabdab --output data/raw/sabdab_sequences.csv
    python scripts/fetch_real_antibody_data.py --source pdb --output data/raw/pdb_antibodies.csv
    python scripts/fetch_real_antibody_data.py --source uniprot --output data/raw/therapeutic_antibodies.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session() -> requests.Session:
    """Create requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_pdb_sequence(pdb_id: str, chain: str, session: requests.Session) -> Optional[str]:
    """Fetch sequence for a PDB ID and chain from PDB REST API."""
    try:
        # Use RCSB PDB Data API
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{chain}"
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Try multiple possible sequence fields
            for field in ["entity_poly.pdbx_seq_one_letter_code", "sequence"]:
                if field in data:
                    return data[field]
            # Try nested structure
            if "entity_poly" in data and "pdbx_seq_one_letter_code" in data["entity_poly"]:
                return data["entity_poly"]["pdbx_seq_one_letter_code"]

        # Fallback: try FASTA endpoint
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
        response = session.get(fasta_url, timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            # Parse FASTA format
            for i, line in enumerate(lines):
                if line.startswith(">") and f"_{chain}" in line or f"|{chain}" in line:
                    # Next line(s) contain sequence
                    seq_lines = []
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith(">"):
                            break
                        seq_lines.append(lines[j].strip())
                    if seq_lines:
                        return "".join(seq_lines)

        return None
    except Exception as e:
        print(f"Warning: Failed to fetch {pdb_id}:{chain}: {e}")
        return None


def process_sabdab_metadata(metadata_path: Path, output_path: Path, max_entries: Optional[int] = None) -> pd.DataFrame:
    """Process SAbDab metadata and fetch sequences from PDB."""
    print(f"Reading SAbDab metadata from {metadata_path}")
    df = pd.read_csv(metadata_path, sep="\t")
    print(f"Found {len(df)} entries")

    if max_entries:
        df = df.head(max_entries)
        print(f"Limited to {max_entries} entries for testing")

    # Filter for reasonable quality structures
    df = df[df["method"].str.contains("DIFFRACTION|ELECTRON MICROSCOPY", na=False, case=False)]
    df = df[df["resolution"].notna()]
    # Convert resolution to numeric, coercing errors to NaN
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df = df[df["resolution"].notna()]
    df = df[df["resolution"] < 4.0]  # Only high-quality structures

    print(f"After filtering: {len(df)} high-quality structures")

    session = create_session()
    sequences = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(df)}...")

        pdb_id = row["pdb"]
        h_chain = row["Hchain"]
        l_chain = row["Lchain"]

        # Process heavy chain
        if pd.notna(h_chain) and h_chain and h_chain.upper() != "NA":
            h_seq = fetch_pdb_sequence(pdb_id, h_chain, session)
            if h_seq:
                sequences.append({
                    "pdb_id": pdb_id,
                    "chain": "H",
                    "chain_id": h_chain,
                    "sequence": h_seq,
                    "species": row.get("heavy_species", "unknown"),
                    "resolution": row["resolution"],
                    "method": row["method"],
                    "date": row["date"],
                    "scfv": row.get("scfv", False),
                    "engineered": row.get("engineered", False),
                    "antigen_type": row.get("antigen_type", "unknown"),
                })

        # Process light chain
        if pd.notna(l_chain) and l_chain and l_chain.upper() != "NA":
            l_seq = fetch_pdb_sequence(pdb_id, l_chain, session)
            if l_seq:
                sequences.append({
                    "pdb_id": pdb_id,
                    "chain": "L",
                    "chain_id": l_chain,
                    "sequence": l_seq,
                    "species": row.get("light_species", "unknown"),
                    "resolution": row["resolution"],
                    "method": row["method"],
                    "date": row["date"],
                    "scfv": row.get("scfv", False),
                    "engineered": row.get("engineered", False),
                    "antigen_type": row.get("antigen_type", "unknown"),
                })

        # Rate limiting
        time.sleep(0.1)

    result_df = pd.DataFrame(sequences)
    print(f"\nSuccessfully fetched {len(result_df)} sequences")

    # Clean up sequences
    result_df["sequence"] = result_df["sequence"].str.replace("\\s+", "", regex=True).str.upper()
    result_df = result_df[result_df["sequence"].str.len() > 50]  # Minimum length
    result_df = result_df[result_df["sequence"].str.len() < 500]  # Maximum length

    # Normalize species
    result_df["species"] = result_df["species"].str.lower().str.strip()
    result_df["species"] = result_df["species"].replace({
        "homo sapiens": "human",
        "mus musculus": "mouse",
        "": "unknown",
    })

    # Add minimal germline info (would need ANARCI for real annotation)
    result_df["germline_v"] = "UNKNOWN"
    result_df["germline_j"] = "UNKNOWN"
    result_df["is_productive"] = True
    result_df["cdr3"] = None

    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return result_df


def fetch_therapeutic_antibodies_from_uniprot(output_path: Path, max_entries: int = 100) -> pd.DataFrame:
    """Fetch therapeutic antibody sequences from UniProt."""
    print("Fetching therapeutic antibodies from UniProt...")

    # Simpler query for immunoglobulin chains - default fields work better
    query = "immunoglobulin AND organism_id:9606"
    url = "https://rest.uniprot.org/uniprotkb/search"

    session = create_session()
    params = {
        "query": query,
        "format": "tsv",
        "size": max_entries,
    }

    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text), sep="\t")

        print(f"Found {len(df)} entries from UniProt")

        # Process into our format
        sequences = []
        for _, row in df.iterrows():
            seq = row.get("Sequence", "")
            if len(seq) < 50 or len(seq) > 500:
                continue

            # Try to infer chain type from protein name
            name = str(row.get("Protein names", "")).lower()
            if "heavy" in name or "vhh" in name:
                chain = "H"
            elif "light" in name or "kappa" in name or "lambda" in name:
                chain = "L"
            else:
                chain = "H"  # Default to heavy

            sequences.append({
                "uniprot_id": row.get("Entry", ""),
                "gene_name": row.get("Gene Names", ""),
                "protein_name": row.get("Protein names", ""),
                "sequence": seq,
                "chain": chain,
                "species": "human",
                "length": len(seq),
                "source": "uniprot",
                "germline_v": "UNKNOWN",
                "germline_j": "UNKNOWN",
                "is_productive": True,
                "cdr3": None,
            })

        result_df = pd.DataFrame(sequences)
        result_df.to_csv(output_path, index=False)
        print(f"Saved {len(result_df)} therapeutic sequences to {output_path}")

        return result_df

    except Exception as e:
        print(f"ERROR: Failed to fetch from UniProt: {e}")
        return pd.DataFrame()


def fetch_oas_bulk_data(output_path: Path, study_id: str = "Briney_2019", max_sequences: int = 10000) -> pd.DataFrame:
    """
    Fetch OAS bulk data from their S3 bucket or download page.
    Note: This is a placeholder - OAS data is typically very large and requires
    manual download from their website.
    """
    print(f"OAS bulk data download typically requires manual download from:")
    print("http://opig.stats.ox.ac.uk/webapps/oas/oas")
    print("\nFor automated access, you can try:")
    print("1. Visit the OAS website and select a study")
    print("2. Download the CSV/TSV file for that study")
    print("3. Place it in data/raw/")
    print("\nExample studies:")
    print("- Briney_2019: Healthy adult repertoires")
    print("- Jaffe_2022: COVID-19 antibodies")
    print("- DeKosky_2016: Human antibody repertoires")

    # For now, return empty DataFrame with message
    print(f"\nCreating placeholder file at {output_path}")
    df = pd.DataFrame({
        "message": ["Please download OAS data manually from the website"],
        "url": ["http://opig.stats.ox.ac.uk/webapps/oas/oas"],
    })
    df.to_csv(output_path, index=False)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch real antibody sequence data")
    parser.add_argument(
        "--source",
        choices=["sabdab", "uniprot", "oas"],
        required=True,
        help="Data source to fetch from",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to fetch (for testing)",
    )
    parser.add_argument(
        "--sabdab-metadata",
        type=Path,
        default=Path("data/raw/sabdab_summary.tsv"),
        help="Path to SAbDab metadata TSV",
    )

    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "sabdab":
        if not args.sabdab_metadata.exists():
            print(f"ERROR: SAbDab metadata not found at {args.sabdab_metadata}")
            print("Please download it first with:")
            print('curl -L "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all" -o data/raw/sabdab_summary.tsv')
            sys.exit(1)

        process_sabdab_metadata(args.sabdab_metadata, args.output, args.max_entries)

    elif args.source == "uniprot":
        fetch_therapeutic_antibodies_from_uniprot(args.output, args.max_entries or 100)

    elif args.source == "oas":
        fetch_oas_bulk_data(args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
