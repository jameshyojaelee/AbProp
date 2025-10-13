#!/usr/bin/env python3
"""
Compare CDR definitions across different numbering schemes.

This script analyzes the differences between Chothia, Kabat, and IMGT
CDR definitions and validates the gold standard dataset.

Usage:
    python scripts/validate_cdr_schemes.py \
        --dataset data/processed/cdr_gold_standard/cdr_gold_standard_full.parquet \
        --output outputs/cdr_scheme_comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, skipping plots")

import numpy as np
import pandas as pd
from collections import Counter


def compare_cdr_masks(
    mask1: List[int],
    mask2: List[int],
    name1: str = "Scheme1",
    name2: str = "Scheme2"
) -> Dict:
    """
    Compare two CDR masks and compute agreement metrics.

    Args:
        mask1: First CDR mask
        mask2: Second CDR mask
        name1: Name of first scheme
        name2: Name of second scheme

    Returns:
        Dictionary with comparison metrics
    """
    # Ensure same length
    min_len = min(len(mask1), len(mask2))
    mask1 = mask1[:min_len]
    mask2 = mask2[:min_len]

    # Convert to numpy for easier computation
    m1 = np.array(mask1)
    m2 = np.array(mask2)

    # Compute metrics
    total = len(m1)
    agreement = (m1 == m2).sum()
    disagreement = (m1 != m2).sum()

    # CDR-specific metrics
    cdr1 = m1.sum()  # Number of CDR residues in scheme 1
    cdr2 = m2.sum()  # Number of CDR residues in scheme 2

    both_cdr = ((m1 == 1) & (m2 == 1)).sum()  # Both call it CDR
    both_fw = ((m1 == 0) & (m2 == 0)).sum()   # Both call it framework

    only_1_cdr = ((m1 == 1) & (m2 == 0)).sum()  # Only scheme 1 calls it CDR
    only_2_cdr = ((m1 == 0) & (m2 == 1)).sum()  # Only scheme 2 calls it CDR

    return {
        "scheme1": name1,
        "scheme2": name2,
        "total_residues": total,
        "agreement": agreement,
        "disagreement": disagreement,
        "agreement_pct": 100 * agreement / total if total > 0 else 0,
        "cdr_residues_scheme1": cdr1,
        "cdr_residues_scheme2": cdr2,
        "both_cdr": both_cdr,
        "both_framework": both_fw,
        "only_scheme1_cdr": only_1_cdr,
        "only_scheme2_cdr": only_2_cdr,
    }


def analyze_cdr_distribution(df: pd.DataFrame, scheme: str = "chothia") -> Dict:
    """
    Analyze CDR distribution statistics.

    Args:
        df: DataFrame with CDR annotations
        scheme: Scheme to analyze (chothia, kabat, imgt)

    Returns:
        Dictionary with statistics
    """
    mask_col = f"cdr_mask_{scheme}"
    if mask_col not in df.columns:
        return {}

    stats = {
        "scheme": scheme,
        "total_sequences": len(df),
        "chains": {},
    }

    for chain in ["H", "L"]:
        chain_df = df[df["chain"] == chain]
        if len(chain_df) == 0:
            continue

        cdr_fractions = []
        cdr_counts = []

        for idx, row in chain_df.iterrows():
            mask = json.loads(row[mask_col])
            cdr_count = sum(mask)
            cdr_fraction = cdr_count / len(mask) if len(mask) > 0 else 0

            cdr_fractions.append(cdr_fraction)
            cdr_counts.append(cdr_count)

        stats["chains"][chain] = {
            "n_sequences": len(chain_df),
            "mean_cdr_fraction": np.mean(cdr_fractions),
            "std_cdr_fraction": np.std(cdr_fractions),
            "mean_cdr_count": np.mean(cdr_counts),
            "std_cdr_count": np.std(cdr_counts),
            "min_cdr_count": min(cdr_counts) if cdr_counts else 0,
            "max_cdr_count": max(cdr_counts) if cdr_counts else 0,
        }

    return stats


def plot_cdr_comparison(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    Create visualization comparing CDR definitions.

    Args:
        df: DataFrame with CDR annotations
        output_dir: Output directory for plots
    """
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = ["chothia", "kabat", "imgt"]

    # Plot 1: CDR fraction distribution per scheme
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for chain_idx, chain in enumerate(["H", "L"]):
        ax = axes[chain_idx]
        chain_df = df[df["chain"] == chain]

        for scheme in schemes:
            mask_col = f"cdr_mask_{scheme}"
            if mask_col not in chain_df.columns:
                continue

            cdr_fractions = []
            for idx, row in chain_df.iterrows():
                mask = json.loads(row[mask_col])
                cdr_fraction = sum(mask) / len(mask) if len(mask) > 0 else 0
                cdr_fractions.append(cdr_fraction)

            ax.hist(cdr_fractions, bins=20, alpha=0.5, label=scheme.upper())

        ax.set_xlabel("CDR Fraction")
        ax.set_ylabel("Count")
        ax.set_title(f"CDR Fraction Distribution ({chain} chain)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "cdr_fraction_distribution.png", dpi=150)
    plt.close(fig)

    # Plot 2: Pairwise agreement heatmap
    comparisons = []
    for i, scheme1 in enumerate(schemes):
        for scheme2 in schemes[i+1:]:
            agreements = []

            for idx, row in df.iterrows():
                mask1 = json.loads(row[f"cdr_mask_{scheme1}"])
                mask2 = json.loads(row[f"cdr_mask_{scheme2}"])

                comp = compare_cdr_masks(mask1, mask2, scheme1, scheme2)
                agreements.append(comp["agreement_pct"])

            comparisons.append({
                "scheme1": scheme1,
                "scheme2": scheme2,
                "mean_agreement": np.mean(agreements),
                "std_agreement": np.std(agreements),
            })

    # Create agreement matrix
    n_schemes = len(schemes)
    agreement_matrix = np.eye(n_schemes) * 100  # Diagonal is 100%

    for comp in comparisons:
        i = schemes.index(comp["scheme1"])
        j = schemes.index(comp["scheme2"])
        agreement_matrix[i, j] = comp["mean_agreement"]
        agreement_matrix[j, i] = comp["mean_agreement"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(agreement_matrix, cmap="RdYlGn", vmin=80, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(n_schemes))
    ax.set_yticks(np.arange(n_schemes))
    ax.set_xticklabels([s.upper() for s in schemes])
    ax.set_yticklabels([s.upper() for s in schemes])

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n_schemes):
        for j in range(n_schemes):
            text = ax.text(j, i, f"{agreement_matrix[i, j]:.1f}%",
                          ha="center", va="center", color="black", fontsize=12)

    ax.set_title("CDR Definition Agreement Between Schemes", fontsize=14, pad=20)
    fig.colorbar(im, ax=ax, label="Agreement (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "scheme_agreement_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\n✓ Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and compare CDR numbering schemes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/cdr_gold_standard/cdr_gold_standard_full.parquet"),
        help="Gold standard dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cdr_scheme_comparison"),
        help="Output directory",
    )

    args = parser.parse_args()

    print("="*70)
    print("CDR Numbering Scheme Comparison")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    df = pd.read_parquet(args.dataset)
    print(f"  Loaded {len(df)} sequences")
    print(f"  Heavy chains: {(df['chain'] == 'H').sum()}")
    print(f"  Light chains: {(df['chain'] == 'L').sum()}")

    # Analyze each scheme
    print("\n" + "="*70)
    print("CDR Distribution Statistics")
    print("="*70)

    for scheme in ["chothia", "kabat", "imgt"]:
        stats = analyze_cdr_distribution(df, scheme)
        if not stats:
            continue

        print(f"\n{scheme.upper()} Scheme:")
        for chain, chain_stats in stats.get("chains", {}).items():
            print(f"  {chain} chain:")
            print(f"    Mean CDR fraction: {chain_stats['mean_cdr_fraction']:.2%} ± {chain_stats['std_cdr_fraction']:.2%}")
            print(f"    Mean CDR count: {chain_stats['mean_cdr_count']:.1f} ± {chain_stats['std_cdr_count']:.1f}")
            print(f"    Range: {chain_stats['min_cdr_count']}-{chain_stats['max_cdr_count']} residues")

    # Pairwise comparisons
    print("\n" + "="*70)
    print("Pairwise Scheme Comparisons")
    print("="*70)

    schemes = ["chothia", "kabat", "imgt"]
    for i, scheme1 in enumerate(schemes):
        for scheme2 in schemes[i+1:]:
            print(f"\n{scheme1.upper()} vs {scheme2.upper()}:")

            agreements = []
            for chain in ["H", "L"]:
                chain_df = df[df["chain"] == chain]
                chain_agreements = []

                for idx, row in chain_df.iterrows():
                    mask1 = json.loads(row[f"cdr_mask_{scheme1}"])
                    mask2 = json.loads(row[f"cdr_mask_{scheme2}"])

                    comp = compare_cdr_masks(mask1, mask2, scheme1, scheme2)
                    chain_agreements.append(comp["agreement_pct"])

                if chain_agreements:
                    mean_agree = np.mean(chain_agreements)
                    std_agree = np.std(chain_agreements)
                    print(f"  {chain} chain agreement: {mean_agree:.2f}% ± {std_agree:.2f}%")
                    agreements.extend(chain_agreements)

            if agreements:
                overall_mean = np.mean(agreements)
                overall_std = np.std(agreements)
                print(f"  Overall agreement: {overall_mean:.2f}% ± {overall_std:.2f}%")

    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)

    plot_cdr_comparison(df, args.output)

    # Save summary report
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("CDR Numbering Scheme Comparison Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Total sequences: {len(df)}\n")
        f.write(f"Heavy chains: {(df['chain'] == 'H').sum()}\n")
        f.write(f"Light chains: {(df['chain'] == 'L').sum()}\n\n")

        for scheme in schemes:
            stats = analyze_cdr_distribution(df, scheme)
            f.write(f"\n{scheme.upper()} Scheme Statistics:\n")
            f.write("-"*40 + "\n")
            for chain, chain_stats in stats.get("chains", {}).items():
                f.write(f"{chain} chain:\n")
                f.write(f"  Mean CDR fraction: {chain_stats['mean_cdr_fraction']:.2%}\n")
                f.write(f"  Mean CDR count: {chain_stats['mean_cdr_count']:.1f}\n\n")

    print(f"\n✓ Report saved to {report_path}")

    print("\n" + "="*70)
    print("✓ Validation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
