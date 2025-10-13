#!/usr/bin/env python
"""Example: Using the cross-validation framework.

This script demonstrates how to use AbProp's clonotype-aware k-fold
cross-validation framework for robust model evaluation.
"""

from pathlib import Path

import pandas as pd

from abprop.data.cross_validation import (
    ClonotypeAwareKFold,
    GroupKFoldDataset,
    generate_cv_folds,
    stratified_cv_summary,
)


def example_1_basic_splitting():
    """Example 1: Basic clonotype-aware k-fold splitting."""
    print("=" * 80)
    print("Example 1: Basic Clonotype-Aware K-Fold Splitting")
    print("=" * 80)

    # Create synthetic dataset with clonotypes
    data = {
        "sequence": [
            "QVQLVQSGAEVKKPG",
            "QVQLVQSGAEVKKPG",  # Same as above (same clonotype)
            "DIQMTQSPSSLSASV",
            "EVQLVESGGGLVQPG",
            "EVQLVESGGGLVQPG",  # Same as above (same clonotype)
        ] * 10,  # 50 sequences total
        "chain": ["H", "H", "L", "H", "H"] * 10,
        "species": ["human"] * 50,
        "cdr3": ["CDR3A", "CDR3A", "CDR3B", "CDR3C", "CDR3C"] * 10,
    }
    df = pd.DataFrame(data)

    print(f"Dataset: {len(df)} sequences")
    print(f"Unique clonotypes: {df.groupby(['chain', 'cdr3']).ngroups}")
    print()

    # Perform 3-fold CV
    splitter = ClonotypeAwareKFold(n_splits=3, shuffle=True, seed=42)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df)):
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_idx)} sequences")
        print(f"  Val:   {len(val_idx)} sequences")

        # Verify no clonotype leakage
        if len(val_idx) > 0:
            train_clonotypes = set(
                df.iloc[list(train_idx)].apply(lambda r: f"{r['chain']}|{r['cdr3']}", axis=1)
            )
            val_clonotypes = set(
                df.iloc[list(val_idx)].apply(lambda r: f"{r['chain']}|{r['cdr3']}", axis=1)
            )
            overlap = train_clonotypes & val_clonotypes
            print(f"  Clonotype overlap: {len(overlap)} ✓ (should be 0)")
        else:
            print(f"  Note: Empty validation set (can happen with very few clonotypes)")
        print()


def example_2_fold_statistics():
    """Example 2: Generate and inspect fold statistics."""
    print("=" * 80)
    print("Example 2: Fold Statistics and Stratification")
    print("=" * 80)

    # Create more realistic synthetic dataset
    data = {
        "sequence": (
            ["QVQLVQSGAEVKKPG"] * 20
            + ["DIQMTQSPSSLSASV"] * 20
            + ["EVQLVESGGGLVQPG"] * 20
        ),
        "chain": ["H"] * 40 + ["L"] * 20,
        "species": ["human"] * 30 + ["mouse"] * 30,
        "cdr3": [f"CDR3_{i}" for i in range(60)],
        "length": [150] * 60,
    }
    df = pd.DataFrame(data)

    print(f"Dataset: {len(df)} sequences")
    print(f"  Species: {df['species'].value_counts().to_dict()}")
    print(f"  Chains: {df['chain'].value_counts().to_dict()}")
    print()

    # Generate CV folds with test set holdout
    folds = generate_cv_folds(df, n_splits=5, test_ratio=0.2, seed=42)

    print(f"Generated {len(folds)} folds with 20% test holdout")
    print()

    # Get fold summary statistics
    summary = stratified_cv_summary(df.iloc[: len(df) - 12], folds)  # Exclude test set
    print("Fold Statistics:")
    print(summary.to_string(index=False))


def example_3_dataset_usage():
    """Example 3: Using GroupKFoldDataset with PyTorch."""
    print("\n" + "=" * 80)
    print("Example 3: GroupKFoldDataset for Training")
    print("=" * 80)

    # This example shows how you would use the dataset in practice
    # Note: Requires actual parquet data to run

    print("Usage example (requires actual data):")
    print()
    print("```python")
    print("from torch.utils.data import DataLoader")
    print("from abprop.data import GroupKFoldDataset, build_collate_fn")
    print()
    print("# Create fold-specific dataset")
    print("train_dataset = GroupKFoldDataset(")
    print('    parquet_dir="data/processed/oas",')
    print("    fold_idx=0,")
    print("    n_splits=5,")
    print('    split_type="train",')
    print("    seed=42,")
    print(")")
    print()
    print("# Create dataloader")
    print("collate_fn = build_collate_fn(generate_mlm=True)")
    print("train_loader = DataLoader(")
    print("    train_dataset,")
    print("    batch_size=32,")
    print("    collate_fn=collate_fn,")
    print("    shuffle=True,")
    print(")")
    print()
    print("# Train model")
    print("for batch in train_loader:")
    print("    # Your training loop here")
    print("    pass")
    print("```")


def example_4_ensemble_inference():
    """Example 4: Ensemble inference from CV checkpoints."""
    print("\n" + "=" * 80)
    print("Example 4: Ensemble Inference")
    print("=" * 80)

    print("Deploy ensemble inference server from CV checkpoints:")
    print()
    print("```python")
    print("from pathlib import Path")
    print("from abprop.server.app import create_ensemble_app_from_cv")
    print()
    print("# Create ensemble app from CV directory")
    print("app = create_ensemble_app_from_cv(")
    print('    cv_dir=Path("outputs/cv_5fold"),')
    print('    model_config=Path("configs/model.yaml"),')
    print('    device="cuda",')
    print(")")
    print()
    print("# Run with: uvicorn app:app --host 0.0.0.0 --port 8000")
    print("```")
    print()
    print("API usage:")
    print()
    print("```python")
    print("import requests")
    print()
    print("response = requests.post(")
    print('    "http://localhost:8000/score/liabilities",')
    print("    json={")
    print('        "sequences": ["QVQLVQSGAEVKKPG..."],')
    print('        "return_std": True,  # Get confidence intervals')
    print("    },")
    print(")")
    print()
    print("result = response.json()")
    print("# {")
    print('#   "liabilities": {')
    print('#     "mean": [{"NG": 0.12, "NX": 0.05, ...}],')
    print('#     "std": [{"NG": 0.02, "NX": 0.01, ...}]')
    print("#   }")
    print("# }")
    print("```")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AbProp Cross-Validation Examples" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    example_1_basic_splitting()
    print("\n")

    example_2_fold_statistics()
    print("\n")

    example_3_dataset_usage()
    print("\n")

    example_4_ensemble_inference()

    print("\n" + "=" * 80)
    print("For full documentation, see: docs/CROSS_VALIDATION.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
