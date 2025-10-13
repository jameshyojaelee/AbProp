# Cross-Validation Framework

This document describes AbProp's k-fold cross-validation framework for robust model performance estimation. The framework ensures clonotype-aware splitting to prevent data leakage and provides ensemble predictions with confidence intervals.

## Table of Contents

1. [Overview](#overview)
2. [Clonotype-Aware Splitting](#clonotype-aware-splitting)
3. [Usage Guide](#usage-guide)
4. [Interpreting CV Results](#interpreting-cv-results)
5. [When to Use CV vs Single Split](#when-to-use-cv-vs-single-split)
6. [Ensemble Inference](#ensemble-inference)
7. [API Reference](#api-reference)

## Overview

Cross-validation (CV) is a statistical technique for robust model evaluation that:

- **Reduces variance** in performance estimates by averaging over multiple train/test splits
- **Detects overfitting** by comparing in-fold vs out-of-fold performance
- **Enables ensemble predictions** by averaging predictions from multiple models
- **Provides confidence intervals** through standard deviation estimates

AbProp's CV framework uses **clonotype-aware k-fold splitting** to ensure sequences from the same clonotype (antibodies with the same CDR3 region) never appear in both training and validation sets.

### Key Features

✅ **Clonotype-aware splitting**: Prevents data leakage across folds
✅ **Stratified by species and chain**: Maintains balanced distributions
✅ **Configurable k-folds**: Default 5-fold, customizable to any k ≥ 2
✅ **Ensemble predictions**: Average across all k models with std dev
✅ **Production-ready**: Integrated with training, evaluation, and inference pipelines

## Clonotype-Aware Splitting

### What is a Clonotype?

A **clonotype** represents a family of related antibody sequences that share the same:
- Chain type (Heavy or Light)
- CDR3 amino acid sequence (or full sequence if CDR3 unavailable)

Sequences from the same clonotype are likely to be similar, so keeping them together prevents inflated performance estimates.

### Splitting Strategy

The `ClonotypeAwareKFold` splitter implements the following algorithm:

1. **Group by clonotype**: Assign each sequence a `clonotype_key = chain|cdr3`
2. **Stratify by species & chain**: Split within each (species, chain) group
3. **Distribute clonotypes across folds**: Each fold gets ~equal number of clonotypes
4. **Keep clonotypes intact**: All sequences from a clonotype go to the same fold

### Example

Given 1000 sequences with 200 unique clonotypes:

```
5-fold CV:
- Fold 0: 160 sequences (40 clonotypes) → validation
          840 sequences (160 clonotypes) → training
- Fold 1: 180 sequences (40 clonotypes) → validation
          820 sequences (160 clonotypes) → training
...
```

Each fold uses different clonotypes for validation, ensuring independent estimates.

## Usage Guide

### Training with Cross-Validation

Train k models on k folds using `scripts/train_cv.py`:

```bash
# Basic 5-fold CV training
python scripts/train_cv.py \
  --config-path configs/train.yaml \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml \
  --output-dir outputs/cv_5fold

# Custom 10-fold CV
python scripts/train_cv.py \
  --n-folds 10 \
  --output-dir outputs/cv_10fold

# Resume from specific fold
python scripts/train_cv.py \
  --n-folds 5 \
  --start-fold 2 \
  --output-dir outputs/cv_5fold

# Quick test with synthetic data
python scripts/train_cv.py \
  --synthetic \
  --dry-run-steps 10 \
  --n-folds 3
```

**Output Structure:**

```
outputs/cv_5fold/
├── fold_0/
│   ├── checkpoints/
│   │   ├── best.pt
│   │   └── last.pt
│   └── logs/
├── fold_1/
│   └── ...
├── ...
├── fold_4/
│   └── ...
├── cv_results.json          # Aggregated metrics
└── cv_summary.csv           # Summary table
```

### Evaluating CV Models

Evaluate all folds and generate ensemble predictions using `scripts/eval_cv.py`:

```bash
# Evaluate all CV folds on their validation sets
python scripts/eval_cv.py \
  --cv-dir outputs/cv_5fold \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml

# Evaluate on test split (held-out data)
python scripts/eval_cv.py \
  --cv-dir outputs/cv_5fold \
  --test-split

# Generate ensemble predictions for specific sequences
python scripts/eval_cv.py \
  --cv-dir outputs/cv_5fold \
  --predict-file sequences.csv \
  --output-file predictions.npz
```

**Input format for `--predict-file`:**

```csv
sequence
QVQLVQSGAEVKKPGASVKVSCKAS...
DIQMTQSPSSLSASVGDRVTITC...
```

**Output:**

```
outputs/cv_5fold/
├── eval_results.json         # Fold-wise and aggregated metrics
├── ensemble_predictions.npz  # NumPy arrays with mean/std
└── ensemble_predictions.csv  # Human-readable summary
```

### Programmatic API Usage

```python
from pathlib import Path
import pandas as pd
from abprop.data.cross_validation import (
    ClonotypeAwareKFold,
    GroupKFoldDataset,
    generate_cv_folds,
    stratified_cv_summary,
)

# Load your dataset
df = pd.read_parquet("data/processed/oas")

# Generate CV folds
folds = generate_cv_folds(df, n_splits=5, test_ratio=0.2, seed=42)

# Inspect fold statistics
summary = stratified_cv_summary(df, folds)
print(summary)

# Create fold-specific datasets for training
train_dataset = GroupKFoldDataset(
    parquet_dir="data/processed/oas",
    fold_idx=0,
    n_splits=5,
    split_type="train",
    seed=42,
)

val_dataset = GroupKFoldDataset(
    parquet_dir="data/processed/oas",
    fold_idx=0,
    n_splits=5,
    split_type="val",
    seed=42,
)
```

## Interpreting CV Results

### Metrics Output

After training, `cv_results.json` contains:

```json
{
  "n_folds": 5,
  "aggregated_metrics": {
    "eval_loss": {
      "mean": 2.345,
      "std": 0.123,
      "min": 2.201,
      "max": 2.489,
      "values": [2.345, 2.401, 2.201, 2.489, 2.389]
    },
    "eval_mlm_loss": { ... }
  },
  "fold_metrics": [ ... ]
}
```

### Understanding the Statistics

- **Mean**: Average performance across all k folds
- **Std Dev**: Variability in performance (lower = more consistent)
- **Min/Max**: Best and worst fold performance
- **Values**: Per-fold results for transparency

### What Good Results Look Like

✅ **Low std dev** (< 10% of mean): Consistent performance across folds
✅ **No outlier folds**: Min/max within ~2 std devs of mean
✅ **Comparable to single split**: CV mean ≈ single train/val/test result

⚠️ **Warning signs:**

- High std dev (> 20% of mean): Model sensitive to data split
- Large min/max gap: Potential overfitting or data imbalance
- CV mean << single split: Data leakage in single split evaluation

### Example Interpretation

```
eval_loss: 2.345 ± 0.123
  - Mean loss of 2.345 with 5.2% relative std dev
  - Indicates robust, consistent performance

eval_loss: 2.345 ± 0.687
  - Mean loss of 2.345 with 29.3% relative std dev
  - High variance suggests model instability
  - Consider: more data, regularization, or different architecture
```

## When to Use CV vs Single Split

### Use Cross-Validation When:

✅ **Publishing results**: CV provides rigorous, publication-quality estimates
✅ **Small datasets**: Maximizes use of available data
✅ **Comparing models**: Reduces noise in performance comparisons
✅ **Production deployment**: Ensemble models often outperform single models
✅ **Uncertainty quantification**: Need confidence intervals on predictions

### Use Single Split When:

✅ **Large datasets** (> 100k sequences): Single split is representative
✅ **Quick iteration**: Faster development cycle
✅ **Resource constraints**: CV requires k× compute time
✅ **Deployment size matters**: Single model has smaller memory footprint

### Hybrid Approach

For best of both worlds:

1. **Development**: Use single split for rapid iteration
2. **Final evaluation**: Run CV on best model for robust estimates
3. **Deployment**: Use single best model or ensemble based on requirements

## Ensemble Inference

Ensemble predictions combine outputs from all k fold models, providing:

- **Better accuracy**: Averaging reduces overfitting to any single fold
- **Uncertainty estimates**: Standard deviation indicates prediction confidence
- **Robustness**: Less sensitive to outlier predictions from individual models

### Server Deployment

Deploy ensemble inference server from CV checkpoints:

```python
from pathlib import Path
from abprop.server.app import create_ensemble_app_from_cv

# Create ensemble app from CV directory
app = create_ensemble_app_from_cv(
    cv_dir=Path("outputs/cv_5fold"),
    model_config=Path("configs/model.yaml"),
    device="cuda",
)

# Run with uvicorn
# uvicorn app:app --host 0.0.0.0 --port 8000
```

### API Usage

```python
import requests

# Score liabilities with ensemble
response = requests.post(
    "http://localhost:8000/score/liabilities",
    json={
        "sequences": ["QVQLVQSGAEVKKPG..."],
        "return_std": True,  # Request std dev estimates
    }
)

result = response.json()
# {
#   "liabilities": {
#     "mean": [{"NG": 0.12, "NX": 0.05, ...}],
#     "std": [{"NG": 0.02, "NX": 0.01, ...}]
#   }
# }
```

### Health Check

```python
response = requests.get("http://localhost:8000/health")
# {
#   "status": "ok",
#   "device": "cuda:0",
#   "ensemble_mode": true,
#   "n_models": 5
# }
```

## API Reference

### Core Classes

#### `ClonotypeAwareKFold`

Clonotype-aware k-fold splitter.

```python
from abprop.data.cross_validation import ClonotypeAwareKFold

splitter = ClonotypeAwareKFold(n_splits=5, shuffle=True, seed=42)

for train_idx, val_idx in splitter.split(df, clonotype_col="clonotype_key"):
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    # Train model on train_data, evaluate on val_data
```

**Parameters:**
- `n_splits` (int): Number of folds (≥ 2)
- `shuffle` (bool): Whether to shuffle before splitting
- `seed` (int): Random seed for reproducibility

**Methods:**
- `split(df, clonotype_col, stratify_cols)`: Generate train/val indices

#### `GroupKFoldDataset`

PyTorch Dataset wrapper for k-fold CV splits.

```python
from abprop.data.cross_validation import GroupKFoldDataset

dataset = GroupKFoldDataset(
    parquet_dir="data/processed/oas",
    fold_idx=0,
    n_splits=5,
    split_type="train",  # or "val"
    seed=42,
)
```

**Parameters:**
- `parquet_dir` (Path): Path to parquet dataset
- `fold_idx` (int): Which fold to use (0 to n_splits-1)
- `n_splits` (int): Total number of folds
- `split_type` (str): "train" or "val"
- `columns` (Optional[Sequence[str]]): Columns to load
- `seed` (int): Random seed

**Attributes:**
- `data` (pd.DataFrame): Filtered data for this fold
- `lengths` (List[int]): Sequence lengths for bucketing
- `has_cdr` (bool): Whether CDR masks are available

### Utility Functions

#### `generate_cv_folds`

Generate k-fold splits with optional test set holdout.

```python
from abprop.data.cross_validation import generate_cv_folds

folds = generate_cv_folds(
    df=data,
    n_splits=5,
    test_ratio=0.2,  # Hold out 20% for final testing
    seed=42,
)
```

**Returns:** `List[CVFold]` with train/val/test indices

#### `stratified_cv_summary`

Generate summary statistics showing fold balance.

```python
from abprop.data.cross_validation import stratified_cv_summary

summary = stratified_cv_summary(df, folds)
print(summary)
#    fold  split  n_sequences  n_heavy  n_light  n_species
# 0     0  train        8000     4500     3500          3
# 1     0    val        2000     1125      875          3
# ...
```

### Server API

#### `create_ensemble_app_from_cv`

Create FastAPI app with ensemble inference from CV directory.

```python
from pathlib import Path
from abprop.server.app import create_ensemble_app_from_cv

app = create_ensemble_app_from_cv(
    cv_dir=Path("outputs/cv_5fold"),
    model_config=Path("configs/model.yaml"),
    device="cuda",
)
```

**Endpoints:**

- `GET /health`: Server status and model info
- `POST /score/perplexity`: Score sequence perplexity
- `POST /score/liabilities`: Score liability motifs

**Request format:**

```json
{
  "sequences": ["QVQLVQSGAEVKKPG...", "DIQMTQSPSSLSASV..."],
  "return_std": true
}
```

**Response format:**

```json
{
  "liabilities": {
    "mean": [
      {"NG": 0.12, "NX": 0.05, "NP": 0.03},
      {"NG": 0.08, "NX": 0.02, "NP": 0.01}
    ],
    "std": [
      {"NG": 0.02, "NX": 0.01, "NP": 0.005},
      {"NG": 0.015, "NX": 0.008, "NP": 0.003}
    ]
  }
}
```

## Best Practices

### 1. Choose Appropriate k

- **k=5**: Standard choice, good balance of bias/variance
- **k=10**: More robust estimates, higher compute cost
- **k=3**: Quick experiments, higher variance
- **Leave-one-out (k=n)**: Maximum data use, expensive for large datasets

### 2. Set Fixed Random Seed

Always use the same seed across folds for reproducibility:

```python
python scripts/train_cv.py --seed 42 --n-folds 5
```

### 3. Monitor Fold Variance

High variance suggests:
- Data imbalance → Inspect stratification
- Model instability → Increase regularization
- Overfitting → Add dropout, reduce capacity

### 4. Use Test Set for Final Evaluation

CV folds are still "training data" in a sense. Hold out a final test set:

```python
folds = generate_cv_folds(df, n_splits=5, test_ratio=0.2)
```

Train on CV folds, select best hyperparameters, then evaluate on test set once.

### 5. Leverage Ensemble Predictions

Ensemble predictions almost always outperform single models:

```python
# During inference, use ensemble for production
app = create_ensemble_app_from_cv(cv_dir, model_config, device="cuda")
```

## Troubleshooting

### Issue: Fold imbalance

**Symptom:** Some folds have very different sizes

**Solution:** Check stratification columns match your data:

```python
# Default stratifies by species and chain
# Add more columns if needed
splitter.split(df, stratify_cols=["species", "chain", "germline_v"])
```

### Issue: High variance across folds

**Symptom:** `std > 0.2 * mean`

**Solutions:**
1. Increase k (more folds → lower variance)
2. Add regularization (dropout, weight decay)
3. Check for data quality issues
4. Use ensemble predictions

### Issue: Memory errors with ensemble

**Symptom:** OOM when loading all k models

**Solutions:**
1. Load models sequentially (slower but lower memory)
2. Use smaller batch size
3. Deploy single best model instead of ensemble
4. Use model distillation to create single lightweight model

### Issue: Clonotype leakage

**Symptom:** CV results much better than test set

**Diagnosis:** Check clonotype assignment:

```python
df = assign_clonotype(df)
print(df["clonotype_key"].nunique(), "unique clonotypes")
print(len(df), "total sequences")
# Should have significantly fewer clonotypes than sequences
```

If `n_clonotypes ≈ n_sequences`, CDR3 data may be missing and falling back to full sequence.

## References

- **Original ETL code**: [src/abprop/data/etl.py:140](../src/abprop/data/etl.py#L140) - `assign_clonotype`
- **CV splitter**: [src/abprop/data/cross_validation.py](../src/abprop/data/cross_validation.py)
- **Training script**: [scripts/train_cv.py](../scripts/train_cv.py)
- **Evaluation script**: [scripts/eval_cv.py](../scripts/eval_cv.py)
- **Server API**: [src/abprop/server/app.py](../src/abprop/server/app.py)

## Citation

If you use this cross-validation framework in your research, please cite:

```bibtex
@software{abprop_cv,
  title = {AbProp: Clonotype-Aware Cross-Validation for Antibody Property Prediction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/abprop}
}
```

---

**Questions or issues?** Open an issue on GitHub or contact the maintainers.
