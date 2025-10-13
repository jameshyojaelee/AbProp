# Cross-Validation Quick Start

A minimal guide to using AbProp's k-fold cross-validation framework.

## 🚀 Quick Start

### 1. Train with 5-Fold CV

```bash
# Basic training with default 5-fold CV
python scripts/train_cv.py \
  --config-path configs/train.yaml \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml \
  --output-dir outputs/cv_5fold \
  --seed 42
```

**Output:**
```
outputs/cv_5fold/
├── fold_0/checkpoints/best.pt
├── fold_1/checkpoints/best.pt
├── ...
├── cv_results.json          # Aggregated metrics: mean ± std
└── cv_summary.csv           # Summary table
```

### 2. Evaluate CV Models

```bash
# Evaluate all folds and report aggregated metrics
python scripts/eval_cv.py \
  --cv-dir outputs/cv_5fold \
  --model-config configs/model.yaml \
  --data-config configs/data.yaml
```

**Output:**
```
eval_loss: 2.345 ± 0.123
eval_mlm_loss: 1.987 ± 0.098
eval_cls_loss: 0.245 ± 0.032
```

### 3. Deploy Ensemble Inference

```python
from pathlib import Path
from abprop.server.app import create_ensemble_app_from_cv

# Create ensemble app from all fold checkpoints
app = create_ensemble_app_from_cv(
    cv_dir=Path("outputs/cv_5fold"),
    model_config=Path("configs/model.yaml"),
    device="cuda",
)

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

**API Usage:**

```bash
curl -X POST http://localhost:8000/score/liabilities \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["QVQLVQSGAEVKKPG..."],
    "return_std": true
  }'
```

**Response:**

```json
{
  "liabilities": {
    "mean": [{"NG": 0.12, "NX": 0.05, "NP": 0.03}],
    "std": [{"NG": 0.02, "NX": 0.01, "NP": 0.005}]
  }
}
```

## 📊 What is Clonotype-Aware CV?

**Problem:** Naive k-fold CV can put similar sequences in both train and validation sets, inflating performance estimates.

**Solution:** Group sequences by **clonotype** (chain + CDR3 region) and ensure all sequences from the same clonotype stay together in one fold.

**Example:**

```
❌ Naive split (data leakage):
Train:   QVQL...CDR3A...KPG  ← Same CDR3
Val:     QVQL...CDR3A...KPG  ← Same CDR3 (leaked!)

✅ Clonotype-aware split:
Train:   QVQL...CDR3A...KPG  ← Same CDR3
         QVQL...CDR3A...LPG  ← Same CDR3 (kept together)
Val:     DIQM...CDR3B...SSV  ← Different CDR3 (no leak)
```

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Clonotype-aware** | Prevents data leakage across folds |
| **Stratified** | Maintains species/chain balance |
| **Ensemble predictions** | Average across k models with std dev |
| **Configurable k** | Default 5-fold, supports any k ≥ 2 |
| **Reproducible** | Fixed random seeds for consistency |

## 📈 Interpreting Results

### Good Results ✅

```
eval_loss: 2.345 ± 0.123
```
- Low std dev (< 10% of mean)
- Indicates robust, consistent performance

### Warning Signs ⚠️

```
eval_loss: 2.345 ± 0.687
```
- High std dev (> 20% of mean)
- Model sensitive to data split
- Consider: more data, regularization, different architecture

## 🔧 Common Commands

### Resume Training

```bash
# Resume from fold 3 onwards
python scripts/train_cv.py \
  --n-folds 5 \
  --start-fold 3 \
  --output-dir outputs/cv_5fold
```

### 10-Fold CV

```bash
# More robust estimates, higher compute
python scripts/train_cv.py \
  --n-folds 10 \
  --output-dir outputs/cv_10fold
```

### Quick Test

```bash
# Test with synthetic data
python scripts/train_cv.py \
  --synthetic \
  --dry-run-steps 10 \
  --n-folds 3
```

### Ensemble Predictions

```bash
# Generate ensemble predictions for sequences.csv
python scripts/eval_cv.py \
  --cv-dir outputs/cv_5fold \
  --predict-file sequences.csv \
  --output-file predictions.npz
```

## 📚 Full Documentation

For comprehensive documentation, see:
- **[CROSS_VALIDATION.md](CROSS_VALIDATION.md)** - Complete guide with methodology, API reference, troubleshooting
- **[examples/cv_example.py](../examples/cv_example.py)** - Runnable examples

## 🤔 When to Use CV?

| Use Case | Recommendation |
|----------|----------------|
| Publishing results | ✅ Use CV (robust estimates) |
| Small datasets (< 10k) | ✅ Use CV (maximize data use) |
| Model comparison | ✅ Use CV (reduce noise) |
| Production deployment | ✅ Use ensemble (better accuracy) |
| Quick iteration | ❌ Use single split (faster) |
| Very large datasets | ❌ Use single split (representative) |

## 💡 Pro Tips

1. **Always use the same seed** for reproducibility: `--seed 42`
2. **Monitor fold variance**: High variance → model instability
3. **Hold out test set**: Use `test_ratio=0.2` in `generate_cv_folds()`
4. **Leverage ensembles**: Almost always outperform single models
5. **Check clonotype counts**: Should have significantly fewer clonotypes than sequences

## 🐛 Troubleshooting

### Issue: Clonotype overlap detected

```python
# Check clonotype assignment
from abprop.data.etl import assign_clonotype
df = assign_clonotype(df)
print(f"Unique clonotypes: {df['clonotype_key'].nunique()}")
print(f"Total sequences: {len(df)}")
```

If `n_clonotypes ≈ n_sequences`, CDR3 data may be missing.

### Issue: Memory error with ensemble

Solutions:
1. Use smaller batch size
2. Load models sequentially (slower but lower memory)
3. Deploy single best model instead

### Issue: High fold variance

Solutions:
1. Increase k (e.g., 10-fold instead of 5-fold)
2. Add regularization (dropout, weight decay)
3. Check for data quality issues

## 📖 Citation

```bibtex
@software{abprop_cv,
  title = {AbProp: Clonotype-Aware Cross-Validation for Antibody Property Prediction},
  year = {2025},
  url = {https://github.com/yourusername/abprop}
}
```
