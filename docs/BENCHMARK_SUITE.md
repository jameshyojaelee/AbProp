# AbProp Benchmark Suite Documentation

**Comprehensive benchmark infrastructure for evaluating antibody property prediction models**

## Overview

The AbProp benchmark suite provides a standardized, modular framework for evaluating antibody property prediction models across multiple dimensions. It includes 6 specialized benchmarks covering language modeling, CDR prediction, liability assessment, developability ranking, zero-shot generalization, and difficulty stratification.

### Key Features

✅ **Modular Design**: Registry-based architecture for easy extensibility
✅ **Comprehensive Metrics**: Task-specific metrics with statistical analysis
✅ **Rich Visualizations**: Automated generation of plots and heatmaps
✅ **MLflow Integration**: Automatic logging of metrics and artifacts
✅ **Parallel Execution**: Optional parallel benchmark running
✅ **HTML Reports**: Shareable, interactive evaluation reports
✅ **Difficulty Diagnostics**: Built-in tooling to expose performance cliffs

## Quick Start

### Command Line Usage

Run all benchmarks on a trained model:

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_epoch_10.pt \
    --all \
    --html-report
```

Run specific benchmarks:

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_epoch_10.pt \
    --benchmarks perplexity liability cdr_classification
```

### Programmatic Usage

```python
from abprop.benchmarks import get_registry
from abprop.benchmarks.registry import BenchmarkConfig
from abprop.models import AbPropModel

# Load model
model = AbPropModel.from_checkpoint("path/to/checkpoint.pt")

# Configure and run benchmark
config = BenchmarkConfig(data_path="data/processed/oas")
registry = get_registry()
benchmark = registry.create("perplexity", config)
result = benchmark.run(model)

# Access results
print(f"Perplexity: {result.metrics['overall_perplexity']:.2f}")
```

## Benchmark Tracks

### 1. Perplexity Benchmark

**Purpose**: Evaluate masked language modeling quality on natural antibody sequences.

**What it measures**:
- Overall perplexity across all sequences
- Perplexity by chain type (Heavy vs Light)
- Perplexity by species (human, mouse, etc.)
- Perplexity by sequence length bins

**Why it matters**: Low perplexity indicates the model has learned meaningful antibody sequence patterns and can predict masked residues accurately.

**Expected results**:
- Good model: Perplexity < 5.0
- Baseline model: Perplexity 8-12
- Random model: Perplexity ≈ 20 (vocab size)

**Output files**:
- `perplexity_by_chain.png` - Bar plot comparing H/L chains
- `perplexity_by_species.png` - Performance across species
- `perplexity_by_length.png` - Length-dependent analysis
- `perplexity_distribution.png` - Sequence-level distribution
- `metrics.json` - Detailed metrics
- `sequence_level.json` - Per-sequence results

**Example metrics**:
```json
{
  "overall_perplexity": 4.23,
  "perplexity_chain_H": 4.15,
  "perplexity_chain_L": 4.32,
  "perplexity_species_human": 3.89,
  "perplexity_species_mouse": 4.56
}
```

---

### 2. CDR Classification Benchmark

**Purpose**: Evaluate token-level CDR region prediction accuracy.

**What it measures**:
- Binary classification: Framework (0) vs CDR (1)
- Precision, recall, F1, accuracy
- Per-position accuracy across sequence
- Per-chain metrics (Heavy vs Light)

**Why it matters**: CDR identification is crucial for antibody engineering. Accurate CDR prediction enables targeted optimization of binding regions.

**Expected results**:
- Good model: F1 > 0.85, Accuracy > 0.90
- Baseline model: F1 0.70-0.80
- Rule-based (IMGT): F1 ≈ 0.95 (upper bound)

**Output files**:
- `confusion_matrix.png` - Heatmap of TP/FP/TN/FN
- `position_accuracy.png` - Per-position accuracy line plot
- `metrics_by_chain.png` - Precision/recall/F1 by chain type
- `metrics.json` - Classification metrics
- `detailed_results.json` - Full results

**Example metrics**:
```json
{
  "accuracy": 0.92,
  "precision": 0.87,
  "recall": 0.84,
  "f1": 0.85,
  "tp": 12543,
  "fp": 1876,
  "tn": 48321,
  "fn": 2234
}
```

---

### 3. Liability Benchmark

**Purpose**: Evaluate regression performance for antibody liability prediction.

**What it measures**:
- MSE, R², Spearman correlation per liability type
- Calibration: predicted vs actual values
- Risk stratification: performance on low/medium/high liability sequences

**Liability types**:
- N-glycosylation sites (nglyc)
- Deamidation sites
- Isomerization sites
- Oxidation sites
- Cysteine pairs
- Sequence length

**Why it matters**: Accurate liability prediction is essential for developability assessment. High-liability antibodies may have stability, immunogenicity, or manufacturing issues.

**Expected results**:
- Good model: R² > 0.7, Spearman > 0.75
- Baseline model: R² 0.4-0.6
- Perfect model: R² = 1.0

**Output files**:
- `scatter_plots.png` - Predicted vs actual per liability
- `calibration_plots.png` - Binned calibration analysis
- `risk_stratification.png` - MSE by risk level
- `metrics.json` - Regression metrics
- `risk_stratification.json` - Detailed risk analysis

**Example metrics**:
```json
{
  "overall_mse": 0.134,
  "overall_r2": 0.78,
  "overall_spearman": 0.82,
  "nglyc_r2": 0.85,
  "deamidation_r2": 0.73,
  "oxidation_r2": 0.76,
  "nglyc_risk_low_mse": 0.08,
  "nglyc_risk_high_mse": 0.21
}
```

---

### 4. Developability Benchmark

**Purpose**: Evaluate therapeutic antibody developability prediction.

**Dataset schema (`data/processed/therapeutic_benchmark/therapeutic_benchmark.parquet`)**

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | str | Heavy or light chain amino-acid sequence |
| `chain` | str | Chain type (`H`/`L`) |
| `split` | str | Dataset split (`train`, `val`, `test`) |
| `length` | int | Sequence length |
| `liability_ln` | JSON str | Canonical liabilities (nglyc, deamidation, isomerization, oxidation, free_cysteines) normalized by length |
| `liability_counts` | JSON str | Raw liability counts matching the canonical keys |
| `clinical_phase` | int | Clinical stage (0 = preclinical … 4 = approved) |
| `aggregation_score` | float | Experimental or in-silico aggregation propensity |
| `immunogenicity_score` | float | Immunogenicity risk estimate |
| `developability_score` | float | Composite developability target |
| `known_issues` | str | Optional free-text description of observed liabilities |
| `therapeutic_id` | str | Identifier for traceability |

**What it measures**:
- Composite developability score (from liabilities)
- ROC-AUC for clinical progression (approved vs not)
- Spearman correlation with clinical phase
- Correlation with aggregation propensity
- Correlation with immunogenicity risk

**Why it matters**: Predicting which antibodies will succeed in clinical development saves time and resources. Good models can prioritize candidates with better drug-like properties.

**Expected results**:
- Good model: AUC > 0.75, Spearman with phase > 0.5
- Baseline model: AUC 0.55-0.65
- Random model: AUC ≈ 0.5

**Output files**:
- `score_distribution.png` - Histogram of developability scores
- `roc_curve.png` - ROC for approved vs not approved
- `clinical_phase_boxplot.png` - Scores by clinical phase
- `aggregation_correlation.png` - Developability vs aggregation
- `immunogenicity_correlation.png` - Developability vs immunogenicity
- `metrics.json` - Developability metrics

**Example metrics**:
```json
{
  "mean_developability_score": 0.45,
  "std_developability_score": 1.23,
  "roc_auc_approved": 0.78,
  "spearman_clinical_phase": 0.62,
  "spearman_aggregation": -0.54,
  "spearman_immunogenicity": -0.48
}
```

**Note**: Requires specialized therapeutic antibody dataset with clinical labels. Use `scripts/curate_therapeutic_dataset.py` to create this dataset.

---

### 5. Zero-Shot Benchmark

**Purpose**: Evaluate generalization to unseen species and germline families.

**Dataset schema (`data/processed/zero_shot/zero_shot.parquet`)**

| Column | Type | Description |
|--------|------|-------------|
| `sequence` | str | Antibody sequence used for evaluation |
| `chain` | str | Chain type (`H`/`L`) |
| `split` | str | Dataset split (`train`, `val`, `test`) |
| `length` | int | Sequence length |
| `species` | str | Source species (e.g., `camelus dromedarius`, `homo sapiens`) |
| `germline_v` | str | V gene designation |
| `germline_j` | str | J gene designation |
| `liability_ln` | JSON str | Canonical liabilities normalized by length |
| `liability_counts` | JSON str | Raw liability counts |

**What it measures**:
- Perplexity on rare/novel species (camel, llama, shark)
- Perplexity by germline family (including VHH nanobodies)
- Liability prediction on novel sequences
- Performance gap: common vs rare species

**Why it matters**: Real-world applications require models to generalize beyond training distribution. Zero-shot performance indicates model robustness and transferability.

**Expected results**:
- Good model: Perplexity gap < 2.0 (rare vs common)
- Baseline model: Perplexity gap 3-5
- Poor generalization: Perplexity gap > 8

**Output files**:
- `perplexity_by_species.png` - Bar plot (color-coded common vs rare)
- `perplexity_by_germline.png` - Top 20 germline families
- `liability_by_species.png` - Liability metrics by species
- `species_distribution.png` - Pie chart of species coverage
- `metrics.json` - Zero-shot metrics
- `detailed_results.json` - Full results

**Example metrics**:
```json
{
  "total_sequences": 2341,
  "n_species": 15,
  "n_germlines": 78,
  "mean_perplexity_common": 4.12,
  "mean_perplexity_rare": 5.87,
  "perplexity_gap": 1.75,
  "perplexity_human": 3.89,
  "perplexity_camel": 6.21,
  "perplexity_llama": 5.94
}
```

---

### 6. Difficulty-Stratified Benchmark

**Purpose**: Reveal performance cliffs by evaluating the model across targeted difficulty buckets.

**What it measures**:
- MLM perplexity, CDR classification, and liability regression metrics per difficulty bucket
- Delta-to-baseline metrics that expose regressions on challenging subsets
- Comparative plots across length, complexity, liability load, germline frequency, and species

**Why it matters**: Aggregate metrics can mask systematic failures (e.g., long sequences, rare germlines, zero-shot species). This benchmark surfaces those gaps so you can prioritise mitigation.

**Prerequisites**:
1. Generate balanced stratified splits:
   ```bash
   python scripts/create_difficulty_splits.py \
       --input data/processed/oas_real_full \
       --output data/processed/stratified_test \
       --split test
   ```
2. Ensure `configs/benchmarks.yaml` points `stratified_difficulty.data_path` at the new directory.

**Outputs**:
- `stratified_metrics.json` – Full metric breakdown by dimension/bucket
- `plots/` – Difficulty curves, heatmaps, and error analysis charts (auto-generated)
- Registry metrics keyed as `dimension:bucket:metric` for MLflow logging

**Example metrics**:
```json
{
  "length:short:mlm_perplexity": 3.98,
  "length:long:mlm_perplexity": 5.42,
  "complexity:unusual_composition:cls_f1": 0.61,
  "germline:novel_germline:reg_r2": 0.32,
  "species:other_species:mlm_perplexity_delta": 1.27
}
```

**Interpretation tips**:
- Focus on the largest positive `*_delta` values—they mark the steepest drops vs baseline.
- Check `heatmap_*.png` to spot clusters of weakness across buckets.
- Re-run after remediation steps (curriculum finetuning, targeted augmentation) to confirm improvements.

---

## Configuration

Benchmarks are configured via `configs/benchmarks.yaml`:

```yaml
# Global settings
output_dir: ./outputs/benchmarks
batch_size: 32
device: cuda
mlflow_tracking: true

# Per-benchmark configuration
perplexity:
  data_path: ./data/processed/oas
  batch_size: 32
  max_samples: null  # null = all data

liability:
  data_path: ./data/processed/oas
  risk_stratification:
    enabled: true
    thresholds: [33, 66]

developability:
  data_path: ./data/processed/therapeutic
  batch_size: 16

zero_shot:
  data_path: ./data/processed/zero_shot
  unseen_species:
    - camel
    - llama
    - shark
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | Path | Required | Path to benchmark dataset |
| `batch_size` | int | 32 | Batch size for evaluation |
| `max_samples` | int? | null | Limit number of samples (for testing) |
| `device` | str | cuda | Device to run on (cuda/cpu) |
| `output_dir` | Path | outputs/benchmarks | Output directory |
| `mlflow_tracking` | bool | true | Enable MLflow logging |

## Command Line Options

### Basic Options

```bash
--checkpoint PATH          Path to model checkpoint (required)
--config PATH             Path to benchmark config (default: configs/benchmarks.yaml)
--model-config PATH       Path to model config (default: configs/model.yaml)
--output-dir PATH         Output directory (default: outputs/benchmarks)
```

### Benchmark Selection

```bash
--all                     Run all available benchmarks
--benchmarks NAME [NAME]  Run specific benchmarks (e.g., perplexity liability)
```

### Evaluation Parameters

```bash
--batch-size INT         Batch size (default: 32)
--max-samples INT        Max samples per benchmark (for testing)
--device STR             Device (cuda or cpu)
```

### Execution Options

```bash
--parallel               Run benchmarks in parallel (experimental)
--no-mlflow             Disable MLflow logging
--html-report           Generate HTML report
```

## Output Structure

```
outputs/benchmarks/
├── summary.json                    # Overall summary
├── report.html                     # HTML report (if --html-report)
├── perplexity/
│   ├── metrics.json
│   ├── sequence_level.json
│   ├── perplexity_by_chain.png
│   ├── perplexity_by_species.png
│   ├── perplexity_by_length.png
│   └── perplexity_distribution.png
├── cdr_classification/
│   ├── metrics.json
│   ├── detailed_results.json
│   ├── confusion_matrix.png
│   ├── position_accuracy.png
│   └── metrics_by_chain.png
├── liability/
│   ├── metrics.json
│   ├── risk_stratification.json
│   ├── scatter_plots.png
│   ├── calibration_plots.png
│   └── risk_stratification.png
├── developability/
│   ├── metrics.json
│   ├── developability_scores.json
│   ├── score_distribution.png
│   ├── roc_curve.png
│   ├── clinical_phase_boxplot.png
│   ├── aggregation_correlation.png
│   └── immunogenicity_correlation.png
└── zero_shot/
    ├── metrics.json
    ├── detailed_results.json
    ├── perplexity_by_species.png
    ├── perplexity_by_germline.png
    ├── liability_by_species.png
    └── species_distribution.png
```

## MLflow Integration

Benchmarks automatically log results to MLflow when enabled (default):

### Logged Artifacts

- All metrics as MLflow metrics (e.g., `perplexity/overall_perplexity`)
- All plots as artifacts under `benchmarks/{benchmark_name}/`
- Summary JSON as artifact
- HTML report (if generated)

### Viewing Results

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser
http://localhost:5000
```

Navigate to "abprop-benchmarks" experiment to view results.

### Comparing Runs

Use MLflow UI to compare metrics across different model checkpoints:

1. Select multiple runs
2. Click "Compare"
3. View metric plots and tables

## Example Workflows

### 1. Quick Evaluation

Test benchmarks on limited data:

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_latest.pt \
    --benchmarks perplexity \
    --max-samples 100
```

### 2. Full Evaluation

Run all benchmarks with HTML report:

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_epoch_10.pt \
    --all \
    --html-report \
    --output-dir outputs/evaluation_epoch_10
```

### 3. Parallel Execution

Run benchmarks in parallel (faster, but more GPU memory):

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_epoch_10.pt \
    --all \
    --parallel
```

### 4. Custom Configuration

Use custom benchmark config:

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_epoch_10.pt \
    --config configs/my_benchmarks.yaml \
    --all
```

### 5. CPU Evaluation

Run on CPU (slower, but no GPU required):

```bash
python scripts/run_benchmarks.py \
    --checkpoint outputs/checkpoints/model_epoch_10.pt \
    --all \
    --device cpu
```

## Adding Custom Benchmarks

### 1. Create Benchmark Class

Create a new file `src/abprop/benchmarks/my_benchmark.py`:

```python
from abprop.benchmarks.registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark

@register_benchmark("my_benchmark")
class MyBenchmark(Benchmark):
    def load_data(self) -> DataLoader:
        # Load your data
        dataset = MyDataset(self.config.data_path)
        return DataLoader(dataset, batch_size=self.config.batch_size)

    def evaluate(self, model, dataloader) -> Dict[str, Any]:
        # Run evaluation
        results = {}
        for batch in dataloader:
            outputs = model(batch)
            # Process outputs...
        return results

    def report(self, results) -> BenchmarkResult:
        # Generate report
        metrics = {"my_metric": results["my_value"]}
        plots = {}  # Generate plots...
        metadata = {}
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )
```

### 2. Register Benchmark

Add import in `src/abprop/benchmarks/__init__.py`:

```python
from . import my_benchmark
```

### 3. Add Configuration

Add to `configs/benchmarks.yaml`:

```yaml
my_benchmark:
  data_path: ./data/processed/my_data
  batch_size: 32
```

### 4. Run

```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --benchmarks my_benchmark
```

## Best Practices

### Data Preparation

1. **Use test splits**: Always evaluate on held-out test data
2. **Check data quality**: Validate sequences before benchmarking
3. **Stratify splits**: Ensure balanced species/chain representation

### Evaluation

1. **Baseline comparison**: Compare against random/rule-based baselines
2. **Multiple checkpoints**: Evaluate at different training epochs
3. **Statistical significance**: Use multiple random seeds for training

### Reporting

1. **Document setup**: Record model config, data version, hyperparameters
2. **Share artifacts**: Use MLflow to share results with collaborators
3. **Generate HTML reports**: Create shareable reports for presentations

### Performance

1. **GPU memory**: Reduce batch size if OOM errors occur
2. **Parallel execution**: Use with caution (experimental)
3. **Sample limiting**: Use `--max-samples` for quick testing

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory during evaluation

**Solution**:
```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --batch-size 16  # Reduce batch size
    --benchmarks perplexity  # Run one at a time
```

### Missing Data Path

**Problem**: `FileNotFoundError: data/processed/oas not found`

**Solution**: Update `configs/benchmarks.yaml` with correct paths or run ETL pipeline:
```bash
python scripts/process_real_data_etl.py \
    --input data/raw/sabdab_sequences_full.csv \
    --output data/processed/oas
```

### Benchmark Not Found

**Problem**: `ValueError: Benchmark 'xyz' not found`

**Solution**: Check benchmark name and ensure it's imported in `__init__.py`:
```python
# Available benchmarks
from abprop.benchmarks import get_registry
print(get_registry().list_benchmarks())
```

### Slow Evaluation

**Problem**: Benchmarks taking too long

**Solution**: Use sample limiting for testing:
```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --max-samples 1000 \
    --all
```

## Comparison with Existing Evaluation

### Old Evaluation (`src/abprop/commands/eval.py`)

- Single-file implementation
- Basic metrics only
- Limited stratification
- No registry pattern
- Manual plot generation

### New Benchmark Suite

- Modular, extensible architecture
- Comprehensive metrics with statistical analysis
- Multi-dimensional stratification
- Registry-based discovery
- Automated visualization
- MLflow integration
- HTML reports
- Parallel execution support

### Migration Path

Continue using old evaluation for quick checks:
```bash
python scripts/eval.py --checkpoint path/to/checkpoint.pt --splits test
```

Use benchmark suite for comprehensive evaluation:
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all
```

## Performance Expectations

### Runtime

| Benchmark | Dataset Size | Batch Size | Device | Time |
|-----------|--------------|------------|--------|------|
| Perplexity | 10,000 seqs | 32 | V100 | ~5 min |
| CDR Classification | 10,000 seqs | 32 | V100 | ~5 min |
| Liability | 10,000 seqs | 32 | V100 | ~6 min |
| Developability | 1,000 seqs | 16 | V100 | ~2 min |
| Zero-Shot | 5,000 seqs | 32 | V100 | ~4 min |
| Difficulty-Stratified | 5,000 seqs | 32 | V100 | ~5 min |
| **Total** | - | - | V100 | **~27 min** |

### Memory Usage

| Benchmark | Batch Size 32 | Batch Size 64 |
|-----------|---------------|---------------|
| Perplexity | ~8 GB | ~14 GB |
| CDR Classification | ~8 GB | ~14 GB |
| Liability | ~9 GB | ~16 GB |
| Developability | ~7 GB | ~12 GB |
| Zero-Shot | ~8 GB | ~14 GB |
| Difficulty-Stratified | ~9 GB | ~16 GB |

## Future Enhancements

Planned improvements:

- [ ] Multi-task joint benchmarks
- [ ] Temporal validation (time-based splits)
- [ ] Cross-dataset evaluation
- [ ] Adversarial robustness tests
- [ ] Interpretability metrics
- [ ] Antibody-specific perplexity (vs general proteins)
- [ ] Binding affinity prediction benchmark
- [ ] Structure-based evaluation
- [ ] Real-time monitoring dashboard

## Citation

If you use the AbProp benchmark suite in your research, please cite:

```bibtex
@software{abprop_benchmarks,
  title={AbProp: Comprehensive Benchmark Suite for Antibody Property Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/AbProp}
}
```

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/yourusername/AbProp/issues
- Documentation: [src/abprop/benchmarks/README.md](../src/abprop/benchmarks/README.md)
- Examples: [examples/run_benchmark_example.py](../examples/run_benchmark_example.py)

---

**Last Updated**: October 13, 2025
**Status**: ✅ Complete
**Version**: 1.0.0
