# AbProp Benchmark Suite

Comprehensive benchmark infrastructure for evaluating antibody property prediction models across multiple dimensions.

## Overview

The AbProp benchmark suite provides standardized evaluation tracks to measure model performance on various antibody-related tasks:

- **Perplexity Benchmark**: Language modeling quality on natural antibody sequences
- **CDR Classification Benchmark**: Token-level CDR region prediction
- **Liability Benchmark**: Regression metrics for developability liabilities
- **Developability Benchmark**: Therapeutic antibody ranking and clinical progression
- **Zero-Shot Benchmark**: Generalization to unseen species and germlines

## Architecture

### Registry Pattern

All benchmarks inherit from the `Benchmark` base class and implement three core methods:

```python
class Benchmark(ABC):
    def load_data(self) -> DataLoader:
        """Load and prepare the benchmark dataset."""
        pass

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Run evaluation on the model."""
        pass

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate benchmark report with metrics and visualizations."""
        pass
```

Benchmarks are registered using the `@register_benchmark` decorator:

```python
@register_benchmark("perplexity")
class PerplexityBenchmark(Benchmark):
    ...
```

### BenchmarkConfig

Each benchmark is configured via a `BenchmarkConfig` dataclass:

```python
@dataclass
class BenchmarkConfig:
    data_path: Path              # Path to benchmark dataset
    batch_size: int = 32         # Batch size for evaluation
    max_samples: Optional[int]   # Optional sample limit
    device: str = "cuda"         # Device to run on
    output_dir: Path             # Directory for results
    mlflow_tracking: bool = True # Enable MLflow logging
```

### BenchmarkResult

Evaluation results are returned as a `BenchmarkResult`:

```python
@dataclass
class BenchmarkResult:
    benchmark_name: str           # Name of the benchmark
    metrics: Dict[str, float]     # Metric name -> value
    plots: Dict[str, Path]        # Plot name -> file path
    metadata: Dict[str, Any]      # Additional metadata
```

## Running Benchmarks

### Command Line Interface

Run all benchmarks:

```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all
```

Run specific benchmarks:

```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --benchmarks perplexity liability cdr_classification
```

With custom configuration:

```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --config configs/benchmarks.yaml \
    --output-dir outputs/my_benchmarks \
    --batch-size 64
```

Generate HTML report:

```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --all \
    --html-report
```

### Programmatic Usage

```python
from abprop.benchmarks import get_registry
from abprop.benchmarks.registry import BenchmarkConfig
from abprop.models import AbPropModel

# Load model
model = AbPropModel.from_checkpoint("path/to/checkpoint.pt")

# Get registry
registry = get_registry()

# Configure benchmark
config = BenchmarkConfig(
    data_path="data/processed/oas",
    batch_size=32,
    output_dir="outputs/benchmarks",
)

# Create and run benchmark
benchmark = registry.create("perplexity", config)
result = benchmark.run(model)

# Access results
print(f"Perplexity: {result.metrics['overall_perplexity']:.2f}")
for plot_name, plot_path in result.plots.items():
    print(f"Plot saved: {plot_path}")
```

## Benchmark Details

### 1. Perplexity Benchmark

**Purpose**: Evaluate language modeling quality on natural antibody sequences.

**Metrics**:
- Overall perplexity across all sequences
- Perplexity stratified by chain type (Heavy, Light)
- Perplexity stratified by species (human, mouse, etc.)
- Perplexity stratified by sequence length bins

**Visualizations**:
- Bar plots: perplexity by chain/species
- Line plot: perplexity vs sequence length
- Histogram: distribution of sequence-level perplexities

**Usage**:
```python
config = BenchmarkConfig(data_path="data/processed/oas")
benchmark = registry.create("perplexity", config)
result = benchmark.run(model)
```

### 2. CDR Classification Benchmark

**Purpose**: Evaluate token-level CDR region prediction accuracy.

**Metrics**:
- Overall precision, recall, F1, accuracy
- Per-chain metrics (Heavy vs Light)
- Confusion matrix (TP, FP, TN, FN)

**Visualizations**:
- Confusion matrix heatmap
- Per-position accuracy line plot
- Per-chain metrics bar plots

**Usage**:
```python
config = BenchmarkConfig(data_path="data/processed/oas")
benchmark = registry.create("cdr_classification", config)
result = benchmark.run(model)
```

### 3. Liability Benchmark

**Purpose**: Evaluate regression performance for antibody liability prediction.

**Metrics**:
- Overall MSE, R², Spearman correlation
- Per-liability metrics (nglyc, deamidation, oxidation, etc.)
- Risk stratification (low/medium/high liability MSE)

**Visualizations**:
- Scatter plots: predicted vs actual per liability
- Calibration plots: binned predictions vs actual
- Bar plot: MSE by risk stratification

**Usage**:
```python
config = BenchmarkConfig(data_path="data/processed/oas")
benchmark = registry.create("liability", config)
result = benchmark.run(model)
```

### 4. Developability Benchmark

**Purpose**: Evaluate therapeutic antibody developability prediction.

**Metrics**:
- Mean developability score
- ROC-AUC for clinical progression (approved vs not approved)
- Spearman correlation with clinical phase
- Correlation with aggregation/immunogenicity scores

**Visualizations**:
- Developability score distribution
- ROC curve for approved vs not approved
- Box plot by clinical phase
- Scatter plots: aggregation/immunogenicity vs developability

**Usage**:
```python
config = BenchmarkConfig(data_path="data/processed/therapeutic")
benchmark = registry.create("developability", config)
result = benchmark.run(model)
```

**Note**: Requires specialized therapeutic antibody dataset with clinical labels.

### 5. Zero-Shot Benchmark

**Purpose**: Evaluate generalization to unseen species and germlines.

**Metrics**:
- Perplexity by species (including rare species)
- Perplexity by germline family
- Liability prediction metrics by species
- Performance gap (common vs rare species)

**Visualizations**:
- Bar plot: perplexity by species (color-coded common vs rare)
- Bar plot: perplexity by germline family
- Bar plots: liability metrics by species
- Pie chart: species distribution

**Usage**:
```python
config = BenchmarkConfig(data_path="data/processed/zero_shot")
benchmark = registry.create("zero_shot", config)
result = benchmark.run(model)
```

## Configuration

Benchmarks are configured via `configs/benchmarks.yaml`:

```yaml
# Global settings
output_dir: ./outputs/benchmarks
batch_size: 32
mlflow_tracking: true

# Per-benchmark configuration
perplexity:
  data_path: ./data/processed/oas
  batch_size: 32
  max_samples: null

liability:
  data_path: ./data/processed/oas
  risk_stratification:
    enabled: true
    thresholds: [33, 66]

# ... more configurations
```

## MLflow Integration

Benchmarks automatically log results to MLflow when enabled:

- **Metrics**: All numeric metrics logged as MLflow metrics
- **Artifacts**: All plots and JSON reports logged as artifacts
- **Experiment**: Results logged to "abprop-benchmarks" experiment

Disable MLflow logging:

```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --no-mlflow
```

## Adding New Benchmarks

1. **Create benchmark class**:

```python
from abprop.benchmarks.registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark

@register_benchmark("my_benchmark")
class MyBenchmark(Benchmark):
    def load_data(self) -> DataLoader:
        # Load your data
        pass

    def evaluate(self, model, dataloader) -> Dict[str, Any]:
        # Run evaluation
        pass

    def report(self, results) -> BenchmarkResult:
        # Generate report
        pass
```

2. **Import in `__init__.py`**:

```python
from . import my_benchmark
```

3. **Add configuration**:

```yaml
my_benchmark:
  data_path: ./data/processed/my_data
  batch_size: 32
```

4. **Run**:

```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks my_benchmark
```

## Parallel Execution

Run benchmarks in parallel (experimental):

```bash
python scripts/run_benchmarks.py \
    --checkpoint path/to/checkpoint.pt \
    --all \
    --parallel
```

**Note**: Parallel execution may have GPU memory issues. Use with caution.

## Output Structure

```
outputs/benchmarks/
├── summary.json                    # Overall summary
├── report.html                     # HTML report (if --html-report)
├── perplexity/
│   ├── metrics.json
│   ├── perplexity_by_chain.png
│   ├── perplexity_by_species.png
│   └── ...
├── cdr_classification/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── ...
├── liability/
│   ├── metrics.json
│   ├── scatter_plots.png
│   └── ...
└── ...
```

## Best Practices

1. **Use test splits**: Always evaluate on held-out test data
2. **Set max_samples for debugging**: Use `--max-samples 100` for quick testing
3. **Monitor GPU memory**: Reduce batch size if OOM errors occur
4. **Track with MLflow**: Enable MLflow tracking to compare across model versions
5. **Generate HTML reports**: Use `--html-report` for shareable results

## Troubleshooting

**Issue**: Out of memory errors

**Solution**: Reduce batch size:
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --batch-size 16
```

**Issue**: Missing data path

**Solution**: Update `configs/benchmarks.yaml` with correct data paths

**Issue**: Benchmark not found

**Solution**: Check that benchmark is imported in `__init__.py` and registered with `@register_benchmark`

## Citation

If you use the AbProp benchmark suite in your research, please cite:

```bibtex
@software{abprop_benchmarks,
  title={AbProp Benchmark Suite},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/AbProp}
}
```

## License

Same as AbProp main project.
