# AbProp Benchmark Suite - Quick Reference Card

## Quick Commands

### Run All Benchmarks
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all --html-report
```

### Run Specific Benchmarks
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks perplexity liability
```

### Quick Test (Limited Data)
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks perplexity --max-samples 100
```

### CPU Evaluation
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all --device cpu
```

### Custom Configuration
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --config configs/my_benchmarks.yaml --all
```

---

## Available Benchmarks

| Benchmark | Name | Purpose | Metrics |
|-----------|------|---------|---------|
| 🔤 **Perplexity** | `perplexity` | Language modeling quality | Perplexity by chain/species/length |
| 🎯 **CDR Classification** | `cdr_classification` | Token-level CDR prediction | Precision, Recall, F1, Accuracy |
| ⚠️ **Liability** | `liability` | Developability liabilities | MSE, R², Spearman per liability |
| 💊 **Developability** | `developability` | Therapeutic ranking | AUC, Spearman, Correlations |
| 🌍 **Zero-Shot** | `zero_shot` | Generalization to novel data | Perplexity gap, Cross-species |

---

## Programmatic Usage

```python
from abprop.benchmarks import get_registry
from abprop.benchmarks.registry import BenchmarkConfig
from abprop.models import AbPropModel

# Load model
model = AbPropModel.from_checkpoint("path/to/checkpoint.pt")

# Configure benchmark
config = BenchmarkConfig(data_path="data/processed/oas")
registry = get_registry()

# Run benchmark
benchmark = registry.create("perplexity", config)
result = benchmark.run(model)

# Access results
print(f"Perplexity: {result.metrics['overall_perplexity']:.2f}")
```

---

## Output Structure

```
outputs/benchmarks/
├── summary.json              # Overall summary
├── report.html              # HTML report (if --html-report)
├── perplexity/
│   ├── metrics.json
│   └── *.png
├── cdr_classification/
│   ├── metrics.json
│   └── *.png
├── liability/
│   ├── metrics.json
│   └── *.png
└── ...
```

---

## Configuration File

`configs/benchmarks.yaml`:

```yaml
output_dir: ./outputs/benchmarks
batch_size: 32
device: cuda
mlflow_tracking: true

perplexity:
  data_path: ./data/processed/oas
  max_samples: null

liability:
  data_path: ./data/processed/oas
  risk_stratification:
    enabled: true
    thresholds: [33, 66]
```

---

## MLflow Integration

### View Results
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Disable MLflow
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --all --no-mlflow
```

---

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint PATH` | Model checkpoint (required) | - |
| `--config PATH` | Benchmark config | `configs/benchmarks.yaml` |
| `--all` | Run all benchmarks | - |
| `--benchmarks NAME [NAME]` | Specific benchmarks | - |
| `--batch-size INT` | Batch size | 32 |
| `--max-samples INT` | Sample limit (testing) | None |
| `--device STR` | cuda or cpu | cuda |
| `--output-dir PATH` | Output directory | `outputs/benchmarks` |
| `--html-report` | Generate HTML report | False |
| `--no-mlflow` | Disable MLflow | False |
| `--parallel` | Parallel execution | False |

---

## Expected Performance

### Good Model Metrics

| Benchmark | Metric | Good | Baseline | Random |
|-----------|--------|------|----------|--------|
| Perplexity | PPL | < 5.0 | 8-12 | ~20 |
| CDR Classification | F1 | > 0.85 | 0.70-0.80 | ~0.5 |
| Liability | R² | > 0.7 | 0.4-0.6 | 0 |
| Developability | AUC | > 0.75 | 0.55-0.65 | 0.5 |
| Zero-Shot | Gap | < 2.0 | 3-5 | > 8 |

### Runtime (10K sequences, V100)

| Benchmark | Time |
|-----------|------|
| Perplexity | ~5 min |
| CDR Classification | ~5 min |
| Liability | ~6 min |
| Developability | ~2 min |
| Zero-Shot | ~4 min |
| **Total** | **~22 min** |

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --batch-size 16 --benchmarks perplexity
```

### Missing Data
```bash
# Check data path in configs/benchmarks.yaml or run ETL
python scripts/process_real_data_etl.py --input data/raw/sequences.csv --output data/processed/oas
```

### Slow Evaluation
```bash
# Use sample limiting
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --max-samples 1000 --all
```

---

## File Locations

### Core Implementation
- Registry: `src/abprop/benchmarks/registry.py`
- Benchmarks: `src/abprop/benchmarks/*_benchmark.py`
- Runner: `scripts/run_benchmarks.py`

### Configuration
- Config: `configs/benchmarks.yaml`
- Model config: `configs/model.yaml`

### Documentation
- User guide: `docs/BENCHMARK_SUITE.md`
- Technical docs: `src/abprop/benchmarks/README.md`
- Examples: `examples/run_benchmark_example.py`
- Summary: `BENCHMARK_IMPLEMENTATION_SUMMARY.md`

---

## Adding Custom Benchmarks

1. Create `src/abprop/benchmarks/my_benchmark.py`:
```python
from .registry import Benchmark, register_benchmark

@register_benchmark("my_benchmark")
class MyBenchmark(Benchmark):
    def load_data(self): ...
    def evaluate(self, model, dataloader): ...
    def report(self, results): ...
```

2. Import in `src/abprop/benchmarks/__init__.py`:
```python
from . import my_benchmark
```

3. Add config to `configs/benchmarks.yaml`:
```yaml
my_benchmark:
  data_path: ./data/processed/my_data
```

4. Run:
```bash
python scripts/run_benchmarks.py --checkpoint path/to/checkpoint.pt --benchmarks my_benchmark
```

---

## Documentation Links

- 📘 **User Guide**: [docs/BENCHMARK_SUITE.md](docs/BENCHMARK_SUITE.md)
- 🔧 **Technical Docs**: [src/abprop/benchmarks/README.md](src/abprop/benchmarks/README.md)
- 💡 **Examples**: [examples/run_benchmark_example.py](examples/run_benchmark_example.py)
- 📊 **Summary**: [BENCHMARK_IMPLEMENTATION_SUMMARY.md](BENCHMARK_IMPLEMENTATION_SUMMARY.md)

---

## Support

For issues or questions:
1. Check documentation files above
2. Review example usage in `examples/`
3. Examine configuration in `configs/benchmarks.yaml`
4. Test with `--max-samples 100` for quick debugging

---

**Last Updated**: October 13, 2025
**Status**: ✅ Complete
