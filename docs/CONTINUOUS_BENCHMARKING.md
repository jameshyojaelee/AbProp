# Continuous Benchmarking

This workflow guards against performance regressions by recomputing core benchmarks and comparing them to a stored baseline.

## Quick Start

1. Populate a baseline JSON:
   ```bash
   python scripts/run_benchmarks.py --config configs/benchmarks.yaml --output benchmarks/results/baseline.json
   ```
2. Commit the baseline file and update [LEADERBOARD.md](../LEADERBOARD.md).
3. Trigger the GitHub Action (`Benchmark Regression Guardrail`) manually or wait for the weekly schedule.
4. Inspect uploaded artifacts under the workflow run and promote `benchmarks/results/latest.json` to the leaderboard if results improve.

## Regression Checks

`scripts/check_regression.py` compares two JSON files with structure:

```json
{
  "commit": "<sha>",
  "timestamp": "2024-01-01T00:00:00Z",
  "benchmarks": {
    "perplexity": {"metric": "perplexity", "value": 1.95, "higher_is_better": false}
  }
}
```

Usage:

```bash
python scripts/check_regression.py \
  --new benchmarks/results/latest.json \
  --reference benchmarks/results/baseline.json \
  --max-drop 0.05 \
  --threshold liability_regression=0.1
```

If any metric regresses beyond the configured threshold the script exits with status `1`, causing CI to fail.

## MLflow Logging

Provide `--mlflow-uri` to log benchmark metrics into an experiment (`abprop-benchmarks` by default). This enables long-term trend monitoring alongside leaderboards.

## Directory Layout

```
benchmarks/
└── results/
    ├── baseline_example.json
    ├── latest.json
    └── YYYY-MM-DD.json  # optional historical snapshots
```

Keep historical JSON files committed for provenance; the GitHub Action uploads the most recent run as an artifact.
