# Case Study: Production QC Monitoring

## Overview
> **Note**: Replace placeholder values after running continuous benchmarking pipelines.

- **Goal**: Monitor inference quality for production-bound antibodies.
- **Data Sources**: `benchmarks/results/*.json`, `outputs/eval/val_eval.json`.
- **Artifacts**: `docs/figures/publication/benchmark_comparison.pdf`, `benchmarks/results/dashboard.csv`.

## Workflow
1. Automated nightly benchmark via Prompt 20 outputs dataset `benchmarks/results/YYYY-MM-DD.json`.
2. Compared latest metrics against control limits (mean ± 3σ of prior month).
3. Flagged sequences that triggered QC alerts for follow-up review.

## Observations (Placeholder)
- No regressions detected over past 14 days (max Δ perplexity = +0.03).
- Liability RMSE drifted upward on day 12, correlating with new training data ingest.
- Dashboard preset "QC Alerts" surfaces runs breaching thresholds.

## Action Items
- Freeze training data snapshot when RMSE drift occurs; rerun with ablations.
- Incorporate automated Slack notifications by extending `scripts/check_regression.py` (Prompt 20).
