# Case Study: Production QC Monitoring

## Overview

- **Goal**: Detect performance regressions before promoting checkpoints to production scoring.
- **Data Sources**: `benchmarks/results/*.json`, `outputs/eval/val_eval.json`, MLflow metrics snapshot.
- **Artifacts**: `docs/figures/publication/benchmark_comparison.pdf`, GitHub Action logs, dashboard "Benchmark Explorer" table.

## Workflow
1. Scheduled GitHub Action (`.github/workflows/benchmark.yml`) runs every Monday at 06:00 UTC.
2. The workflow copies raw metrics into `benchmarks/results/latest.json` and calls `scripts/check_regression.py --max-drop 0.05` against the baseline.
3. A post-processing notebook (internal) enriches the JSON with control-limit calculations (mean ± 3σ) and exports `benchmarks/results/dashboard.csv` for the dashboard.
4. Reviewers inspect the dashboard preset "QC Alerts" and update [LEADERBOARD.md](../../LEADERBOARD.md) when improvements land.

## Observations
- Over the past two weeks, perplexity remained within ±0.03 of baseline (1.95), confirming language modeling stability.
- Day 12 introduced a liability RMSE spike to 0.31 after ingesting new ETL batches; the guardrail flagged this via exit code 1.
- Triage revealed the new data lacked CDR masks, prompting an ETL fix and a re-run that restored RMSE to 0.27.

## Action Items
- Freeze dataset snapshots before training runs destined for production; log checksums in `data/DATA_PROVENANCE.md`.
- Extend `scripts/check_regression.py` to emit Slack / Teams notifications when guardrails fail.
- Archive weekly JSON files to build a longer-term trend view (feeding MLflow dashboards).
