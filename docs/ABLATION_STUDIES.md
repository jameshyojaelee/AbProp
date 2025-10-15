# Ablation Studies

This document captures the structure, execution workflow, and reporting
guidelines for the automated AbProp ablation suite.

## Overview

The ablation tooling explores how architectural choices, optimisation
hyperparameters, data availability, and masking strategies affect AbProp's
performance. The workflow comprises two stages:

1. **Sweep execution** – `scripts/run_ablations.py` materialises per-experiment
   configuration files, launches training, and records metadata in
   `outputs/ablations/<plan>/<experiment>/`.
2. **Post-hoc analysis** – `scripts/analyze_ablations.py` aggregates CSV/JSON
   artefacts, computes Pareto-efficient candidates, and produces ready-to-share
   tables/plots.

Plan definitions live in `configs/ablations/`. Each declarative file lists
experiments with nested overrides applied to the base train/model/data configs.

## Running a Sweep

```bash
# 1. Inspect or edit the plan
cat configs/ablations/plan.yaml

# 2. Launch the sweep (dry-run first if desired)
python scripts/run_ablations.py --plan configs/ablations/plan.yaml --dry-run
python scripts/run_ablations.py --plan configs/ablations/plan.yaml
```

### Key Options

```
--filter EXP1 EXP2     # limit to specific experiments
--skip-existing        # skip runs with existing metadata
--max-runs N           # cap the number of executions
```

Each run writes:

- `configs/train.yaml` / `model.yaml` / `data.yaml` – resolved configurations
- `artifacts/` – training outputs (logs, checkpoints, metrics)
- `metadata.json` – command, status, timings, tags
- `summary.json` – quick metrics (final loss, min loss, step time, etc.)

## Analysing Results

```bash
python scripts/analyze_ablations.py \
    --results outputs/ablations \
    --summary-csv outputs/ablations/summary.csv \
    --summary-json outputs/ablations/summary.json \
    --plot outputs/ablations/loss_chart.png \
    --metric metrics_loss
```

The analyser:

- Aggregates all runs and writes a unified CSV/JSON summary
- Computes a Pareto front (loss vs. wall-clock duration by default)
- Optionally emits plots (requires Matplotlib)

## Reading the Leaderboard

The generated CSV includes:

| Column | Description |
|--------|-------------|
| `plan` | Plan identifier (e.g., `baseline_sweep`) |
| `experiment` | Sanitised experiment name |
| `status` | `completed`, `failed`, `skipped`, or `interrupted` |
| `metrics_loss` | Latest logged training loss |
| `loss_min` | Minimum observed training loss |
| `duration_seconds` | Total runtime of the training command |
| `step_time_avg` | Mean step time (if logged) |
| `tag_*` | Flattened MLflow tags/metadata |

Use the Pareto listing to identify configurations that balance runtime and
loss reduction.

## Findings Checklist

When documenting results, capture:

1. **Model scaling** – impact of hidden dimension/layer depth on loss and
   convergence speed.
2. **Task weight shifts** – how re-weighting MLM/CLS/REG affects the respective
   metrics.
3. **Data regime sensitivity** – behaviour under 25%, 50%, and full-data
   regimes.
4. **Masking probability** – effect on MLM perplexity and downstream tasks.
5. **Optimiser swaps** – stability/runtime differences between AdamW variants
   and SGD + momentum.

Summarise the top-performing configuration (loss/time/notes) and note any
regressions relative to the `baseline_reference`.

## Recommended Defaults (Template)

- **Architecture**: `d_model=384`, `num_layers=3` (baseline) with optional
  bump to `d_model=512` if compute permits.
- **Task weights**: Balanced unless liability metrics are prioritised; in that
  case use the `regression_focus` variant.
- **Masking**: Retain 15% unless training logs show instability, where 10% is
  safer.
- **Optimiser**: AdamW (`lr=1e-4`, `weight_decay=1e-2`). SGD with momentum is
  slower to converge in most trials.

> Replace the placeholders above with real numbers/observations after running
> the sweep on your infrastructure.

## Troubleshooting

- **Missing metrics**: Ensure `artifacts/logs/abprop-train.csv` exists. If not,
  re-run the experiment with `--dry-run-steps 50` to validate the pipeline.
- **Dataset errors**: Plans can override `train_fraction` / `val_fraction`
  without touching dataset paths. Use dataset-specific copies if you need
  alternative splits.
- **MLflow tags**: Custom tags are injected automatically (`ablation_plan`,
  `ablation_experiment`). Additional tags can be set per experiment within the
  declarative YAML.

## Next Steps

- Enrich the plan with additional axes (e.g., gradient checkpointing, precision
  modes).
- Push summaries to dashboards or MLflow runs for long-term tracking.
- Integrate benchmark evaluation (`scripts/eval.py`) post-training to include
  validation metrics beyond training loss.

