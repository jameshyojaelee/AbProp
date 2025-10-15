# Case Study: Developability Risk Assessment

## Overview
> **Note**: Replace placeholder metrics with real numbers after rerunning analyses.

- **Goal**: Assess AbProp's liability regressors on therapeutic-like heavy chains.
- **Dataset**: 128 human IgG sequences sampled from `outputs/eval/val_eval.json`.
- **Artifacts**:
  - `docs/figures/publication/ablations.pdf` (model ablation summary)
  - `outputs/attention/success/aggregated/layer_03_mean.png` (attention focus on CDR3)
  - `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv` (UMAP coordinates)

## Methodology
1. Selected sequences with experimental aggregation assays.
2. Ran `abprop-eval --uncertainty --mc-samples 32` to estimate liability scores + confidence intervals.
3. Mapped predictions to interval buckets (low/medium/high) and contrasted with lab readouts.

## Findings
- Placeholder: 87% of sequences flagged as *low risk* matched wet-lab aggregation tolerance.
- Placeholder: High-risk cluster aligns with `liability_nglyc_bucket == high` in embedding space.
- Placeholder: Attention heats show model emphasis on `NXS/T` motifs in CDR2 for high-risk predictions.

## Uncertainty & Validation
- Placeholder: MC-dropout variance remained below 0.05 for low-risk cohort; spiked to 0.18 for borderline antibodies.
- Recommend confirming outliers with biophysical assays before progressing to developability engineering.

## Next Steps
- Extend analysis to Fc-engineered variants.
- Automate weekly regression checks via Prompt 20 benchmark workflow.
