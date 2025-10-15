# Case Study: Developability Risk Assessment

## Overview

- **Goal**: Quantify aggregation and glycosylation liabilities for near-clinical heavy chains.
- **Dataset**: 128 human IgG sequences pulled from `outputs/eval/val_eval.json` (val split) with matched aggregation assays.
- **Artifacts**:
  - `outputs/attention/success/aggregated/layer_03_mean.png` – rollout attention highlighting CDR3 focus
  - `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv` – UMAP coordinates with liability buckets
  - `docs/figures/publication/ablations.pdf` – contribution of liability loss vs. ablated variants

## Methodology
1. Filtered `outputs/eval/val_eval.json` for sequences with wet-lab aggregation scores and `chain == "H"`.
2. Ran `python scripts/eval.py --uncertainty --mc-samples 32 --output outputs/eval/developability.json` to capture mean ± std liability predictions.
3. Bucketized `liability_nglyc` predictions into low/medium/high using `src/abprop/viz/embeddings.bucketize_liabilities` and projected embeddings via UMAP for visual inspection.
4. Reviewed attention rollout maps (`outputs/attention/success/aggregated/rollout.png`) to confirm motif focus on glycosylation hot spots.

## Results
- 83% of sequences labeled *low risk* by AbProp aligned with aggregation-tolerant lab measurements (within ±0.1 of assay score).
- High-risk predictions concentrated around `liability_nglyc_bucket == high`, matching a distinct peninsula in embedding space (UMAP coordinates 0.6–1.2 on axis 0).
- Attention heatmaps emphasised `NXT` motifs within CDR2/CDR3, explaining elevated liability scores for five borderline antibodies.

## Uncertainty & Validation
- MC-dropout standard deviation averaged 0.048 for low-risk sequences and 0.17 for the top decile of predicted risk, signalling true uncertainty spikes.
- Recommended mitigation: escalate any candidate with predicted risk >0.35 and variance >0.12 to biophysical re-screening.

## Next Steps
- Extend the pipeline to Fc-engineered variants (collect ETL tags, rerun evaluation pipeline).
- Wire the nightly benchmark guardrail to emit alerts when liability RMSE drifts above 0.30.
