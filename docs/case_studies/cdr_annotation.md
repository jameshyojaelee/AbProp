# Case Study: CDR Annotation Consistency

## Overview
> **Note**: Metrics below are placeholders until updated with fresh evaluation runs.

- **Goal**: Validate token-level CDR predictions on paired AIRR sequences.
- **Dataset**: 96 sequences with IMGT-aligned CDR annotations stored in `data/processed/oas_real_full`.
- **Artifacts**:
  - `outputs/eval/cdr_report.json`
  - `outputs/attention/success/aggregated/layer_02_mean.png`
  - `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`

## Methodology
1. Loaded held-out sequences via `scripts/eval.py --splits val --tasks cls`.
2. Compared predicted vs. reference CDR spans (precision/recall/F1).
3. Reviewed attention heatmaps for misclassified residues.

## Findings (Placeholder)
- Macro F1: 0.92 for heavy chains, 0.89 for light chains.
- Misclassifications cluster around framework-CDR boundaries with low experimental agreement.
- Attention focuses on motif boundaries (e.g., `YYC`) enabling manual correction.

## Recommendations
- Augment training with species-specific CDR templates.
- Add dashboard preset filter to surface low-confidence boundary cases.
