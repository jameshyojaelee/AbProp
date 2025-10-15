# Case Study: Failure Analysis & Interpretability

## Overview
> **Note**: Placeholder insights; refresh after logging real failure cases.

- **Goal**: Diagnose sequences where AbProp liability predictions diverge from assays.
- **Data**: 12 antibody sequences with high lab-measured viscosity yet low predicted liability.
- **Artifacts**: `outputs/attention/failure/aggregated/layer_02_mean.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`, `outputs/eval/error_samples.json`.

## Investigation
1. Reviewed uncertainty estimates (`--uncertainty --mc-samples 64`) to confirm overconfidence.
2. Inspected attention rollout heatmaps for each failure sequence.
3. Checked dataset provenance for missing liabilities or mislabeled species.

## Findings (Placeholder)
- Attention concentrated on framework region while ignoring glycosylation motif in CDR3.
- Embedding coordinates located near sparse region dominated by camelid sequences.
- Uncertainty remained low, indicating blind spot in training distribution.

## Remediation Plan
- Augment training set with glycosylation-heavy examples.
- Add dashboard toggle to highlight sequences with high assay-per-model disagreement.
- Revisit token labeling to ensure CDR3 annotations accurately capture motifs.
