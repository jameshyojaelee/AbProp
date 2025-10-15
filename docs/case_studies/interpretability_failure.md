# Case Study: Failure Analysis & Interpretability

## Overview

- **Goal**: Understand why AbProp underestimates liabilities for high-viscosity antibodies.
- **Data**: 12 sequences flagged by lab assays but predicted low risk (`outputs/eval/error_samples.json`).
- **Artifacts**: `outputs/attention/failure/aggregated/layer_02_mean.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`, dashboard preset "Failure Focus".

## Investigation
1. Extracted mispredicted sequences using `scripts/eval.py` residuals and saved them to `outputs/eval/error_samples.json`.
2. Re-ran attention visualization (`scripts/visualize_attention.py --label failure`) to inspect head-by-head behaviour.
3. Examined embedding positions relative to species/germline metadata; computed nearest-neighbour accuracy for the failure cohort.
4. Cross-checked ETL logs to ensure liability annotations were present and species labels correct.

## Findings
- Attention focussed on framework residues, skipping glycosylation motifs in CDR3, suggesting insufficient training coverage for those patterns.
- Failures cluster in a sparse embedding region dominated by camelid sequences, yet the metadata tags identified them as human — pointing to data drift.
- MC-dropout uncertainty remained low (std ~0.04), confirming the model is confidently wrong; this triggered a new guardrail condition (large residual + low variance).

## Remediation Plan
- Augment the training dataset with glycosylation-heavy examples (extend ETL to include additional public datasets).
- Introduce an uncertainty-aware alert: `if abs(residual) > 0.15 and mc_std < 0.05 → flag in dashboard`.
- Validate species labels upstream and consider adding a species-aware loss term to penalize misclustered embeddings.
