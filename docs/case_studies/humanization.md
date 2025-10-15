# Case Study: Humanization Pathways

## Overview
> **Note**: Update metrics after running focused humanization experiments.

- **Goal**: Evaluate AbProp-guided edits for murine-to-human conversion.
- **Input**: 24 murine VH sequences + suggested mutations from AbProp token logits.
- **Artifacts**: `outputs/attention/failure/aggregated/layer_01_mean.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`.

## Process
1. Generated top-5 mutation proposals using token-level logits and masked language modeling.
2. Scored each variant for liabilities and framework compliance.
3. Reviewed attention changes pre/post humanization for interpretability.

## Findings (Placeholder)
- 18/24 sequences moved into human germline clusters in embedding space after edits.
- Liability scores improved by median 0.12 (relative units).
- Attention pivoted from murine framework motifs to human consensus residues.

## Recommendations
- Integrate humanization preset into dashboard prediction sandbox.
- Pair with experimental validation (surface plasmon resonance) before progressing to candidate selection.
