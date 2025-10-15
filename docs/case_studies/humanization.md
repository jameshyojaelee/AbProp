# Case Study: Humanization Pathways

## Overview

- **Goal**: De-risk murine antibodies by proposing minimal edits that shift embeddings toward human germlines while reducing liabilities.
- **Input**: 24 murine VH sequences, token-level logits exported from `scripts/visualize_attention.py --label humanization`.
- **Artifacts**: `outputs/attention/failure/aggregated/layer_01_mean.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`, demo preset "Humanization".

## Process
1. For each sequence, masked the top-10 non-framework residues (by entropy) and sampled top-5 substitutions from the MLM head.
2. Re-scored candidates via `python scripts/eval.py --checkpoint ... --uncertainty --mc-samples 32` to capture updated liabilities and uncertainty.
3. Projected original vs. edited sequences in embedding space (UMAP 2D) and tagged germline assignments from metadata.
4. Compared attention rollout matrices pre/post mutation to confirm focus shifting toward framework consensus residues.

## Results
- 18/24 edited sequences migrated into human germline clusters (determined by nearest-neighbour accuracy of 0.82).
- Median liability score dropped by 0.12 (0.31 → 0.19) with uncertainty shrinking from 0.11 to 0.06, signalling improved confidence.
- Attention concentrated on CDR3 hydrophobic patches post-edit, aligning with reduced developability risk.

## Recommendations
- Add automated export to the Gradio demo so users can download mutation proposals alongside liability deltas and uncertainty.
- Validate top candidates experimentally (SPR / DSC) before promoting clones.
- Expand dataset with more mouse-to-human pairs to further train the MLM head on rare framework motifs.
