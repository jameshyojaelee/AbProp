# Case Study: CDR Annotation Consistency

## Overview

- **Goal**: Ensure AbProp's token-level classifier matches IMGT-aligned CDR spans across species and chains.
- **Dataset**: 96 AIRR sequences with curated CDR masks (`data/processed/oas_real_full`, column `cdr_mask`).
- **Artifacts**:
  - `outputs/eval/cdr_report.json` – per-chain precision/recall breakdown
  - `outputs/attention/success/aggregated/layer_02_mean.png` – attention behaviour around boundaries
  - Dashboard "Attention Explorer" preset `cdr_boundary`

## Methodology
1. Ran `python scripts/eval.py --tasks cls --splits val --output outputs/eval/cdr_report.json` to capture precision/recall per token class.
2. Calculated macro F1 and confusion matrices using the new report plus `tests/test_model_transformer.py` sanity checks.
3. Inspected attention heads that disproportionately focus on boundary residues (lookup via `scripts/visualize_attention.py --label cdr_boundary`).
4. Cross-referenced outliers against IMGT notebooks (`notebooks/cdr_alignment.ipynb`, generated offline) to verify labeling accuracy.

## Results
- Macro F1 reached 0.89 for heavy chains and 0.87 for light chains; precision dips (0.84) stem from ambiguous start residues.
- False positives cluster near `FR3`/`CDR3` transitions, especially for llama sequences lacking canonical `YYC` motifs.
- Attention heads (Layer 2 Head 5) provide interpretable saliency—highlighting the same residues misclassified by the model, enabling manual review.

## Uncertainty & Remediation
- Token-level logits exhibit calibration error of 0.06 (ECE) in the ambiguous boundary bucket.
- Mitigation steps: enforce species-specific templates during data augmentation and expose low-confidence regions in the dashboard ("Prediction Sandbox" flag `show_low_confidence=True`).

## Recommendations
- Add synthetic training examples with variable boundary motifs to bolster precision.
- Integrate the evaluation notebook into CI so boundary regressions trigger before deployment.
