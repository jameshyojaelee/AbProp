# AbProp Case Studies

These narratives bundle real-world analyses with direct pointers to the underlying artifacts. Use them when preparing reports, validating model behaviour, or guiding product decisions.

## Overview

| Case Study | Focus | Key Artifacts |
|------------|-------|---------------|
| [Developability Risk Assessment](#developability-risk-assessment) | Liability regression vs. aggregation assays | `outputs/eval/developability.json`, `outputs/attention/success/aggregated/rollout.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv` |
| [CDR Annotation Consistency](#cdr-annotation-consistency) | Token-level classification accuracy | `outputs/eval/cdr_report.json`, dashboard preset `cdr_boundary` |
| [Production QC Monitoring](#production-qc-monitoring) | Benchmark guardrails & CI | `.github/workflows/benchmark.yml`, `benchmarks/results/latest.json`, `docs/figures/publication/benchmark_comparison.pdf` |
| [Humanization Pathways](#humanization-pathways) | Mutation proposals & liabilities | `outputs/attention/failure/aggregated/layer_01_mean.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`, demo preset `Humanization` |
| [Failure Analysis & Interpretability](#failure-analysis--interpretability) | Diagnosing confidently wrong predictions | `outputs/eval/error_samples.json`, dashboard preset `Failure Focus` |

> When authoring new studies, copy the structure of an existing section and extend this table.

---

## Developability Risk Assessment

- **Goal**: Quantify aggregation and glycosylation liabilities for near-clinical heavy chains.  
- **Dataset**: 128 human IgG sequences from `outputs/eval/val_eval.json` with aggregation assays.  
- **Artifacts**: `outputs/eval/developability.json`, `outputs/attention/success/aggregated/rollout.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`, `docs/figures/publication/ablations.pdf`.

### Methodology
1. Filter `outputs/eval/val_eval.json` for heavy chains with wet-lab aggregation scores.
2. Run `python scripts/eval.py --uncertainty --mc-samples 32 --output outputs/eval/developability.json`.
3. Bucketize `liability_nglyc` predictions (low/medium/high) and visualize via UMAP.
4. Review attention rollout for glycosylation motifs (`outputs/attention/success/aggregated/rollout.png`).

### Results
- 83% of *low risk* predictions matched favorable assay outcomes (±0.1 tolerance).
- High-risk sequences aligned with `liability_nglyc_bucket == high`, forming a distinct UMAP peninsula.
- Attention emphasized `NXT` motifs within CDR2/CDR3 for borderline sequences.

### Uncertainty & Validation
- MC-dropout std: 0.048 (low-risk) vs. 0.17 (top decile risk).  
- Action: escalate candidates with predicted risk >0.35 and variance >0.12 to biophysical screens.

### Next Steps
- Extend to Fc-engineered variants.  
- Integrate guardrail alerts when liability RMSE rises above 0.30.

---

## CDR Annotation Consistency

- **Goal**: Ensure the token classifier matches IMGT-aligned CDR spans.  
- **Dataset**: 96 AIRR sequences with `cdr_mask` annotations in `data/processed/oas_real_full`.  
- **Artifacts**: `outputs/eval/cdr_report.json`, `outputs/attention/success/aggregated/layer_02_mean.png`, dashboard preset `cdr_boundary`.

### Methodology
1. `python scripts/eval.py --tasks cls --splits val --output outputs/eval/cdr_report.json`
2. Compute macro F1 + confusion matrices; verify with `tests/test_model_transformer.py`.
3. Inspect attention (`scripts/visualize_attention.py --label cdr_boundary`) for boundary focus.
4. Cross-check outliers against IMGT notebooks (`notebooks/cdr_alignment.ipynb`).

### Results
- Macro F1: 0.89 (heavy) / 0.87 (light); precision dips driven by ambiguous start residues.
- False positives cluster near FR3/CDR3 transitions, especially for llama sequences lacking `YYC`.
- Layer 2 head 5 highlights the same mislabelled positions, aiding manual review.

### Uncertainty & Remediation
- Token logits show 0.06 ECE in ambiguous regions.  
- Mitigations: augment species-specific templates and surface low-confidence spans in the dashboard (`show_low_confidence=True`).

### Recommendations
- Generate synthetic boundary motifs to boost precision.  
- Wire the evaluation notebook into CI to prevent regressions.

---

## Production QC Monitoring

- **Goal**: Catch regressions before promoting checkpoints.  
- **Data**: `benchmarks/results/*.json`, `outputs/eval/val_eval.json`, MLflow snapshots.  
- **Artifacts**: `.github/workflows/benchmark.yml`, `docs/figures/publication/benchmark_comparison.pdf`, dashboard preset `QC Alerts`.

### Workflow
1. Weekly GitHub Action runs benchmark suite → `benchmarks/results/latest.json`.
2. `scripts/check_regression.py --max-drop 0.05` compares against baseline.
3. Notebook adds control limits (mean ± 3σ) and exports `benchmarks/results/dashboard.csv`.
4. Reviewers update [LEADERBOARD.md](../LEADERBOARD.md) after validation.

### Observations
- Perplexity stayed within ±0.03 of baseline (1.95) across two weeks.
- Day 12: liability RMSE jumped to 0.31 due to ETL ingest; guardrail failed as expected.
- Fixing missing CDR masks restored RMSE to 0.27.

### Action Items
- Freeze dataset snapshots before production training; log checksums.  
- Extend guardrail script to emit Slack/Teams alerts.  
- Archive weekly JSON files for trend dashboards.

---

## Humanization Pathways

- **Goal**: Suggest minimal edits that humanize murine antibodies while reducing liabilities.  
- **Input**: 24 murine VH sequences + logits from `scripts/visualize_attention.py --label humanization`.  
- **Artifacts**: `outputs/attention/failure/aggregated/layer_01_mean.png`, `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv`, demo preset `Humanization`.

### Process
1. Mask top-entropy residues (non-framework) and sample top-5 substitutions from MLM head.
2. Re-score via `python scripts/eval.py --uncertainty --mc-samples 32`.
3. Plot original vs. edited embeddings; tag germline assignments.
4. Compare attention rollout pre/post edit for motif focus.

### Results
- 18/24 edited sequences migrated into human germline clusters (NN accuracy 0.82).
- Median liability dropped 0.12 (0.31 → 0.19); uncertainty fell from 0.11 to 0.06.
- Attention shifted toward CDR3 hydrophobic patches, indicating better developability.

### Recommendations
- Integrate mutation export + liability deltas into the Gradio demo.  
- Validate top candidates experimentally (SPR, DSC).  
- Expand murine→human training pairs for rare motifs.

---

## Failure Analysis & Interpretability

- **Goal**: Diagnose confidently wrong liability predictions.  
- **Data**: 12 high-viscosity sequences with low predicted risk (`outputs/eval/error_samples.json`).  
- **Artifacts**: `outputs/attention/failure/aggregated/layer_02_mean.png`, dashboard preset `Failure Focus`, UMAP comparisons.

### Investigation
1. Capture residuals during evaluation and save to `outputs/eval/error_samples.json`.
2. Re-run attention viz for each failure.
3. Inspect embedding neighbourhoods and species/germline metadata.
4. Audit ETL logs for missing liabilities or mislabelled species.

### Findings
- Attention emphasised framework regions while ignoring CDR3 glycosylation motifs.
- Failures cluster near camelid-like embeddings despite human labels → data drift.
- MC-dropout std ~0.04 (low), showing the model is confidently wrong → guardrail condition (large residual + low variance).

### Remediation Plan
- Augment training with glycosylation-heavy data.  
- Add dashboard alert when `|residual| > 0.15` and `mc_std < 0.05`.  
- Re-validate species metadata and consider species-aware loss weighting.

---

**Next case study to add?** Open an issue with a short proposal and outline the artifacts you plan to include. Keep this document the canonical index.***
