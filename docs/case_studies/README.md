# AbProp Case Studies

This directory collects narrative deep dives to accompany the main results report.

| Case Study | Focus | Key Artifacts |
|------------|-------|---------------|
| [Developability Risk Assessment](developability.md) | Liability regression vs. wet-lab assays | `docs/figures/publication/ablations.pdf`, `outputs/attention/success/` |
| [CDR Annotation Consistency](cdr_annotation.md) | Token-level classification | `outputs/eval/cdr_report.json`, `outputs/attention/success/` |
| [Production QC Monitoring](qc_monitoring.md) | Regression guardrails | `benchmarks/results/*.json`, `docs/figures/publication/benchmark_comparison.pdf` |
| [Humanization Pathways](humanization.md) | Mutation proposal evaluation | `outputs/attention/failure/`, `docs/figures/embeddings/` |
| [Failure Analysis & Interpretability](interpretability_failure.md) | Error diagnosis & uncertainty | `outputs/eval/error_samples.json`, `outputs/attention/failure/` |

> Metrics currently contain placeholders; update each document after running the full evaluation suite. Use these templates to record observations, link artifacts, and track follow-up actions.
