# AbProp Case Studies

This directory collects narrative deep dives to accompany the main results report. Follow [TEMPLATE.md](TEMPLATE.md) when adding new stories.

| Case Study | Focus | Key Artifacts |
|------------|-------|---------------|
| [Developability Risk Assessment](developability.md) | Liability regression vs. wet-lab assays | `docs/figures/publication/ablations.pdf`, `outputs/attention/success/`, `outputs/eval/developability.json` |
| [CDR Annotation Consistency](cdr_annotation.md) | Token-level classification | `outputs/eval/cdr_report.json`, `outputs/attention/success/` |
| [Production QC Monitoring](qc_monitoring.md) | Regression guardrails | `benchmarks/results/*.json`, `docs/figures/publication/benchmark_comparison.pdf` |
| [Humanization Pathways](humanization.md) | Mutation proposal evaluation | `outputs/attention/failure/`, `docs/figures/embeddings/` |
| [Failure Analysis & Interpretability](interpretability_failure.md) | Error diagnosis & uncertainty | `outputs/eval/error_samples.json`, `outputs/attention/failure/` |

Link notebooks alongside the markdown files whenever possible to keep the narratives reproducible.
