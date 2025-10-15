## Simple Baseline Comparisons

### Publication Figures Pipeline

Generate reproducible plots for papers and slide decks with:

```bash
python scripts/generate_paper_figures.py \
  --style configs/publication.mplstyle \
  --output docs/figures/publication \
  --figures all \
  --training-metrics outputs/real_data_run/logs/training_metrics.csv \
  --benchmark-results outputs/benchmarks/summary.csv \
  --attention-dir outputs/attention/success/aggregated \
  --embedding-csv docs/figures/embeddings/umap_2d/comparison/scatter_source.csv \
  --error-json outputs/eval/val_errors.json \
  --ablation-csv outputs/ablations/results.csv
```

Each figure is exported as PDF + PNG under `docs/figures/publication/`. Update paths as needed and archive the generated figures in version control when promoting results.

### Case Study Index

Detailed narratives live under [`docs/case_studies/`](case_studies/README.md). Use them to capture developability dives, CDR reviews, QC incidents, humanization experiments, and failure analyses with links back to raw artifacts.

To contextualise AbProp’s performance, we evaluate a suite of lightweight
heuristics that require no gradient updates or heavy training:

| Benchmark | Baselines |
|-----------|-----------|
| **Perplexity** | Uniform token distribution, unigram language model, bigram language model |
| **CDR classification** | Random labelling, frequency-based majority vote, k-NN (AA count vectors) |
| **Liability regression** | Per-liability mean, nearest-neighbour copy, motif heuristics |

### Workflow

1. **Execution**  
   ```bash
   python scripts/eval_baselines.py \
       --benchmarks perplexity cdr_classification liability \
       --output outputs/baselines \
       --abprop-results outputs/eval
   ```
   This produces:
   - `baseline_results.json` with per-baseline metrics and significance estimates.
   - `baseline_summary.csv` for quick tabular inspection.

2. **Visualisation**  
   ```bash
   python scripts/plot_baseline_comparison.py \
       --results outputs/baselines/baseline_results.json \
       --output-dir outputs/baselines/plots
   ```
   The script renders simple bar charts (perplexity ↓, accuracy ↑, MSE ↓).

### Significance Estimates

Where reference AbProp metrics include per-sample signals (e.g., sequence-level
perplexities or confusion matrices), we compute lightweight statistics:

- **Perplexity** – Welch’s t-test between baseline and AbProp sequence
  perplexities.
- **CDR accuracy** – Two-proportion z-test using confusion matrices.
- **Liability** – Currently no paired AbProp residuals are stored; significance
  testing is omitted pending richer logging.

### Caveats

- The heuristics depend on the training split only; ensure data leakage is
  avoided when introducing new baselines.
- K-NN implementations use simple amino acid frequency vectors; more advanced
  embeddings may shift results.
- Motif heuristics emphasise illustrative value over accurate biophysical
  modelling; interpret liability scores qualitatively.

### Reporting

Summaries should highlight:

1. Absolute gap between AbProp and strongest simple baseline per task.
2. Whether differences are statistically significant given current tests.
3. Failure modes where baselines rival AbProp (e.g., scarce data regimes).

These comparisons help demonstrate that AbProp offers meaningful gains beyond
trivial heuristics when presenting experiments or preparing publication-ready
figures.
