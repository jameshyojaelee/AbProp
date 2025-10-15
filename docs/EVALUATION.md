# Evaluation & Interpretability

## Benchmarking

1. Prepare benchmark config: `configs/benchmarks.yaml`
2. Run suite:
   ```bash
   python scripts/run_benchmarks.py \
     --checkpoint outputs/real_data_run/checkpoints/best.pt \
     --config configs/benchmarks.yaml \
     --output benchmarks/results/latest.json
   ```
3. Guard against regressions:
   ```bash
   python scripts/check_regression.py \
     --new benchmarks/results/latest.json \
     --reference benchmarks/results/baseline_example.json \
     --max-drop 0.05
   ```
4. Update [LEADERBOARD.md](../LEADERBOARD.md) and archive JSON snapshots under `benchmarks/results/`.

## Uncertainty

Use Monte Carlo dropout and ensembles during evaluation:
```bash
python scripts/eval.py \
  --checkpoint outputs/real_data_run/checkpoints/best.pt \
  --splits val test \
  --uncertainty --mc-samples 64 \
  --ensemble-checkpoints "runs/fold_*/checkpoints/best.pt" \
  --temperature-calibration
```
Outputs include:
- `uncertainty.json` with coverage vs. error curves
- Per-sequence variance stored alongside liability predictions

## Attention & Embeddings

- Attention introspection: `scripts/visualize_attention.py` (produces PNG + HTML, caches tensors)
- Embedding analysis: `scripts/visualize_embeddings.py` (UMAP/PCA/t-SNE, silhouette/NN accuracy, HTML overlays)
- Drop curated figures in `docs/figures/attention/` and `docs/figures/embeddings/`

## Dashboard & Demo

- Streamlit dashboard: `python scripts/launch_dashboard.py --root outputs`
  - Overview, benchmark explorer, attention explorer, embedding explorer, prediction sandbox, error browser, checkpoint comparison
  - `ABPROP_DASHBOARD_ROOT` / `ABPROP_DASHBOARD_CONFIG` environment variables override paths
- Gradio demo: `python demo/app.py`
  - Supports FASTA input, liabilities, CDR highlighting, attention summaries, MC-dropout uncertainty, CSV/PDF export

## Case Studies

Each case study links back to evaluation artifacts and uncertainty insights. Start with [docs/case_studies/README.md](case_studies/README.md) and follow the template when adding new narratives.
