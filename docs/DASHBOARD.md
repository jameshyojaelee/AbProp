# AbProp Dashboard

A Streamlit dashboard aggregates training metrics, attention studies, embedding projections, and ad-hoc predictions for stakeholders.

## Installation

Install optional dependencies:

```bash
pip install -e '.[dashboard,viz]'
```

## Launching

```bash
python scripts/launch_dashboard.py \
  --root outputs \
  --config configs/dashboard.local.json \
  --port 8501
```

The launcher exports:

- `ABPROP_DASHBOARD_ROOT`: base directory containing runs, benchmarks, eval reports, attention/embedding artifacts, and checkpoints.
- `ABPROP_DASHBOARD_CONFIG`: optional JSON mapping (`runs_dir`, `benchmarks_dir`, `attention_dir`, `embeddings_dir`, `eval_dir`, `checkpoints_dir`).

Example `configs/dashboard.local.json`:

```json
{
  "runs_dir": "outputs/real_data_run",
  "benchmarks_dir": "benchmarks/results",
  "attention_dir": "outputs/attention",
  "embeddings_dir": "docs/figures/embeddings",
  "eval_dir": "outputs/evals",
  "checkpoints_dir": "outputs/real_data_run/checkpoints"
}
```

## Pages

- **Overview** – high-level counts and onboarding tips.
- **Benchmarks** – tabular benchmark JSON summaries with CSV export.
- **Attention Explorer** – browse attention heatmaps produced by `scripts/visualize_attention.py` and download figures.
- **Embedding Explorer** – surface `embedding_metrics.json` statistics and comparison plots from `scripts/visualize_embeddings.py`.
- **Prediction Sandbox** – run single-sequence inference against a selected checkpoint (requires PyTorch + checkpoint).
- **Error Browser** – inspect evaluation JSON payloads.
- **Checkpoint Comparison** – diff metric snapshots saved in checkpoints.

## Sample Data Bundles

Populate the dashboard by running the visualization scripts and copying outputs under `outputs/` (or the root you pass to the launcher). Suggested layout:

```
outputs/
├── attention/
│   └── success/
│       └── aggregated/
├── embeddings/
│   ├── umap_2d/
│   └── embedding_metrics.json
├── benchmarks/
│   └── developability_val.json
├── eval/
│   └── val_eval.json
└── checkpoints/
    ├── early.pt
    └── best.pt
```

## Notes

- Streamlit caches heavy loading operations (`@st.cache_data` / `@st.cache_resource`). Update or clear cache via the Streamlit sidebar if files change.
- Set `--headless` on the launcher when embedding the dashboard into CI artifacts.
- Capture screenshots or GIFs (e.g., with `streamlit-webrtc` or native tools) and place them under `docs/figures/dashboard/` for slides.
