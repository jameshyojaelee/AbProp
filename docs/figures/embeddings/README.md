# Embedding Gallery

This directory captures dimensionality-reduced embeddings for AbProp checkpoints. Scatter and density plots are grouped by reducer (`umap`, `pca`, `tsne`) and dimensionality (`2d`, `3d`). Each checkpoint (or external embedding source) gets its own subfolder alongside comparison overlays.

## Workflow Snapshot

1. Install visualization extras: `pip install -e '.[viz]'`
2. Run the visualization CLI (example):
   ```bash
   python scripts/visualize_embeddings.py \
     --checkpoints outputs/real_data_run/checkpoints/early.pt outputs/real_data_run/checkpoints/best.pt \
     --labels early best \
     --parquet data/processed/oas_real_full \
     --splits val \
     --reducers umap pca \
     --dimensions 2 3 \
     --color-fields species chain germline_v liability_nglyc_bucket \
     --pooling mean \
     --output docs/figures/embeddings \
     --interactive
   ```
3. Review `embedding_metrics.json` for silhouette scores and nearest-neighbor accuracies across metadata fields.
4. Update the highlights below after each new run.

## Highlights

- _2024-XX-YY (pending run)_: Fill in insights after generating fresh plots.
  - Example prompts to consider:
    - Do heavy vs. light chains form separable clusters?
    - Where do high-liability sequences concentrate relative to low-risk ones?
    - Are there clear species- or germline-specific manifolds?
    - Which samples surface as outliers in `comparison` overlays?

Documenting these observations helps maintain a running log for reports, slide decks, and regression tracking.
