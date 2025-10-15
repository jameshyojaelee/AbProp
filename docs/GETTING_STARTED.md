# Getting Started with AbProp

This quick guide walks you through environment setup, data preparation, and your first training + evaluation loop.

## 1. Install

```bash
# Clone and enter the repo
# git clone git@github.com:your-org/AbProp.git
# cd AbProp

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev,serve,bench,viz,dashboard]'
```

Prefer conda or Docker? Use `environment.yml` or `Dockerfile` from the project root.

## 2. Fetch Data

1. Review the provenance checklist in `data/DATA_PROVENANCE.md`.
2. Populate `data/raw/` with OAS or SAbDab exports (CSV/FASTA).
3. Run ETL:
   ```bash
   python scripts/process_real_data_etl.py \
     --input data/raw/sabdab_sequences_full.csv \
     --output data/processed/oas_real_full
   ```
4. Update `configs/data.yaml` with the processed directory and parquet file names.

## 3. Train

```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --output-dir outputs/real_data_run
```

What you get:
- Checkpoints under `outputs/real_data_run/checkpoints/`
- Metrics CSVs in `outputs/real_data_run/logs/`
- MLflow entries if `MLFLOW_TRACKING_URI` is set

## 4. Evaluate & Interpret

```bash
python scripts/eval.py \
  --checkpoint outputs/real_data_run/checkpoints/best.pt \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --splits val test \
  --uncertainty --mc-samples 32 \
  --output outputs/eval/val_eval.json
```

- Generate attention plots:
  ```bash
  python scripts/visualize_attention.py \
    --checkpoint outputs/real_data_run/checkpoints/best.pt \
    --sequence examples/attention_success.fa \
    --output outputs/attention \
    --cdr 30-35,50-65,95-105
  ```
- Explore embeddings:
  ```bash
  python scripts/visualize_embeddings.py \
    --checkpoints outputs/real_data_run/checkpoints/best.pt \
    --parquet data/processed/oas_real_full \
    --splits val \
    --reducers umap \
    --output docs/figures/embeddings
  ```

## 5. Share

- Launch dashboard: `python scripts/launch_dashboard.py --root outputs`
- Publish demo: `python demo/app.py`
- Register the model: `python scripts/registry.py register --id best-val --checkpoint outputs/.../best.pt --metrics-file outputs/eval/val_eval.json`

## Need Help?

- Detailed training tips: [docs/TRAINING.md](TRAINING.md)
- Evaluation cookbook: [docs/EVALUATION.md](EVALUATION.md)
- Reproducibility checklist: [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)
