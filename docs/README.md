# AbProp Playbook

This playbook consolidates the day-to-day docs for setup, data preparation, training, evaluation, visualization, and automation. Use it as the single reference after reading the top-level [README](../README.md).

---

## 1. Environment & Tooling

| Option | Commands |
|--------|----------|
| **Pip** | `python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev,serve,bench,viz,dashboard]'` |
| **Conda** | `conda env create -f environment.yml && conda activate abprop` |
| **Docker** | `docker build -t abprop . && docker run -it --rm -v "$PWD":/workspace -w /workspace abprop bash` |
| **Colab** | Open [`notebooks/quickstart.ipynb`](https://colab.research.google.com/github/abprop/abprop/blob/main/notebooks/quickstart.ipynb) and run `!pip install git+https://github.com/abprop/abprop.git` |

Common extras:

```bash
pip install 'abprop[serve]'         # REST API
pip install -r demo/requirements.txt # Public demo
pip install -e '.[viz]'              # Attention/embedding tooling
pip install -e '.[dashboard]'        # Streamlit dashboard
```

---

## 2. Data Playbook

- **Primary snapshot**: `data/processed/oas_real_full/`
  - 1,502 sequences (train 1,209 · val 144 · test 149)
  - 38 species; heavy + light chains
- **Provenance & checksums**: [data/DATA_PROVENANCE.md](../data/DATA_PROVENANCE.md)
- **Acquire new data**:
  ```bash
  python scripts/fetch_real_antibody_data.py --source sabdab --output data/raw/sabdab_new.csv
  python scripts/process_real_data_etl.py --input data/raw/sabdab_new.csv --output data/processed/oas_real_v3
  ```
- **Quick load**:
  ```python
  from abprop.data import OASDataset
  ds = OASDataset("data/processed/oas_real_full", split="train")
  sample = ds[0]
  ```
- Synthetic sandbox: `python scripts/download_oas_data.py --method synthetic --num-sequences 10000 ...`

---

## 3. Training Workflow

1. **Configure**: tweak `configs/train.yaml`, `configs/model.yaml`, `configs/data.yaml`
2. **Launch**:
   ```bash
   python scripts/train.py \
     --config-path configs/train.yaml \
     --data-config configs/data.yaml \
     --model-config configs/model.yaml \
     --output-dir outputs/real_data_run
   ```
3. **Distributed options**:
   - Single GPU: `abprop-train --distributed none`
   - Multi-GPU (single node): `torchrun --standalone --nproc_per_node 4 python -m abprop.commands.train --distributed ddp`
   - Slurm helper: `python scripts/launch_slurm.py --nodes 2 --gpus-per-node 4 --config configs/train.yaml`
4. **Monitoring**:
   - Logs: `outputs/<run>/logs/metrics.csv`
   - Checkpoints: `outputs/<run>/checkpoints/{last,best}.pt`
   - MLflow: `export MLFLOW_TRACKING_URI=./mlruns && mlflow ui`
5. **Troubleshooting**:
   - Memory: lower `batch_size`, raise `grad_accumulation`
   - NCCL: export `NCCL_DEBUG=warn` and reuse `slurm/env_nccl.sh`
   - Import issues: rely on Docker/conda image or install missing BLAS (`libflexiblas3`)

---

## 4. Evaluation & Interpretability

### Core Evaluation
```bash
python scripts/eval.py \
  --checkpoint outputs/real_data_run/checkpoints/best.pt \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --splits val test \
  --uncertainty --mc-samples 32 \
  --output outputs/eval/val_eval.json
```

### Benchmark Suite
- Run everything: `python scripts/run_benchmarks.py --checkpoint ... --all --html-report`
- Specific tracks: `python scripts/run_benchmarks.py --checkpoint ... --benchmarks perplexity liability cdr_classification`
- Quick reference metrics live in [LEADERBOARD.md](LEADERBOARD.md)

### Visual Analytics
- Attention: `python scripts/visualize_attention.py --checkpoint ... --sequence examples/attention_success.fa --output outputs/attention --label success --interactive`
- Embeddings: `python scripts/visualize_embeddings.py --checkpoints ... --parquet data/processed/oas_real_full --splits val --reducers umap pca --dimensions 2 3 --output docs/figures/embeddings`
- Publication figures: `python scripts/generate_paper_figures.py --style configs/publication.mplstyle --output docs/figures/publication --figures all`

See [docs/RESULTS.md](RESULTS.md) for metrics tables and export recipes. Real-world narratives live in [docs/CASE_STUDIES.md](CASE_STUDIES.md).

---

## 5. Dashboards & Demo

| Tool | Command | Notes |
|------|---------|-------|
| Streamlit dashboard | `python scripts/launch_dashboard.py --root outputs --config configs/dashboard.example.json` | Pages: overview, benchmarks, attention, embeddings, sandbox, errors, checkpoint comparison |
| Gradio demo | `python demo/app.py` | Supports FASTA input, liabilities, CDR highlighting, attention summaries, uncertainty, CSV/PDF export |

Set `ABPROP_DASHBOARD_ROOT`/`ABPROP_DASHBOARD_CONFIG` and `ABPROP_DEMO_CHECKPOINT` to point at curated artifacts.

---

## 6. Automation & Guardrails

- Scheduled CI: `.github/workflows/benchmark.yml`
- Guardrail script:  
  ```bash
  python scripts/check_regression.py \
    --new benchmarks/results/latest.json \
    --reference benchmarks/results/baseline_example.json \
    --max-drop 0.05
  ```
- Artifact workflow:
  1. Refresh baseline JSON & leaderboard
  2. Trigger workflow or wait for weekly schedule
  3. Inspect uploaded `benchmark-results` artifact
  4. Promote improvements and update registry

---

## 7. Cross-Validation & Ablations

- CV data loader helpers: `src/abprop/data/cross_validation.py` (grouped by clonotype)
- Example script: `python scripts/train_cv.py --folds 5 --config configs/train.yaml`
- Ablations:
  ```bash
  python scripts/run_ablations.py \
    --config configs/ablations/liability_vs_mlm.yaml \
    --output outputs/ablations
  python scripts/analyze_ablations.py --input outputs/ablations --report docs/figures/publication/ablations.csv
  ```

---

## 8. Useful Script Index

| Purpose | Script |
|---------|--------|
| Fetch real antibody data | `scripts/fetch_real_antibody_data.py` |
| ETL pipeline | `scripts/process_real_data_etl.py` |
| Therapeutic curation | `scripts/curate_therapeutic_dataset.py` |
| CDR gold standard | `scripts/build_cdr_gold_standard.py` |
| Difficulty splits | `scripts/create_difficulty_splits.py` |
| Difficulty plots | `scripts/plot_difficulty_performance.py` |
| Registry maintenance | `scripts/registry.py` (`list`, `register`, `best`, `export-card`) |
| Reproducibility sweeps | `scripts/reproduce_minimal.sh`, `scripts/reproduce_all.sh` |

---

## 9. Dataset Snapshots

| Dataset | Size | Description | Location |
|---------|------|-------------|----------|
| `oas_real_full` | 1,502 | Primary training corpus | `data/processed/oas_real_full/` |
| `therapeutic_benchmark` | 55 | Clinical-progressed antibodies | `data/processed/therapeutic_benchmark/` |
| `cdr_gold_standard` | 100 | Annotated CDR labels | `data/processed/cdr_gold_standard/` |
| Synthetic sandbox | 10,000 | Random benchmark set | `data/processed/oas_synthetic/` |

---

## 10. Additional Resources

- Results & figures: [docs/RESULTS.md](RESULTS.md)
- Case studies: [docs/CASE_STUDIES.md](CASE_STUDIES.md)
- Reproducibility checklist: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Model registry cards: [../models/MODEL_CARDS.md](../models/MODEL_CARDS.md)
- Demo quickstart (public): [demo/README.md](../demo/README.md)

Keep this file authoritative when reorganizing docs—new topics should extend existing sections instead of spawning separate Markdown files.
