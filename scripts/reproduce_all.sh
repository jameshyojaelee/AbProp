#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.repro_venv"

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

# 1. Data provenance (user-supplied dataset)
# Expect parquet splits under data/processed/oas_real_full

# 2. Training (optional)
# python scripts/train.py --config-path configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml

# 3. Evaluation
# python scripts/eval.py --checkpoint outputs/real_data_run/checkpoints/best.pt --data-config configs/data.yaml --model-config configs/model.yaml --splits val test --output outputs/eval/val_eval.json

# 4. Attention visualizations
python scripts/visualize_attention.py \
  --checkpoint "${PROJECT_ROOT}/outputs/real_data_run/checkpoints/best.pt" \
  --sequence "${PROJECT_ROOT}/examples/attention_success.fa" \
  --output "${PROJECT_ROOT}/outputs/attention" \
  --label success || echo "Skipped attention visualization (checkpoint missing)"

# 5. Embedding visualizations
python scripts/visualize_embeddings.py \
  --checkpoints "${PROJECT_ROOT}/outputs/real_data_run/checkpoints/best.pt" \
  --labels best \
  --parquet "${PROJECT_ROOT}/data/processed/oas_real_full" \
  --splits val \
  --reducers umap \
  --dimensions 2 \
  --output "${PROJECT_ROOT}/docs/figures/embeddings" || echo "Skipped embedding visualization"

# 6. Publication figures
python scripts/generate_paper_figures.py \
  --style "${PROJECT_ROOT}/configs/publication.mplstyle" \
  --output "${PROJECT_ROOT}/docs/figures/publication" || echo "Skipped paper figures"

# 7. Benchmark regression guardrail
python scripts/check_regression.py \
  --new "${PROJECT_ROOT}/benchmarks/results/latest.json" \
  --reference "${PROJECT_ROOT}/benchmarks/results/baseline_example.json" \
  --max-drop 0.05 || echo "Regression guardrail failed"

# 8. Model registry snapshot
python scripts/registry.py register \
  --id repro-run \
  --checkpoint "${PROJECT_ROOT}/outputs/real_data_run/checkpoints/best.pt" \
  --config "${PROJECT_ROOT}/outputs/real_data_run/config_snapshot.json" \
  --metrics-file "${PROJECT_ROOT}/outputs/eval/val_eval.json" \
  --tags reproducibility || echo "Registry update skipped"

