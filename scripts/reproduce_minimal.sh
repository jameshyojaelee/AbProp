#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.repro_minimal_venv"

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

python scripts/visualize_attention.py \
  --checkpoint "${PROJECT_ROOT}/outputs/attention_demo/checkpoints/random.pt" \
  --sequence "${PROJECT_ROOT}/examples/attention_success.fa" \
  --output "${PROJECT_ROOT}/outputs/attention" \
  --label success || echo "Skipped attention visualization"

python scripts/generate_paper_figures.py \
  --style "${PROJECT_ROOT}/configs/publication.mplstyle" \
  --output "${PROJECT_ROOT}/docs/figures/publication" || echo "Skipped publication figures"
