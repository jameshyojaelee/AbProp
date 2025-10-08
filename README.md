# AbProp

AbProp (Antibody Property Modeling) provides a lightweight, HPC-friendly scaffold for building, training, and evaluating sequence-only antibody models with PyTorch 2.1.2 + CUDA 12.x.

## Features
- Data ETL pipeline for OAS-derived antibody sequences with schema validation targets.
- Character-level tokenizer and batching utilities tailored for heavy/light chain workloads.
- Configurable Transformer baseline with metrics for MLM perplexity, CDR/frame classification, and liability regression.
- Distributed-ready launch scripts supporting Slurm single-node and multi-node jobs with NCCL tuning hooks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Project Layout
- `src/abprop/`: Python package (utils, data, tokenizers, models, train, eval, server).
- `configs/`: Hydra-style YAML configs for data, model, training, and distribution.
- `scripts/`: CLI entrypoints for ETL, training, evaluation, and Slurm launch helpers.
- `slurm/`: Batch templates with NCCL environment guidance.
- `tests/`: Unit test skeletons to extend with ETL/tokenizer coverage.
- `data/`: Default data directories (`raw`, `interim`, `processed`).
- `outputs/`: Default experiment output directory (logs, checkpoints, metrics).

## Usage

Activate your environment, install the package, then explore the CLI interfaces:

```bash
python scripts/train.py --help
```

Example training dry-run with default config paths:

```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml
```

## Development

- `make format` – format with Ruff.
- `make lint` – static checks (Ruff + mypy).
- `make test` – run pytest.
- `make clean` – remove build artifacts.

Extend the unit tests under `tests/` to match new components and keep the configs in sync with your experiments.

