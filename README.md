# AbProp

AbProp (Antibody Property Modeling) provides a lightweight, HPC-friendly scaffold for building, training, and evaluating sequence-only antibody models with PyTorch 2.1.2 + CUDA 12.x.

## Features
- Data ETL pipeline for OAS-derived antibody sequences with schema validation targets.
- Character-level tokenizer and batching utilities tailored for heavy/light chain workloads.
- Configurable Transformer baseline with metrics for MLM perplexity, CDR/frame classification, and liability regression.
- Distributed-ready launch scripts supporting Slurm single-node and multi-node jobs with NCCL tuning hooks.

## Installation

On local machines you can still use a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev,serve,bench]'
```

On the cluster, follow the HPC setup below for module loads and conda-based installs.

## Project Layout
- `src/abprop/`: Python package (utils, data, tokenizers, models, train, eval, server).
- `configs/`: Hydra-style YAML configs for data, model, training, and distribution.
- `scripts/`: CLI entrypoints for ETL, training, evaluation, and Slurm launch helpers.
- `slurm/`: Batch templates with NCCL environment guidance.
- `tests/`: Unit test skeletons to extend with ETL/tokenizer coverage.
- `data/`: Default data directories (`raw`, `interim`, `processed`).
- `outputs/`: Default experiment output directory (logs, checkpoints, metrics).
- `docs/`: Supplementary documentation (see [docs/README.md](docs/README.md) for details).

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



## REST API

Install the serving extras first (in your env):

```bash
pip install 'abprop[serve]'
```

Run the inference server:

```bash
abprop-serve --checkpoint outputs/checkpoints/best.pt --model-config configs/model.yaml --host 0.0.0.0 --port 8000
```

Sample requests:

```bash
curl -X GET http://localhost:8000/health

curl -X POST http://localhost:8000/score/perplexity   -H 'Content-Type: application/json'   -d '{"sequences": ["ACDEFG", "ACDGST"]}'

curl -X POST http://localhost:8000/score/liabilities   -H 'Content-Type: application/json'   -d '{"sequences": ["ACDEFG", "ACDGST"]}'
```


## HPC Setup

Load base modules (adjust module names to your site):

```bash
module load Miniconda3/23.10.0-1
module load PyTorch/2.1.2
module load CUDA/12.3.0
module load cuDNN
module load NCCL
```

Create a conda environment or container image and install AbProp with extras:

```bash
module load Miniconda3/23.10.0-1
conda create -p $HOME/.conda/abprop python=3.10 -y
conda activate $HOME/.conda/abprop
python -m pip install --upgrade pip
pip install -e '.[dev,serve,bench]'
```

> **Container note**: If you prefer containers, build an image with the same dependency set (PyTorch 2.1.2 + CUDA 12.3 + NCCL) and mount the project root plus `./data`/`./outputs` directories into the container.

## Commands

ETL:

```bash
abprop-etl --input data/raw/oas.tsv --out data/processed/oas --validate
```

Single GPU training:

```bash
abprop-train --distributed none --config-path configs/train.yaml
```

Single-node DDP (torchrun):

```bash
torchrun --standalone --nproc_per_node 4 python -m abprop.commands.train --distributed ddp --config-path configs/train.yaml
```

Multi-node (Slurm helper):

```bash
abprop-launch --nodes 2 --gpus-per-node 4 --config configs/train.yaml
```

Manual multi-node example: edit `slurm/multi_node.sbatch` and submit with `sbatch` for full control over resources.

Difficulty-stratified evaluation:

```bash
python scripts/create_difficulty_splits.py \
  --input data/processed/oas_real_full \
  --output data/processed/stratified_test \
  --split test

python scripts/run_benchmarks.py \
  --checkpoint outputs/checkpoints/best.pt \
  --benchmarks stratified_difficulty \
  --html-report
```

## MLflow Tracking

By default AbProp logs to `./mlruns`. Override with `export MLFLOW_TRACKING_URI=/path/to/mlruns` before launching training. Each run logs:

- CSV logs under `outputs/logs/` (even without MLflow).
- Checkpoints in `outputs/checkpoints/` (last + best-by-metric).
- MLflow run with parameters, metrics, and artifacts (confusion matrix, scatter plots).

Launch the MLflow UI locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```


## Data

The project includes real antibody sequence data from structural databases:

- **Processed Dataset (Recommended)**: `data/processed/oas_real_full/` (1,502 sequences)
  - Source: SAbDab/PDB crystal structures
  - Train: 1,209 | Val: 144 | Test: 149
  - 38 species including human, mouse, llama, etc.
  - Resolution < 4.0 Å, high-quality structures only

For details on data acquisition, processing, and benchmarking datasets, see **[docs/README.md](docs/README.md)**.

## Troubleshooting

**NCCL**
- Use the settings in `slurm/env_nccl.sh` as a baseline (`NCCL_SOCKET_IFNAME`, `NCCL_NET_GDR_LEVEL`).
- Enable `export NCCL_DEBUG=warn` for verbose diagnostics; look for peer failures or interface issues.
- Confirm GPUs are visible with `nvidia-smi` on every node and that firewall rules allow intra-node TCP/IB traffic.

**Slurm / torchrun**
- Confirm rendezvous variables by printing `$MASTER_ADDR` and `$MASTER_PORT` in your batch script.
- Test connectivity with `srun --ntasks=$WORLD_SIZE hostname` prior to launching torchrun.
- For PMI versions that conflict with torchrun, add `--mpi=pmix` (or your site default) to `srun`.

## Documentation

See the [`docs/`](docs/) directory for detailed documentation:
- **[REAL_DATA_SUMMARY.md](docs/REAL_DATA_SUMMARY.md)** - Real antibody data statistics and usage
- **[DATA_ACQUISITION_GUIDE.md](docs/DATA_ACQUISITION_GUIDE.md)** - How to download additional data
- **[EVALUATION_PROMPTS.md](docs/EVALUATION_PROMPTS.md)** - Roadmap for evaluation infrastructure
