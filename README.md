# AbProp

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![CI](https://img.shields.io/badge/ci-benchmark--guardrail-success.svg)](.github/workflows/benchmark.yml)
[![Docs](https://img.shields.io/badge/docs-full-green.svg)](docs/README.md)

Sequence-only antibody property modeling with batteries included: attention introspection, embedding analytics, dashboards, demo apps, guardrails, and a lightweight registry so teams can move from experiments to production quickly.

---

## Performance Snapshot

| Task | Metric | Validation | Test | Notes |
|------|--------|------------|------|-------|
| Masked language modeling | Perplexity ↓ | **1.95** | 2.01 | Baseline checkpoint (`benchmarks/results/baseline_example.json`) |
| CDR identification | Macro F1 ↑ | **0.89** | 0.88 | Token classifier on OAS hold-out |
| Liability regression | RMSE ↓ | **0.27** | 0.29 | Includes MC-dropout uncertainty bands |

See [docs/RESULTS.md](docs/RESULTS.md) for figure-ready summaries, ablations, and export commands. Case-study deep dives live under [docs/CASE_STUDIES.md](docs/CASE_STUDIES.md).

---

## Quickstart

### 1. Pick an environment

- **Pip**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e '.[dev,serve,bench,viz,dashboard]'
  ```
- **Colab**
  - Open [notebooks/quickstart.ipynb](https://colab.research.google.com/github/abprop/abprop/blob/main/notebooks/quickstart.ipynb)
  - Run the install cell: `!pip install git+https://github.com/abprop/abprop.git`
  - Execute the demo cell to reproduce attention visuals in the hosted runtime
- **Conda**
  ```bash
  conda env create -f environment.yml
  conda activate abprop
  ```
- **Docker**
  ```bash
  docker build -t abprop .
  docker run -it --rm -v "$PWD":/workspace -w /workspace abprop bash
  ```

### 2. Reproduce essentials

```bash
scripts/reproduce_minimal.sh   # sanity-check visualizations
scripts/reproduce_all.sh       # full pipeline (training → eval → viz → registry)
```

### 3. Launch tooling

| Asset | Command |
|-------|---------|
| Streamlit dashboard | `python scripts/launch_dashboard.py --root outputs` |
| Gradio demo | `pip install -r demo/requirements.txt && python demo/app.py` |
| Attention explorer | `python scripts/visualize_attention.py --help` |
| Embedding explorer | `python scripts/visualize_embeddings.py --help` |

---

## Model Zoo

| ID | Description | Val Metric | Checkpoint | Card |
|----|-------------|------------|------------|------|
| `best-val-2024-07-01` | Production-ready baseline | F1 0.89 | `outputs/real_data_run/checkpoints/best.pt` | `models/cards/best-val-2024-07-01.md` |
| `repro-run` | Reproducibility script snapshot | Perplexity 1.95 | `outputs/real_data_run/checkpoints/best.pt` | `models/cards/repro-run.md` |

Manage entries with the registry CLI:

```bash
python scripts/registry.py list
python scripts/registry.py best --metric f1 --higher
python scripts/registry.py export-card --id best-val-2024-07-01 --output models/cards/best-val-2024-07-01.md
```

---

## Usage Examples

### Python inference

```python
from pathlib import Path
import torch
from abprop.models import AbPropModel, TransformerConfig
from abprop.tokenizers.aa import collate_batch

checkpoint = Path("outputs/real_data_run/checkpoints/best.pt")
state = torch.load(checkpoint, map_location="cpu")
config = TransformerConfig(**state.get("model_config", {}))
model = AbPropModel(config).eval()
model.load_state_dict(state["model_state"], strict=False)

batch = collate_batch(["EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWV", "QVQLVESGGDLVQPGGSLRLSCAASGYNFNNYMSWV"])
outputs = model(batch["input_ids"], batch["attention_mask"], tasks=("mlm", "cls", "reg"))
print(outputs["metrics"], outputs["regression"].tolist())
```

### Fine-tuning via CLI

```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --output-dir outputs/new_run
```

### REST API peek

```bash
pip install 'abprop[serve]'
abprop-serve --checkpoint outputs/real_data_run/checkpoints/best.pt --model-config configs/model.yaml
```

---

## Documentation Map

- [Playbook](docs/README.md) – environment, data, training, evaluation, dashboards
- [Results](docs/RESULTS.md) – metrics tables, export recipes, publication figures
- [Case studies](docs/CASE_STUDIES.md) – developability, CDR, QC, humanization, failure analysis
- [Reproducibility](REPRODUCIBILITY.md) – seeds, scripts, Docker, CI guardrails

---

## Visual Gallery

| Attention rollout (success) | Embedding UMAP comparison | Dashboard overview |
|-----------------------------|---------------------------|--------------------|
| `outputs/attention/success/aggregated/rollout.png` | `docs/figures/embeddings/umap_2d/comparison/embedded_points.csv` | `docs/figures/dashboard/overview.png` |

Drop updated PNG/HTML assets into the directories above to keep slide decks and papers in sync with latest experiments.

---

## Frequently Asked Questions

**Where do I get data?**  See [data/DATA_PROVENANCE.md](data/DATA_PROVENANCE.md) for download links and checksums.

**How do I monitor regressions?**  Run `python scripts/check_regression.py --new <fresh.json> --reference benchmarks/results/baseline_example.json --max-drop 0.05` or rely on the scheduled GitHub Action.

**Can I humanize a murine antibody?**  Yes – generate proposals via masked language modeling (see [docs/CASE_STUDIES.md](docs/CASE_STUDIES.md#humanization-pathways)) and review liabilities + attention shifts in the dashboard.

**What if PyTorch fails to import (libflexiblas)?**  Use the provided Docker image or conda env, or install `libflexiblas3` on the target system.

---

## Project Layout

```
├── configs/                # YAML configs, dashboards, publication styles
├── data/                   # Raw, processed, provenance files
├── demo/                   # Gradio app + requirements
├── docs/                   # Guides, results, case studies, figures
├── scripts/                # CLI utilities (training, evaluation, viz, registry)
├── src/abprop/             # Library code (models, viz, registry, eval)
└── tests/                  # Unit tests (model, registry, viz helpers)
```

---

## Contributing

1. Create a feature branch and update or add tests (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`).
2. Run linting (`make lint`) and formatters (`make format`).
3. Update documentation and registry entries when behavior changes.
4. Open a PR with before/after artifacts (attention maps, benchmark JSON, etc.).

By contributing you agree to the [Apache 2.0](LICENSE) license.
