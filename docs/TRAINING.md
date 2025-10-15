# Training AbProp Models

This guide dives deeper into configuration strategies, distributed tips, and logging best practices.

## Configuration Layers

| Config | Purpose | Default |
|--------|---------|---------|
| `configs/train.yaml` | Optimization, scheduler, logging | AMP, cosine LR, grad clipping |
| `configs/data.yaml` | Dataset locations, splits, metadata flags | OAS processed parquet |
| `configs/model.yaml` | Transformer depth, width, dropout, tasks | 3-layer encoder, 6 heads |

Override via CLI flags (`--config-path`, `--data-config`, `--model-config`) or maintain experiment-specific YAMLs under `configs/`.

## Curriculum

1. **Warm-up** – short run (`max_steps=500`) to validate ETL + logging
2. **Baseline** – full MLM+CLS+REG training (`max_steps=5000`)
3. **Ablations** – toggle task weights and compare metrics using `scripts/run_ablations.py`
4. **Ensembles / MC Dropout** – enable `--uncertainty` at evaluation for production scoring

## Distributed Training

- Single GPU: `abprop-train --distributed none`
- Multi-GPU single node: `torchrun --standalone --nproc_per_node 4 python -m abprop.commands.train`
- Multi-node: use `scripts/launch_slurm.py` or adapt `slurm/multi_node.sbatch`
- Set `CUDA_VISIBLE_DEVICES` before launching to avoid conflicts
- For NCCL issues, consult `docs/TRAINING_GUIDE.md` and the troubleshooting section in the README

## Logging & Checkpoints

- Logs: `outputs/<run>/logs/metrics.csv`
- Checkpoints: `outputs/<run>/checkpoints/{last,best}.pt`
- MLflow: configure `MLFLOW_TRACKING_URI`; results appear under the `abprop` experiment
- Snapshot configs: `outputs/<run>/config_snapshot.json`

## Tips

- Reduce `max_position_embeddings` if focusing on short CDR-only sequences
- Increase `mlm_probability` in `build_collate_fn` for tougher corruption regimes
- Use `scripts/registry.py register` after training to keep the registry in sync

## Further Reading

- [Getting started](GETTING_STARTED.md)
- [Evaluation & interpretability](EVALUATION.md)
- [Case studies](case_studies/README.md)
