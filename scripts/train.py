#!/usr/bin/env python
"""CLI entrypoint for AbProp training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abprop.models import SimpleTransformerEncoder, TransformerConfig
from abprop.tokenizers import AminoAcidTokenizer
from abprop.train import Trainer, TrainingConfig
from abprop.utils import DEFAULT_OUTPUT_DIR, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AbProp baseline Transformer model on antibody sequences."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store logs, checkpoints, and outputs.",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=0,
        help="Optional number of synthetic steps to run for smoke testing.",
    )
    return parser.parse_args()


def instantiate_model(model_cfg: dict) -> SimpleTransformerEncoder:
    defaults = TransformerConfig()
    task_weights = model_cfg.get("task_weights", {})
    config = TransformerConfig(
        vocab_size=model_cfg.get("vocab_size", defaults.vocab_size),
        d_model=model_cfg.get("d_model", defaults.d_model),
        nhead=model_cfg.get("nhead", defaults.nhead),
        num_layers=model_cfg.get("num_layers", defaults.num_layers),
        dim_feedforward=model_cfg.get("dim_feedforward", defaults.dim_feedforward),
        dropout=model_cfg.get("dropout", defaults.dropout),
        max_position_embeddings=model_cfg.get(
            "max_position_embeddings", defaults.max_position_embeddings
        ),
        liability_keys=tuple(model_cfg.get("liability_keys", list(defaults.liability_keys))),
        mlm_weight=task_weights.get("mlm", defaults.mlm_weight),
        cls_weight=task_weights.get("cls", defaults.cls_weight),
        reg_weight=task_weights.get("reg", defaults.reg_weight),
    )
    return SimpleTransformerEncoder(config)


def run_training(args: argparse.Namespace) -> None:
    train_cfg = load_yaml_config(args.config_path)
    model_cfg = load_yaml_config(args.model_config)
    data_cfg = load_yaml_config(args.data_config)

    print("Training configuration:", train_cfg)
    print("Model configuration:", model_cfg)
    print("Data configuration:", data_cfg)

    tokenizer = AminoAcidTokenizer.build_default()
    model = instantiate_model(model_cfg)
    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            learning_rate=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-2),
            max_steps=train_cfg.get("max_steps", 1000),
            log_every=train_cfg.get("eval_interval", 100),
            output_dir=args.output_dir,
        ),
    )

    if args.dry_run_steps > 0:
        print(f"Running {args.dry_run_steps} synthetic steps for smoke test.")
        vocab_size = len(tokenizer.vocab)
        batch_size = train_cfg.get("batch_size", 4)
        seq_len = 32
        for step in range(args.dry_run_steps):
            batch = {
                "input_ids": torch.randint(
                    low=0, high=vocab_size, size=(batch_size, seq_len)
                ),
                "labels": torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len)),
                "padding_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
            }
            metrics = trainer.train_step(batch)
            if (step + 1) % max(1, trainer.config.log_every) == 0:
                print(f"Step {step + 1}: loss={metrics['loss']:.4f}")
    else:
        print("Dataset loading not yet implemented. Provide your own pipeline in scripts/etl.py.")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
