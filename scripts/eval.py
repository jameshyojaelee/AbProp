#!/usr/bin/env python
"""CLI entrypoint for AbProp evaluation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abprop.eval import classification_metrics, compute_perplexity
from abprop.models import SimpleTransformerEncoder, TransformerConfig
from abprop.tokenizers import AminoAcidTokenizer
from abprop.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AbProp models on validation data or checkpoints."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a model checkpoint file (.pt).",
    )
    parser.add_argument(
        "--perplexity-nll",
        type=float,
        default=None,
        help="Average negative log-likelihood for computing perplexity.",
    )
    return parser.parse_args()


def load_model(model_config: Path, checkpoint: Path | None) -> SimpleTransformerEncoder:
    cfg = load_yaml_config(model_config)
    model = SimpleTransformerEncoder(
        TransformerConfig(
            vocab_size=cfg.get("vocab_size", TransformerConfig.vocab_size),
            d_model=cfg.get("d_model", TransformerConfig.d_model),
            nhead=cfg.get("nhead", TransformerConfig.nhead),
            num_layers=cfg.get("num_layers", TransformerConfig.num_layers),
            dropout=cfg.get("dropout", TransformerConfig.dropout),
            max_position_embeddings=cfg.get(
                "max_position_embeddings", TransformerConfig.max_position_embeddings
            ),
        )
    )
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


def run_evaluation(args: argparse.Namespace) -> None:
    model = load_model(args.model_config, args.checkpoint)
    tokenizer = AminoAcidTokenizer.build_default()

    print("Model ready for evaluation.")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    if args.perplexity_nll is not None:
        ppl = compute_perplexity(torch.tensor(args.perplexity_nll))
        print(f"Perplexity: {ppl:.4f}")

    # Placeholder classification example.
    sample_logits = torch.randn(4, 2, 2)
    sample_labels = torch.randint(0, 2, (4, 2))
    metrics = classification_metrics(sample_logits, sample_labels)
    print("Sample classification metrics:", metrics)


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

