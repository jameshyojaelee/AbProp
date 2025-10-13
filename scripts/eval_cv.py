#!/usr/bin/env python
"""Evaluate cross-validation models and generate ensemble predictions.

This script evaluates each fold's model on its held-out validation set,
aggregates results with confidence intervals, and generates ensemble
predictions by averaging across all fold models.

Example usage:
    # Evaluate all CV folds and generate ensemble predictions
    python scripts/eval_cv.py --cv-dir outputs/cv

    # Evaluate on test set instead of validation sets
    python scripts/eval_cv.py --cv-dir outputs/cv --test-split

    # Generate predictions for specific sequences
    python scripts/eval_cv.py --cv-dir outputs/cv --predict-file sequences.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from abprop.commands.train import instantiate_model
from abprop.data import BucketBatchSampler, OASDataset, build_collate_fn
from abprop.data.cross_validation import GroupKFoldDataset
from abprop.models import AbPropModel
from abprop.utils import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CV models and generate ensemble predictions."
    )
    parser.add_argument("--cv-dir", type=Path, required=True, help="CV output directory")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--test-split", action="store_true", help="Evaluate on test split")
    parser.add_argument("--predict-file", type=Path, help="CSV file with sequences to predict")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-file", type=Path, help="Output file for predictions")
    return parser


def load_fold_model(
    fold_dir: Path,
    model_cfg: dict,
    device: torch.device,
    checkpoint: str = "best.pt",
) -> AbPropModel:
    """Load a trained model from a specific fold directory."""
    checkpoint_path = fold_dir / "checkpoints" / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, _ = instantiate_model(model_cfg)
    model.to(device)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model


def evaluate_fold(
    model: AbPropModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a model on a dataloader and return metrics."""
    model.eval()

    total_loss = 0.0
    total_mlm_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            chains = batch["chains"]
            liability_ln = batch["liability_ln"]

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                chains=chains,
                liability_ln=liability_ln,
            )

            total_loss += outputs.loss.item()
            if outputs.mlm_loss is not None:
                total_mlm_loss += outputs.mlm_loss.item()
            if outputs.cls_loss is not None:
                total_cls_loss += outputs.cls_loss.item()
            if outputs.reg_loss is not None:
                total_reg_loss += outputs.reg_loss.item()

            n_batches += 1

    metrics = {
        "eval_loss": total_loss / max(n_batches, 1),
        "eval_mlm_loss": total_mlm_loss / max(n_batches, 1),
        "eval_cls_loss": total_cls_loss / max(n_batches, 1),
        "eval_reg_loss": total_reg_loss / max(n_batches, 1),
    }

    return metrics


def ensemble_predict(
    models: List[AbPropModel],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Generate ensemble predictions by averaging across models."""
    all_predictions: List[Dict[str, torch.Tensor]] = []

    for model_idx, model in enumerate(models):
        model.eval()
        predictions: Dict[str, List[torch.Tensor]] = {
            "liability_preds": [],
            "chain_logits": [],
            "embeddings": [],
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Model {model_idx + 1}/{len(models)}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass without labels (inference mode)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                if outputs.liability_preds is not None:
                    predictions["liability_preds"].append(outputs.liability_preds.cpu())
                if outputs.chain_logits is not None:
                    predictions["chain_logits"].append(outputs.chain_logits.cpu())
                if outputs.embeddings is not None:
                    predictions["embeddings"].append(outputs.embeddings.cpu())

        # Concatenate batches
        concat_predictions = {}
        for key, tensors in predictions.items():
            if tensors:
                concat_predictions[key] = torch.cat(tensors, dim=0)

        all_predictions.append(concat_predictions)

    # Average predictions across models
    ensemble_results: Dict[str, np.ndarray] = {}

    # Aggregate liability predictions
    if all(p.get("liability_preds") is not None for p in all_predictions):
        stacked = torch.stack([p["liability_preds"] for p in all_predictions])
        ensemble_results["liability_mean"] = stacked.mean(dim=0).numpy()
        ensemble_results["liability_std"] = stacked.std(dim=0).numpy()

    # Aggregate chain predictions
    if all(p.get("chain_logits") is not None for p in all_predictions):
        stacked = torch.stack([p["chain_logits"] for p in all_predictions])
        mean_logits = stacked.mean(dim=0)
        ensemble_results["chain_probs"] = torch.softmax(mean_logits, dim=-1).numpy()
        ensemble_results["chain_std"] = stacked.std(dim=0).numpy()

    # Average embeddings
    if all(p.get("embeddings") is not None for p in all_predictions):
        stacked = torch.stack([p["embeddings"] for p in all_predictions])
        ensemble_results["embedding_mean"] = stacked.mean(dim=0).numpy()
        ensemble_results["embedding_std"] = stacked.std(dim=0).numpy()

    return ensemble_results


def evaluate_cv_folds(
    cv_dir: Path,
    model_cfg: dict,
    data_cfg: dict,
    batch_size: int,
    device: torch.device,
    test_split: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Evaluate all CV folds and aggregate results."""
    # Discover all fold directories
    fold_dirs = sorted([d for d in cv_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    n_folds = len(fold_dirs)

    if n_folds == 0:
        raise ValueError(f"No fold directories found in {cv_dir}")

    print(f"Found {n_folds} folds to evaluate")

    # Prepare dataloaders
    processed_root = Path(data_cfg.get("processed_dir", "data/processed"))
    parquet_cfg = data_cfg.get("parquet", {})
    parquet_subdir = parquet_cfg.get("output_dir", "oas")
    parquet_dir = processed_root / parquet_subdir

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, fold_dir in enumerate(fold_dirs):
        print(f"\nEvaluating Fold {fold_idx + 1}/{n_folds}: {fold_dir.name}")

        # Load model
        model = load_fold_model(fold_dir, model_cfg, device)

        # Build dataloader
        if test_split:
            # Evaluate on test split (same for all folds)
            dataset = OASDataset(parquet_dir, split="test")
            print(f"  Using test split: {len(dataset)} sequences")
        else:
            # Evaluate on fold's validation split
            dataset = GroupKFoldDataset(
                parquet_dir=parquet_dir,
                fold_idx=fold_idx,
                n_splits=n_folds,
                split_type="val",
            )
            print(f"  Using validation split: {len(dataset)} sequences")

        collate = build_collate_fn(generate_mlm=True, mlm_probability=0.0)  # No masking for eval
        bins = data_cfg.get("length_bins", [64, 128, 256, 512, 1024])

        sampler = BucketBatchSampler(
            dataset.lengths,
            batch_size=batch_size,
            bins=bins,
            shuffle=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # Evaluate
        metrics = evaluate_fold(model, dataloader, device)
        fold_metrics.append(metrics)

        print("  Metrics:")
        for key, value in sorted(metrics.items()):
            print(f"    {key}: {value:.4f}")

    # Aggregate metrics
    print("\n" + "=" * 80)
    print("Aggregated Results")
    print("=" * 80)

    aggregated: Dict[str, Dict[str, float]] = {}
    all_keys = set()
    for metrics in fold_metrics:
        all_keys.update(metrics.keys())

    for key in sorted(all_keys):
        values = [m[key] for m in fold_metrics if key in m]
        if not values:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values))
        aggregated[key] = {
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        print(f"{key}: {mean:.4f} ± {std:.4f}")

    return fold_metrics, aggregated


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load configs
    model_cfg = load_yaml_config(args.model_config)
    data_cfg = load_yaml_config(args.data_config)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Evaluate CV folds
    fold_metrics, aggregated = evaluate_cv_folds(
        cv_dir=args.cv_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        batch_size=args.batch_size,
        device=device,
        test_split=args.test_split,
    )

    # Save results
    results_path = args.cv_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "fold_metrics": fold_metrics,
                "aggregated_metrics": aggregated,
                "test_split": args.test_split,
            },
            f,
            indent=2,
        )

    print(f"\nEvaluation results saved to: {results_path}")

    # Generate ensemble predictions if requested
    if args.predict_file is not None:
        print("\n" + "=" * 80)
        print("Generating Ensemble Predictions")
        print("=" * 80)

        # Load all fold models
        fold_dirs = sorted([d for d in args.cv_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
        models = []
        for fold_dir in fold_dirs:
            model = load_fold_model(fold_dir, model_cfg, device)
            models.append(model)

        print(f"Loaded {len(models)} models for ensemble prediction")

        # Load sequences to predict
        predict_df = pd.read_csv(args.predict_file)
        if "sequence" not in predict_df.columns:
            raise ValueError("predict_file must have a 'sequence' column")

        print(f"Loaded {len(predict_df)} sequences to predict")

        # Create dataset and dataloader
        from abprop.commands.train import SyntheticSequenceDataset
        from abprop.models import TransformerConfig

        model_config, _ = instantiate_model(model_cfg)
        dataset = SyntheticSequenceDataset(
            sequences=predict_df["sequence"].tolist(),
            liability_keys=model_config.config.liability_keys,
        )

        collate = build_collate_fn(generate_mlm=False, mlm_probability=0.0)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

        # Generate predictions
        predictions = ensemble_predict(models, dataloader, device)

        # Save predictions
        output_file = args.output_file or args.cv_dir / "ensemble_predictions.npz"
        np.savez(output_file, **predictions)
        print(f"\nEnsemble predictions saved to: {output_file}")

        # Create human-readable summary
        summary_data = []
        for i, seq in enumerate(predict_df["sequence"]):
            row = {"sequence": seq}
            if "liability_mean" in predictions:
                row["liability_mean"] = predictions["liability_mean"][i].mean()
                row["liability_std"] = predictions["liability_std"][i].mean()
            if "chain_probs" in predictions:
                row["chain_H_prob"] = predictions["chain_probs"][i, 0]
                row["chain_L_prob"] = predictions["chain_probs"][i, 1]
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_file.with_suffix(".csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Prediction summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
