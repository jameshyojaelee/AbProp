"""Liability prediction benchmark for antibody developability assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from abprop.data import OASDataset, build_collate_fn
from abprop.eval.metrics import regression_per_key, regression_summary
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS

from .registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark


@register_benchmark("liability")
class LiabilityBenchmark(Benchmark):
    """Benchmark for antibody liability prediction.

    Evaluates regression performance for each liability type:
    - N-glycosylation sites (nglyc)
    - Deamidation sites
    - Isomerization sites
    - Oxidation sites
    - Free cysteines

    Provides:
    - Regression metrics (MSE, R², Spearman) per liability
    - Calibration plots (predicted vs actual)
    - Risk stratification analysis
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize liability benchmark.

        Args:
            config: Benchmark configuration
        """
        super().__init__(config)
        # Default liability keys - can be overridden
        self.liability_keys = list(CANONICAL_LIABILITY_KEYS)

    def load_data(self) -> DataLoader:
        """Load antibody sequences with liability annotations.

        Returns:
            DataLoader with sequences and liability targets
        """
        dataset = OASDataset(self.config.data_path, split="test")
        collate_fn = build_collate_fn(generate_mlm=False)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate liability prediction performance.

        Args:
            model: AbProp model to evaluate
            dataloader: DataLoader with liability-annotated sequences

        Returns:
            Dictionary containing:
                - predictions: Model predictions tensor
                - targets: Ground truth tensor
                - liability_keys: List of liability feature names
        """
        device = torch.device(self.config.device)
        model.to(device)
        model.eval()

        predictions_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []

        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                liability_targets = batch.get("liability_ln")

                if liability_targets is None:
                    continue

                # Forward pass
                outputs = model(
                    input_ids,
                    attention_mask,
                    mlm_labels=None,
                    token_labels=None,
                    liability_targets=liability_targets,
                    tasks=("reg",),
                )

                preds = outputs["regression"].detach().cpu()
                predictions_list.append(preds)

                # Prepare targets
                target_tensor = model._prepare_regression_targets(
                    liability_targets,
                    batch_size=input_ids.size(0),
                    device=device,
                ).detach().cpu()
                targets_list.append(target_tensor)

                n_samples += input_ids.size(0)
                if self.config.max_samples and n_samples >= self.config.max_samples:
                    break

        if not predictions_list:
            raise ValueError("No liability data found in dataset")

        predictions = torch.cat(predictions_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        # Get liability keys from model config
        if hasattr(model, "config") and hasattr(model.config, "liability_keys"):
            self.liability_keys = list(model.config.liability_keys)

        return {
            "predictions": predictions,
            "targets": targets,
            "liability_keys": self.liability_keys,
        }

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate liability prediction report with visualizations.

        Creates:
        - Scatter plots of predicted vs actual for each liability
        - Calibration plots
        - Metrics table (MSE, R², Spearman) per liability
        - Risk stratification analysis

        Args:
            results: Evaluation results from evaluate()

        Returns:
            BenchmarkResult with metrics and plot paths
        """
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}
        predictions = results["predictions"]
        targets = results["targets"]
        liability_keys = results["liability_keys"]

        # 1. Overall regression metrics
        overall_metrics = regression_summary(predictions, targets)
        metrics = {
            "overall_mse": overall_metrics["mse"],
            "overall_r2": overall_metrics["r2"],
            "overall_spearman": overall_metrics["spearman"],
        }

        # 2. Per-liability metrics
        per_key_metrics = regression_per_key(predictions, targets, liability_keys)

        for key, key_metrics in per_key_metrics.items():
            for metric_name, value in key_metrics.items():
                metrics[f"{key}_{metric_name}"] = value

        # 3. Scatter plots: predicted vs actual for each liability
        n_keys = len(liability_keys)
        n_cols = min(3, n_keys)
        n_rows = (n_keys + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_keys == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, key in enumerate(liability_keys):
            ax = axes[idx]
            pred_vals = predictions[:, idx].numpy()
            target_vals = targets[:, idx].numpy()

            ax.scatter(target_vals, pred_vals, alpha=0.4, s=10, color="steelblue")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{key}\nR²={per_key_metrics[key]['r2']:.3f}")

            # Identity line
            min_val = min(target_vals.min(), pred_vals.min())
            max_val = max(target_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_keys, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plot_path = output_dir / "scatter_plots.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plots["scatter_plots"] = plot_path

        # 4. Calibration plots (binned predicted vs actual)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_keys == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, key in enumerate(liability_keys):
            ax = axes[idx]
            pred_vals = predictions[:, idx].numpy()
            target_vals = targets[:, idx].numpy()

            # Bin predictions
            n_bins = 10
            bins = np.linspace(pred_vals.min(), pred_vals.max(), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_indices = np.digitize(pred_vals, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            bin_means = []
            for b in range(n_bins):
                mask = bin_indices == b
                if mask.sum() > 0:
                    bin_means.append(target_vals[mask].mean())
                else:
                    bin_means.append(np.nan)

            ax.plot(bin_centers, bin_means, "o-", linewidth=2, markersize=6, label="Calibration")
            ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], "r--", linewidth=2, label="Perfect")
            ax.set_xlabel("Predicted (binned)")
            ax.set_ylabel("Actual (mean)")
            ax.set_title(f"{key} Calibration")
            ax.legend()
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_keys, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plot_path = output_dir / "calibration_plots.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plots["calibration_plots"] = plot_path

        # 5. Risk stratification (low/medium/high)
        # Define thresholds based on percentiles
        risk_stratification = {}
        for idx, key in enumerate(liability_keys):
            pred_vals = predictions[:, idx].numpy()
            target_vals = targets[:, idx].numpy()

            # Define risk levels: low (0-33rd), medium (33-66th), high (66-100th)
            low_thresh = np.percentile(target_vals, 33)
            high_thresh = np.percentile(target_vals, 66)

            low_mask = target_vals <= low_thresh
            med_mask = (target_vals > low_thresh) & (target_vals <= high_thresh)
            high_mask = target_vals > high_thresh

            risk_stratification[key] = {
                "low_mse": float(((pred_vals[low_mask] - target_vals[low_mask]) ** 2).mean()),
                "medium_mse": float(((pred_vals[med_mask] - target_vals[med_mask]) ** 2).mean()),
                "high_mse": float(((pred_vals[high_mask] - target_vals[high_mask]) ** 2).mean()),
                "low_count": int(low_mask.sum()),
                "medium_count": int(med_mask.sum()),
                "high_count": int(high_mask.sum()),
            }

            # Add to metrics
            metrics[f"{key}_risk_low_mse"] = risk_stratification[key]["low_mse"]
            metrics[f"{key}_risk_medium_mse"] = risk_stratification[key]["medium_mse"]
            metrics[f"{key}_risk_high_mse"] = risk_stratification[key]["high_mse"]

        # 6. Bar plot of MSE by risk level
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(liability_keys))
        width = 0.25

        low_mse = [risk_stratification[k]["low_mse"] for k in liability_keys]
        med_mse = [risk_stratification[k]["medium_mse"] for k in liability_keys]
        high_mse = [risk_stratification[k]["high_mse"] for k in liability_keys]

        ax.bar(x - width, low_mse, width, label="Low Risk", color="green", alpha=0.7)
        ax.bar(x, med_mse, width, label="Medium Risk", color="orange", alpha=0.7)
        ax.bar(x + width, high_mse, width, label="High Risk", color="red", alpha=0.7)

        ax.set_xlabel("Liability")
        ax.set_ylabel("MSE")
        ax.set_title("MSE by Risk Stratification")
        ax.set_xticks(x)
        ax.set_xticklabels(liability_keys, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plot_path = output_dir / "risk_stratification.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plots["risk_stratification"] = plot_path

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save risk stratification details
        risk_path = output_dir / "risk_stratification.json"
        with open(risk_path, "w", encoding="utf-8") as f:
            json.dump(risk_stratification, f, indent=2)

        metadata = {
            "n_samples": predictions.size(0),
            "n_liabilities": len(liability_keys),
            "liability_keys": liability_keys,
            "metrics_path": str(metrics_path),
            "risk_stratification_path": str(risk_path),
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )


__all__ = ["LiabilityBenchmark"]
