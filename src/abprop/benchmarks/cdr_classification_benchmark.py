"""CDR classification benchmark for token-level CDR region prediction."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from abprop.data import OASDataset, build_collate_fn
from abprop.eval.metrics import classification_summary

from .registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark


@register_benchmark("cdr_classification")
class CDRClassificationBenchmark(Benchmark):
    """Token-level CDR region prediction benchmark.

    Evaluates the model's ability to classify each token as framework (0) or
    CDR (1). Provides metrics including:
    - Overall precision, recall, F1
    - Per-position accuracy heatmaps
    - Per-CDR region metrics (if CDR annotations are granular)
    """

    def load_data(self) -> DataLoader:
        """Load antibody sequences with CDR annotations.

        Returns:
            DataLoader with sequences and token-level CDR labels
        """
        dataset = OASDataset(self.config.data_path, split="test")
        collate_fn = build_collate_fn(generate_mlm=False)  # No MLM needed
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate CDR classification performance.

        Args:
            model: AbProp model to evaluate
            dataloader: DataLoader with CDR-annotated sequences

        Returns:
            Dictionary containing:
                - confusion_matrix: TP, FP, TN, FN counts
                - position_accuracy: Per-position accuracy for heatmap
                - per_chain_metrics: Metrics stratified by chain type
        """
        device = torch.device(self.config.device)
        model.to(device)
        model.eval()

        # Overall confusion matrix
        tp = fp = tn = fn = 0

        # Per-position tracking (up to max length)
        max_len = 512
        position_correct = np.zeros(max_len, dtype=np.int64)
        position_total = np.zeros(max_len, dtype=np.int64)

        # Per-chain metrics
        by_chain: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )

        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_labels = batch.get("token_labels")

                if token_labels is None:
                    continue

                token_labels = token_labels.to(device)

                # Forward pass
                outputs = model(
                    input_ids,
                    attention_mask,
                    mlm_labels=None,
                    token_labels=token_labels,
                    liability_targets=None,
                    tasks=("cls",),
                )

                logits = outputs["cls_logits"]
                preds = logits.argmax(dim=-1)
                mask = token_labels != -100

                # Compute confusion matrix
                tp += int(((preds == 1) & (token_labels == 1) & mask).sum().item())
                fp += int(((preds == 1) & (token_labels == 0) & mask).sum().item())
                tn += int(((preds == 0) & (token_labels == 0) & mask).sum().item())
                fn += int(((preds == 0) & (token_labels == 1) & mask).sum().item())

                # Per-position accuracy
                for seq_idx in range(input_ids.size(0)):
                    seq_mask = mask[seq_idx]
                    seq_len = seq_mask.sum().item()
                    seq_preds = preds[seq_idx][:seq_len]
                    seq_labels = token_labels[seq_idx][:seq_len]

                    correct = (seq_preds == seq_labels).cpu().numpy()
                    position_correct[:seq_len] += correct.astype(np.int64)
                    position_total[:seq_len] += 1

                    # Per-chain metrics
                    chain = batch.get("chain_type", ["unknown"] * input_ids.size(0))[seq_idx]
                    chain_mask = seq_mask[:seq_len]
                    chain_preds = seq_preds[:seq_len]
                    chain_labels = seq_labels[:seq_len]

                    by_chain[chain]["tp"] += int(
                        ((chain_preds == 1) & (chain_labels == 1)).sum().item()
                    )
                    by_chain[chain]["fp"] += int(
                        ((chain_preds == 1) & (chain_labels == 0)).sum().item()
                    )
                    by_chain[chain]["tn"] += int(
                        ((chain_preds == 0) & (chain_labels == 0)).sum().item()
                    )
                    by_chain[chain]["fn"] += int(
                        ((chain_preds == 0) & (chain_labels == 1)).sum().item()
                    )

                    n_samples += 1
                    if self.config.max_samples and n_samples >= self.config.max_samples:
                        break

                if self.config.max_samples and n_samples >= self.config.max_samples:
                    break

        # Compute position accuracy
        position_accuracy = np.divide(
            position_correct,
            position_total,
            out=np.zeros_like(position_correct, dtype=np.float32),
            where=position_total > 0,
        )

        # Compute per-chain metrics
        chain_metrics = {}
        for chain, counts in by_chain.items():
            chain_metrics[chain] = classification_summary(
                counts["tp"],
                counts["fp"],
                counts["tn"],
                counts["fn"],
            )
            chain_metrics[chain]["confusion_matrix"] = counts

        return {
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "position_accuracy": position_accuracy.tolist(),
            "position_total": position_total.tolist(),
            "per_chain_metrics": chain_metrics,
        }

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate CDR classification report with visualizations.

        Creates:
        - Confusion matrix heatmap
        - Per-position accuracy line plot
        - Per-chain metrics bar plots

        Args:
            results: Evaluation results from evaluate()

        Returns:
            BenchmarkResult with metrics and plot paths
        """
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # 1. Overall metrics
        cm = results["confusion_matrix"]
        overall_metrics = classification_summary(cm["tp"], cm["fp"], cm["tn"], cm["fn"])
        metrics = {
            "accuracy": overall_metrics["accuracy"],
            "precision": overall_metrics["precision"],
            "recall": overall_metrics["recall"],
            "f1": overall_metrics["f1"],
            "tp": cm["tp"],
            "fp": cm["fp"],
            "tn": cm["tn"],
            "fn": cm["fn"],
        }

        # 2. Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_matrix = np.array([
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]],
        ])
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Framework", "CDR"],
            yticklabels=["Framework", "CDR"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("CDR Classification Confusion Matrix")
        plt.tight_layout()
        plot_path = output_dir / "confusion_matrix.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plots["confusion_matrix"] = plot_path

        # 3. Per-position accuracy line plot
        position_accuracy = np.array(results["position_accuracy"])
        position_total = np.array(results["position_total"])
        valid_positions = position_total > 0
        valid_acc = position_accuracy[valid_positions]

        if len(valid_acc) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            positions = np.arange(len(position_accuracy))[valid_positions]
            ax.plot(positions, valid_acc, linewidth=1.5, color="darkblue")
            ax.set_xlabel("Position")
            ax.set_ylabel("Accuracy")
            ax.set_title("Per-Position CDR Classification Accuracy")
            ax.set_ylim([0, 1.05])
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / "position_accuracy.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["position_accuracy"] = plot_path

        # 4. Per-chain metrics bar plots
        if results["per_chain_metrics"]:
            chains = list(results["per_chain_metrics"].keys())
            metric_names = ["precision", "recall", "f1"]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for idx, metric_name in enumerate(metric_names):
                values = [
                    results["per_chain_metrics"][chain][metric_name]
                    for chain in chains
                ]
                axes[idx].bar(chains, values, color="teal")
                axes[idx].set_xlabel("Chain Type")
                axes[idx].set_ylabel(metric_name.capitalize())
                axes[idx].set_title(f"{metric_name.capitalize()} by Chain Type")
                axes[idx].set_ylim([0, 1.05])

            plt.tight_layout()
            plot_path = output_dir / "metrics_by_chain.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["metrics_by_chain"] = plot_path

            # Add per-chain metrics
            for chain, chain_metrics in results["per_chain_metrics"].items():
                for metric_name, value in chain_metrics.items():
                    if metric_name != "confusion_matrix":
                        metrics[f"{metric_name}_chain_{chain}"] = value

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save detailed results
        detailed_path = output_dir / "detailed_results.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        metadata = {
            "metrics_path": str(metrics_path),
            "detailed_results_path": str(detailed_path),
            "n_chains": len(results.get("per_chain_metrics", {})),
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )


__all__ = ["CDRClassificationBenchmark"]
