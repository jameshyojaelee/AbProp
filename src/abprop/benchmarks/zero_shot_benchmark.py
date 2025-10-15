"""Zero-shot generalization benchmark for evaluating model transfer learning."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from abprop.data import OASDataset, build_collate_fn
from abprop.eval.metrics import compute_perplexity, regression_summary

from .registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark


@register_benchmark("zero_shot")
class ZeroShotBenchmark(Benchmark):
    """Zero-shot generalization benchmark for antibody models.

    Evaluates model performance on:
    - Unseen species (e.g., camel, llama, shark)
    - Unseen germline families
    - Non-human antibody formats (nanobodies, VHH)
    - Cross-species transfer learning

    Metrics:
    - Perplexity on held-out species
    - Liability prediction on novel germlines
    - Performance degradation vs training distribution
    """

    def load_data(self) -> DataLoader:
        """Load dataset containing diverse/novel antibody sequences.

        Expected to contain sequences from species/germlines not in training set.

        Returns:
            DataLoader with diverse antibody sequences
        """
        dataset = OASDataset(self.config.data_path, split="test")
        required = {"species", "germline_v"}
        missing = required - set(dataset.frame.columns)
        if missing:
            raise ValueError(
                "Zero-shot benchmark requires dataset columns "
                f"{sorted(required)}. Missing: {sorted(missing)}"
            )
        collate_fn = build_collate_fn(generate_mlm=True)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate zero-shot generalization performance.

        Args:
            model: AbProp model to evaluate
            dataloader: DataLoader with diverse antibody sequences

        Returns:
            Dictionary containing:
                - perplexity_by_species: Perplexity stratified by species
                - perplexity_by_germline: Perplexity by germline family
                - liability_by_species: Liability prediction metrics by species
                - overall_stats: Overall statistics
        """
        device = torch.device(self.config.device)
        model.to(device)
        model.eval()

        # Track perplexity by species and germline
        by_species: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "tokens": 0})
        by_germline: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "tokens": 0})

        # Track liability predictions by species
        liability_by_species: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {"predictions": [], "targets": []}
        )

        # Get liability keys
        if hasattr(model, "config") and hasattr(model.config, "liability_keys"):
            liability_keys = list(model.config.liability_keys)
        else:
            liability_keys = []

        n_samples = 0
        total_sequences = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mlm_labels = batch.get("labels")
                liability_targets = batch.get("liability_ln")

                if mlm_labels is not None:
                    mlm_labels = mlm_labels.to(device)

                # Determine which tasks to run
                tasks = []
                if mlm_labels is not None:
                    tasks.append("mlm")
                if liability_targets is not None:
                    tasks.append("reg")

                if not tasks:
                    continue

                tasks = tuple(tasks)

                # Forward pass
                outputs = model(
                    input_ids,
                    attention_mask,
                    mlm_labels=mlm_labels,
                    token_labels=None,
                    liability_targets=liability_targets,
                    tasks=tasks,
                )

                # Process MLM results by species/germline
                species_meta = batch.get("species")
                germline_meta = batch.get("germline_v") or batch.get("v_gene")

                if mlm_labels is not None and "mlm" in tasks:
                    logits = outputs["mlm_logits"]
                    mask = mlm_labels != -100

                    for idx in range(input_ids.size(0)):
                        seq_mask = mask[idx]
                        seq_tokens = seq_mask.sum().item()

                        if seq_tokens == 0:
                            continue

                        seq_logits = logits[idx][seq_mask]
                        seq_labels = mlm_labels[idx][seq_mask]

                        seq_loss = torch.nn.functional.cross_entropy(
                            seq_logits,
                            seq_labels,
                            reduction="sum",
                        ).item()

                        # Get metadata
                        species_value = None
                        if species_meta is not None and idx < len(species_meta):
                            species_value = species_meta[idx]
                        species = species_value if species_value else "unknown"

                        germline_value = None
                        if germline_meta is not None and idx < len(germline_meta):
                            germline_value = germline_meta[idx]
                        germline = germline_value if germline_value else "unknown"

                        # Stratify by species
                        by_species[species]["loss"] += seq_loss
                        by_species[species]["tokens"] += seq_tokens

                        # Stratify by germline family (extract family from v_gene)
                        germline_family = germline.split("-")[0] if "-" in germline else germline
                        by_germline[germline_family]["loss"] += seq_loss
                        by_germline[germline_family]["tokens"] += seq_tokens

                        total_sequences += 1

                # Process liability predictions by species
                if liability_targets is not None and "reg" in tasks:
                    preds = outputs["regression"].detach().cpu()
                    target_tensor = model._prepare_regression_targets(
                        liability_targets,
                        batch_size=input_ids.size(0),
                        device=device,
                    ).detach().cpu()

                    for idx in range(input_ids.size(0)):
                        species_value = None
                        if species_meta is not None and idx < len(species_meta):
                            species_value = species_meta[idx]
                        species = species_value if species_value else "unknown"
                        liability_by_species[species]["predictions"].append(preds[idx])
                        liability_by_species[species]["targets"].append(target_tensor[idx])

                n_samples += input_ids.size(0)
                if self.config.max_samples and n_samples >= self.config.max_samples:
                    break

        # Compute perplexities
        species_perplexities = {
            species: compute_perplexity(stats["loss"], stats["tokens"])
            for species, stats in by_species.items()
            if stats["tokens"] > 0
        }

        germline_perplexities = {
            germline: compute_perplexity(stats["loss"], stats["tokens"])
            for germline, stats in by_germline.items()
            if stats["tokens"] > 0
        }

        # Compute liability metrics by species
        liability_metrics_by_species = {}
        for species, data in liability_by_species.items():
            if data["predictions"]:
                preds = torch.stack(data["predictions"])
                targets = torch.stack(data["targets"])
                liability_metrics_by_species[species] = regression_summary(preds, targets)

        return {
            "perplexity_by_species": species_perplexities,
            "perplexity_by_germline": germline_perplexities,
            "liability_by_species": liability_metrics_by_species,
            "total_sequences": total_sequences,
            "species_counts": {
                species: stats["tokens"] // 100  # Approximate sequence count
                for species, stats in by_species.items()
            },
        }

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate zero-shot generalization report with visualizations.

        Creates:
        - Bar plot of perplexity by species
        - Bar plot of perplexity by germline family
        - Liability prediction metrics by species
        - Performance comparison vs training distribution

        Args:
            results: Evaluation results from evaluate()

        Returns:
            BenchmarkResult with metrics and plot paths
        """
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}
        metrics = {}

        # 1. Overall statistics
        metrics["total_sequences"] = results["total_sequences"]
        metrics["n_species"] = len(results["perplexity_by_species"])
        metrics["n_germlines"] = len(results["perplexity_by_germline"])

        # 2. Bar plot: perplexity by species
        if results["perplexity_by_species"]:
            species_list = sorted(
                results["perplexity_by_species"].keys(),
                key=lambda s: results["perplexity_by_species"][s]
            )
            perplexities = [results["perplexity_by_species"][s] for s in species_list]

            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(species_list)), perplexities, color="steelblue", alpha=0.7)

            # Color code: common species (human, mouse) vs rare
            common_species = {"human", "mouse"}
            for idx, species in enumerate(species_list):
                if species.lower() in common_species:
                    bars[idx].set_color("green")
                else:
                    bars[idx].set_color("orange")

            ax.set_xlabel("Species")
            ax.set_ylabel("Perplexity")
            ax.set_title("Zero-Shot Perplexity by Species\n(Green = Common, Orange = Rare)")
            ax.set_xticks(range(len(species_list)))
            ax.set_xticklabels(species_list, rotation=45, ha="right")
            ax.axhline(
                np.mean(perplexities),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(perplexities):.2f}"
            )
            ax.legend()
            plt.tight_layout()
            plot_path = output_dir / "perplexity_by_species.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["perplexity_by_species"] = plot_path

            # Add to metrics
            for species, ppl in results["perplexity_by_species"].items():
                metrics[f"perplexity_{species}"] = ppl

            # Compute statistics
            common_ppls = [
                results["perplexity_by_species"][s]
                for s in species_list
                if s.lower() in common_species
            ]
            rare_ppls = [
                results["perplexity_by_species"][s]
                for s in species_list
                if s.lower() not in common_species
            ]

            if common_ppls:
                metrics["mean_perplexity_common"] = float(np.mean(common_ppls))
            if rare_ppls:
                metrics["mean_perplexity_rare"] = float(np.mean(rare_ppls))
                if common_ppls:
                    metrics["perplexity_gap"] = float(np.mean(rare_ppls) - np.mean(common_ppls))

        # 3. Bar plot: perplexity by germline family (top 20)
        if results["perplexity_by_germline"]:
            # Sort by count (approximate from tokens)
            germline_list = sorted(
                results["perplexity_by_germline"].keys(),
                key=lambda g: results["perplexity_by_germline"][g],
            )[:20]  # Top 20
            perplexities = [results["perplexity_by_germline"][g] for g in germline_list]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(range(len(germline_list)), perplexities, color="coral", alpha=0.7)
            ax.set_xlabel("Germline Family")
            ax.set_ylabel("Perplexity")
            ax.set_title("Zero-Shot Perplexity by Germline Family (Top 20)")
            ax.set_xticks(range(len(germline_list)))
            ax.set_xticklabels(germline_list, rotation=45, ha="right")
            plt.tight_layout()
            plot_path = output_dir / "perplexity_by_germline.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["perplexity_by_germline"] = plot_path

        # 4. Liability metrics by species (if available)
        if results["liability_by_species"]:
            species_list = list(results["liability_by_species"].keys())
            metric_names = ["mse", "r2", "spearman"]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for idx, metric_name in enumerate(metric_names):
                values = [
                    results["liability_by_species"][s][metric_name]
                    for s in species_list
                ]
                axes[idx].bar(range(len(species_list)), values, color="purple", alpha=0.7)
                axes[idx].set_xlabel("Species")
                axes[idx].set_ylabel(metric_name.upper())
                axes[idx].set_title(f"Liability Prediction {metric_name.upper()} by Species")
                axes[idx].set_xticks(range(len(species_list)))
                axes[idx].set_xticklabels(species_list, rotation=45, ha="right")

            plt.tight_layout()
            plot_path = output_dir / "liability_by_species.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["liability_by_species"] = plot_path

            # Add to metrics
            for species, species_metrics in results["liability_by_species"].items():
                for metric_name, value in species_metrics.items():
                    metrics[f"liability_{metric_name}_{species}"] = value

        # 5. Species coverage pie chart
        if results["species_counts"]:
            fig, ax = plt.subplots(figsize=(8, 8))
            species_list = list(results["species_counts"].keys())
            counts = [results["species_counts"][s] for s in species_list]

            ax.pie(
                counts,
                labels=species_list,
                autopct="%1.1f%%",
                startangle=90,
                colors=plt.cm.tab20.colors[:len(species_list)]
            )
            ax.set_title("Species Distribution in Zero-Shot Evaluation")
            plt.tight_layout()
            plot_path = output_dir / "species_distribution.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["species_distribution"] = plot_path

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save detailed results
        detailed_path = output_dir / "detailed_results.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            # Convert non-serializable values
            serializable_results = {
                "perplexity_by_species": results["perplexity_by_species"],
                "perplexity_by_germline": results["perplexity_by_germline"],
                "liability_by_species": results["liability_by_species"],
                "total_sequences": results["total_sequences"],
                "species_counts": results["species_counts"],
            }
            json.dump(serializable_results, f, indent=2)

        metadata = {
            "n_species": len(results["perplexity_by_species"]),
            "n_germlines": len(results["perplexity_by_germline"]),
            "total_sequences": results["total_sequences"],
            "metrics_path": str(metrics_path),
            "detailed_results_path": str(detailed_path),
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )


__all__ = ["ZeroShotBenchmark"]
