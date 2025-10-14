"""Perplexity benchmark for evaluating language modeling quality on antibody sequences."""

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
from abprop.eval.metrics import compute_perplexity

from .registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark


@register_benchmark("perplexity")
class PerplexityBenchmark(Benchmark):
    """Evaluate MLM perplexity on natural antibody sequences.

    This benchmark measures how well the model predicts masked tokens in
    natural antibody sequences compared to random/shuffled controls.
    Results are stratified by:
    - Chain type (Heavy vs Light)
    - Species (human, mouse, etc.)
    - Sequence length bins
    """

    def load_data(self) -> DataLoader:
        """Load antibody sequence dataset for perplexity evaluation.

        Returns:
            DataLoader with antibody sequences and metadata
        """
        dataset = OASDataset(self.config.data_path, split="test")
        collate_fn = build_collate_fn(generate_mlm=True)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Compute perplexity metrics stratified by sequence properties.

        Args:
            model: AbProp model to evaluate
            dataloader: DataLoader with antibody sequences

        Returns:
            Dictionary containing:
                - overall_perplexity: Overall perplexity across all sequences
                - by_chain: Perplexity broken down by chain type
                - by_species: Perplexity broken down by species
                - by_length: Perplexity broken down by sequence length bins
                - sequence_level: Per-sequence perplexity scores with metadata
        """
        device = torch.device(self.config.device)
        model.to(device)
        model.eval()

        # Track loss and tokens overall and per stratification
        overall_loss = 0.0
        overall_tokens = 0

        by_chain: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "tokens": 0})
        by_species: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "tokens": 0})
        by_length: Dict[str, Dict[str, float]] = defaultdict(lambda: {"loss": 0.0, "tokens": 0})

        sequence_level: List[Dict[str, Any]] = []

        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mlm_labels = batch.get("labels")

                if mlm_labels is None:
                    continue

                mlm_labels = mlm_labels.to(device)

                chains_meta = batch.get("chains")
                species_meta = batch.get("species")

                # Forward pass
                outputs = model(
                    input_ids,
                    attention_mask,
                    mlm_labels=mlm_labels,
                    token_labels=None,
                    liability_targets=None,
                    tasks=("mlm",),
                )

                logits = outputs["mlm_logits"]
                mask = mlm_labels != -100

                # Compute loss per sequence
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

                    seq_perplexity = compute_perplexity(seq_loss, seq_tokens)

                    # Overall accumulation
                    overall_loss += seq_loss
                    overall_tokens += seq_tokens

                    # Get metadata
                    chain_value = None
                    if chains_meta is not None and idx < len(chains_meta):
                        chain_value = chains_meta[idx]
                    chain = chain_value if chain_value else "unknown"

                    species_value = None
                    if species_meta is not None and idx < len(species_meta):
                        species_value = species_meta[idx]
                    species = species_value if species_value else "unknown"
                    seq_len = attention_mask[idx].sum().item()

                    # Stratify by chain
                    by_chain[chain]["loss"] += seq_loss
                    by_chain[chain]["tokens"] += seq_tokens

                    # Stratify by species
                    by_species[species]["loss"] += seq_loss
                    by_species[species]["tokens"] += seq_tokens

                    # Stratify by length (bins of 50)
                    length_bin = f"{int(seq_len // 50) * 50}-{int(seq_len // 50 + 1) * 50}"
                    by_length[length_bin]["loss"] += seq_loss
                    by_length[length_bin]["tokens"] += seq_tokens

                    # Store sequence-level results
                    sequence_level.append({
                        "perplexity": seq_perplexity,
                        "chain": chain,
                        "species": species,
                        "length": int(seq_len),
                    })

                    n_samples += 1
                    if self.config.max_samples and n_samples >= self.config.max_samples:
                        break

                if self.config.max_samples and n_samples >= self.config.max_samples:
                    break

        # Compute perplexities
        overall_perplexity = compute_perplexity(overall_loss, overall_tokens)

        chain_perplexities = {
            chain: compute_perplexity(stats["loss"], stats["tokens"])
            for chain, stats in by_chain.items()
        }

        species_perplexities = {
            species: compute_perplexity(stats["loss"], stats["tokens"])
            for species, stats in by_species.items()
        }

        length_perplexities = {
            length: compute_perplexity(stats["loss"], stats["tokens"])
            for length, stats in by_length.items()
        }

        return {
            "overall_perplexity": overall_perplexity,
            "by_chain": chain_perplexities,
            "by_species": species_perplexities,
            "by_length": length_perplexities,
            "sequence_level": sequence_level,
        }

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate perplexity report with visualizations.

        Creates:
        - Bar plots of perplexity by chain type
        - Bar plots of perplexity by species
        - Line plot of perplexity vs sequence length
        - Distribution plot of sequence-level perplexities

        Args:
            results: Evaluation results from evaluate()

        Returns:
            BenchmarkResult with metrics and plot paths
        """
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}

        # 1. Overall perplexity
        metrics = {
            "overall_perplexity": results["overall_perplexity"],
        }

        # 2. Bar plot: perplexity by chain type
        if results["by_chain"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            chains = list(results["by_chain"].keys())
            perplexities = [results["by_chain"][c] for c in chains]
            ax.bar(chains, perplexities, color="steelblue")
            ax.set_xlabel("Chain Type")
            ax.set_ylabel("Perplexity")
            ax.set_title("Perplexity by Chain Type")
            plt.tight_layout()
            plot_path = output_dir / "perplexity_by_chain.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["by_chain"] = plot_path

            # Add to metrics
            for chain, ppl in results["by_chain"].items():
                metrics[f"perplexity_chain_{chain}"] = ppl

        # 3. Bar plot: perplexity by species
        if results["by_species"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            species_list = list(results["by_species"].keys())
            perplexities = [results["by_species"][s] for s in species_list]
            ax.bar(species_list, perplexities, color="coral")
            ax.set_xlabel("Species")
            ax.set_ylabel("Perplexity")
            ax.set_title("Perplexity by Species")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plot_path = output_dir / "perplexity_by_species.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["by_species"] = plot_path

            # Add to metrics
            for species, ppl in results["by_species"].items():
                metrics[f"perplexity_species_{species}"] = ppl

        # 4. Line plot: perplexity by sequence length
        if results["by_length"]:
            # Sort by length bin
            length_bins = sorted(
                results["by_length"].keys(),
                key=lambda x: int(x.split("-")[0])
            )
            perplexities = [results["by_length"][lb] for lb in length_bins]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(length_bins)), perplexities, marker="o", color="green")
            ax.set_xlabel("Sequence Length Bin")
            ax.set_ylabel("Perplexity")
            ax.set_title("Perplexity vs Sequence Length")
            ax.set_xticks(range(len(length_bins)))
            ax.set_xticklabels(length_bins, rotation=45, ha="right")
            plt.tight_layout()
            plot_path = output_dir / "perplexity_by_length.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["by_length"] = plot_path

        # 5. Distribution plot: sequence-level perplexities
        if results["sequence_level"]:
            perplexities = [s["perplexity"] for s in results["sequence_level"]]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(perplexities, bins=50, color="purple", alpha=0.7, edgecolor="black")
            ax.set_xlabel("Perplexity")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Sequence-Level Perplexities")
            ax.axvline(
                results["overall_perplexity"],
                color="red",
                linestyle="--",
                label=f"Mean: {results['overall_perplexity']:.2f}"
            )
            ax.legend()
            plt.tight_layout()
            plot_path = output_dir / "perplexity_distribution.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["distribution"] = plot_path

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save sequence-level data
        seq_level_path = output_dir / "sequence_level.json"
        with open(seq_level_path, "w", encoding="utf-8") as f:
            json.dump(results["sequence_level"], f, indent=2)

        metadata = {
            "n_sequences": len(results["sequence_level"]),
            "n_chains": len(results["by_chain"]),
            "n_species": len(results["by_species"]),
            "metrics_path": str(metrics_path),
            "sequence_level_path": str(seq_level_path),
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )


__all__ = ["PerplexityBenchmark"]
