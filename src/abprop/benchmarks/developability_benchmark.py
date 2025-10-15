"""Developability benchmark for therapeutic antibody assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader

import pandas as pd
from torch.utils.data import Dataset

from abprop.data import build_collate_fn
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS

from .registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark


@register_benchmark("developability")
class DevelopabilityBenchmark(Benchmark):
    """Benchmark for therapeutic antibody developability prediction.

    Evaluates the model's ability to:
    - Rank therapeutic antibodies by developability
    - Predict clinical progression (phase 1/2/3/approved)
    - Correlate with known aggregation propensity
    - Correlate with immunogenicity risk

    Provides:
    - ROC-AUC for clinical progression prediction
    - Ranking metrics (Spearman correlation, NDCG)
    - Aggregation correlation analysis
    """

    def load_data(self) -> DataLoader:
        """Load therapeutic antibody dataset with developability labels.

        Expected labels:
        - clinical_phase: 0 (preclinical), 1 (phase 1), 2 (phase 2), 3 (phase 3), 4 (approved)
        - aggregation_score: Experimental aggregation propensity
        - immunogenicity_score: Immunogenicity risk score

        Returns:
            DataLoader with therapeutic antibodies and labels
        """
        # Note: This assumes a specialized dataset with therapeutic antibodies
        # In practice, you may need to filter or use a separate dataset
        dataset = OASDataset(self.config.data_path, split="test")
        required = {"clinical_phase", "aggregation_score", "immunogenicity_score"}
        missing = required - set(dataset.frame.columns)
        if missing:
            raise ValueError(
                "Developability benchmark requires the following columns in the dataset: "
                f"{sorted(required)}. Missing: {sorted(missing)}"
            )
        collate_fn = build_collate_fn(generate_mlm=False)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def compute_developability_score(
        self,
        liability_predictions: torch.Tensor,
        liability_keys: List[str],
    ) -> torch.Tensor:
        """Compute composite developability score from liability predictions.

        Lower liability = better developability

        Args:
            liability_predictions: Predicted liability values (N, K)
            liability_keys: Names of liability features

        Returns:
            Developability scores (N,) where higher is better
        """
        # Simple strategy: negative sum of liabilities (higher = better)
        # Weight certain liabilities more heavily
        weights = torch.ones(len(liability_keys))

        # Upweight critical liabilities
        for idx, key in enumerate(liability_keys):
            if key in ["nglyc", "deamidation", "oxidation"]:
                weights[idx] = 2.0
            elif key in ["isomerization"]:
                weights[idx] = 1.5

        # Normalize predictions per liability (z-score)
        normalized = (liability_predictions - liability_predictions.mean(dim=0)) / (
            liability_predictions.std(dim=0) + 1e-8
        )

        # Weighted sum (negate so lower liability = higher score)
        developability_scores = -(normalized * weights).sum(dim=1)

        return developability_scores

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate developability prediction performance.

        Args:
            model: AbProp model to evaluate
            dataloader: DataLoader with therapeutic antibodies

        Returns:
            Dictionary containing:
                - developability_scores: Computed developability scores
                - clinical_phases: Clinical phase labels (if available)
                - aggregation_scores: Aggregation labels (if available)
                - immunogenicity_scores: Immunogenicity labels (if available)
                - liability_predictions: Raw liability predictions
                - liability_keys: Liability feature names
        """
        device = torch.device(self.config.device)
        model.to(device)
        model.eval()

        liability_predictions_list: List[torch.Tensor] = []
        clinical_phases_list: List[int] = []
        aggregation_scores_list: List[float] = []
        immunogenicity_scores_list: List[float] = []

        # Get liability keys from model
        if hasattr(model, "config") and hasattr(model.config, "liability_keys"):
            liability_keys = list(model.config.liability_keys)
        else:
            liability_keys = list(CANONICAL_LIABILITY_KEYS)

        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                liability_targets = batch.get("liability_ln")

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
                liability_predictions_list.append(preds)

                # Collect labels if available
                if "clinical_phase" in batch:
                    clinical_phases_list.extend(batch["clinical_phase"])
                if "aggregation_score" in batch:
                    aggregation_scores_list.extend(batch["aggregation_score"])
                if "immunogenicity_score" in batch:
                    immunogenicity_scores_list.extend(batch["immunogenicity_score"])

                n_samples += input_ids.size(0)
                if self.config.max_samples and n_samples >= self.config.max_samples:
                    break

        if not liability_predictions_list:
            raise ValueError("No liability predictions generated")

        liability_predictions = torch.cat(liability_predictions_list, dim=0)

        # Compute developability scores
        developability_scores = self.compute_developability_score(
            liability_predictions,
            liability_keys,
        )

        return {
            "developability_scores": developability_scores.numpy(),
            "clinical_phases": np.array(clinical_phases_list) if clinical_phases_list else None,
            "aggregation_scores": np.array(aggregation_scores_list) if aggregation_scores_list else None,
            "immunogenicity_scores": np.array(immunogenicity_scores_list) if immunogenicity_scores_list else None,
            "liability_predictions": liability_predictions.numpy(),
            "liability_keys": liability_keys,
        }

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        """Generate developability prediction report with visualizations.

        Creates:
        - ROC curves for clinical progression (if labels available)
        - Correlation plots with aggregation/immunogenicity
        - Developability score distribution
        - Ranking analysis

        Args:
            results: Evaluation results from evaluate()

        Returns:
            BenchmarkResult with metrics and plot paths
        """
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {}
        metrics = {}

        developability_scores = results["developability_scores"]
        clinical_phases = results["clinical_phases"]
        aggregation_scores = results["aggregation_scores"]
        immunogenicity_scores = results["immunogenicity_scores"]

        # 1. Developability score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(developability_scores, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Developability Score")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Developability Scores\n(Higher = Better)")
        ax.axvline(
            developability_scores.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {developability_scores.mean():.2f}"
        )
        ax.legend()
        plt.tight_layout()
        plot_path = output_dir / "score_distribution.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plots["score_distribution"] = plot_path

        metrics["mean_developability_score"] = float(developability_scores.mean())
        metrics["std_developability_score"] = float(developability_scores.std())

        # 2. Clinical progression ROC (if available)
        if clinical_phases is not None and len(clinical_phases) > 0:
            # Binary classification: approved (4) vs not approved (0-3)
            binary_labels = (clinical_phases == 4).astype(int)

            if len(np.unique(binary_labels)) == 2:  # Need both classes
                auc = roc_auc_score(binary_labels, developability_scores)
                fpr, tpr, thresholds = roc_curve(binary_labels, developability_scores)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve: Approved vs Not Approved")
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plot_path = output_dir / "roc_curve.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                plots["roc_curve"] = plot_path

                metrics["roc_auc_approved"] = float(auc)

            # Correlation with clinical phase
            spearman_corr, spearman_pval = stats.spearmanr(clinical_phases, developability_scores)
            metrics["spearman_clinical_phase"] = float(spearman_corr)
            metrics["spearman_clinical_phase_pval"] = float(spearman_pval)

            # Box plot by clinical phase
            fig, ax = plt.subplots(figsize=(10, 6))
            phase_names = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Approved"]
            data_by_phase = [
                developability_scores[clinical_phases == i]
                for i in range(5)
                if (clinical_phases == i).any()
            ]
            present_phases = [phase_names[i] for i in range(5) if (clinical_phases == i).any()]

            ax.boxplot(data_by_phase, labels=present_phases)
            ax.set_xlabel("Clinical Phase")
            ax.set_ylabel("Developability Score")
            ax.set_title(f"Developability by Clinical Phase\n(Spearman ρ = {spearman_corr:.3f}, p = {spearman_pval:.3e})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plot_path = output_dir / "clinical_phase_boxplot.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["clinical_phase_boxplot"] = plot_path

        # 3. Aggregation correlation (if available)
        if aggregation_scores is not None and len(aggregation_scores) > 0:
            # Developability should anti-correlate with aggregation
            spearman_corr, spearman_pval = stats.spearmanr(aggregation_scores, developability_scores)
            metrics["spearman_aggregation"] = float(spearman_corr)
            metrics["spearman_aggregation_pval"] = float(spearman_pval)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(aggregation_scores, developability_scores, alpha=0.5, s=20)
            ax.set_xlabel("Aggregation Score")
            ax.set_ylabel("Developability Score")
            ax.set_title(f"Aggregation vs Developability\n(Spearman ρ = {spearman_corr:.3f}, p = {spearman_pval:.3e})")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / "aggregation_correlation.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["aggregation_correlation"] = plot_path

        # 4. Immunogenicity correlation (if available)
        if immunogenicity_scores is not None and len(immunogenicity_scores) > 0:
            # Developability should anti-correlate with immunogenicity
            spearman_corr, spearman_pval = stats.spearmanr(immunogenicity_scores, developability_scores)
            metrics["spearman_immunogenicity"] = float(spearman_corr)
            metrics["spearman_immunogenicity_pval"] = float(spearman_pval)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(immunogenicity_scores, developability_scores, alpha=0.5, s=20, color="coral")
            ax.set_xlabel("Immunogenicity Score")
            ax.set_ylabel("Developability Score")
            ax.set_title(f"Immunogenicity vs Developability\n(Spearman ρ = {spearman_corr:.3f}, p = {spearman_pval:.3e})")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / "immunogenicity_correlation.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            plots["immunogenicity_correlation"] = plot_path

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save developability scores
        scores_path = output_dir / "developability_scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump({
                "scores": developability_scores.tolist(),
                "clinical_phases": clinical_phases.tolist() if clinical_phases is not None else None,
                "aggregation_scores": aggregation_scores.tolist() if aggregation_scores is not None else None,
                "immunogenicity_scores": immunogenicity_scores.tolist() if immunogenicity_scores is not None else None,
            }, f, indent=2)

        metadata = {
            "n_samples": len(developability_scores),
            "has_clinical_data": clinical_phases is not None and len(clinical_phases) > 0,
            "has_aggregation_data": aggregation_scores is not None and len(aggregation_scores) > 0,
            "has_immunogenicity_data": immunogenicity_scores is not None and len(immunogenicity_scores) > 0,
            "metrics_path": str(metrics_path),
            "scores_path": str(scores_path),
        }

        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata=metadata,
        )


__all__ = ["DevelopabilityBenchmark"]
