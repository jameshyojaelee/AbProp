#!/usr/bin/env python3
"""Evaluate simple baselines on AbProp benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import math

import numpy as np
import pandas as pd
from scipy import stats

from abprop.baselines.simple_baselines import (
    BaselineResult,
    BigramPerplexityBaseline,
    KNNCdrBaseline,
    MeanLiabilityBaseline,
    MotifLiabilityBaseline,
    NearestNeighborLiabilityBaseline,
    RandomCDRBaseline,
    FrequencyCDRBaseline,
    SequenceSplit,
    UniformPerplexityBaseline,
    UnigramPerplexityBaseline,
    load_split,
)
from abprop.eval.metrics import classification_summary, regression_summary
from abprop.utils import load_yaml_config


@dataclass
class SignificanceResult:
    metric: str
    diff: float
    p_value: Optional[float]
    baseline_value: float
    abprop_value: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        return {
            "metric": self.metric,
            "diff": self.diff,
            "p_value": self.p_value,
            "baseline_value": self.baseline_value,
            "abprop_value": self.abprop_value,
        }


def resolve_dataset_path(data_cfg: Dict[str, object]) -> Path:
    processed_root = Path(data_cfg.get("processed_dir", "data/processed"))
    parquet_cfg = data_cfg.get("parquet", {})
    parquet_subdir = parquet_cfg.get("output_dir")
    dataset_root = processed_root / parquet_subdir if parquet_subdir else processed_root
    parquet_filename = parquet_cfg.get("filename")
    dataset_path = dataset_root / parquet_filename if parquet_filename else dataset_root
    if parquet_filename and not dataset_path.exists():
        dataset_path = dataset_root
    return dataset_path


def gather_abprop_metrics(abprop_root: Optional[Path], split: str) -> Dict[str, object]:
    if not abprop_root:
        return {}
    summary_path = abprop_root / split / "metrics.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    split_path = abprop_root / f"{split}.json"
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def evaluate_perplexity(
    train_split: SequenceSplit,
    eval_split: SequenceSplit,
) -> List[BaselineResult]:
    baselines = [
        UniformPerplexityBaseline(),
        UnigramPerplexityBaseline(),
        BigramPerplexityBaseline(),
    ]
    results: List[BaselineResult] = []
    for baseline in baselines:
        baseline.fit(train_split)
        result = baseline.evaluate(eval_split)
        results.append(result)
    return results


def evaluate_cdr(
    train_split: SequenceSplit,
    eval_split: SequenceSplit,
) -> List[BaselineResult]:
    baselines = [
        RandomCDRBaseline(),
        FrequencyCDRBaseline(),
        KNNCdrBaseline(),
    ]
    results: List[BaselineResult] = []
    for baseline in baselines:
        baseline.fit(train_split)
        results.append(baseline.evaluate(eval_split))
    return results


def evaluate_liability(
    train_split: SequenceSplit,
    eval_split: SequenceSplit,
) -> List[BaselineResult]:
    baselines = [
        MeanLiabilityBaseline(),
        NearestNeighborLiabilityBaseline(),
        MotifLiabilityBaseline(),
    ]
    results: List[BaselineResult] = []
    for baseline in baselines:
        baseline.fit(train_split)
        results.append(baseline.evaluate(eval_split))
    return results


def significance_perplexity(baseline: BaselineResult, abprop_metrics: Dict[str, object]) -> List[SignificanceResult]:
    abprop_seq = []
    seq_data = abprop_metrics.get("perplexity", {}).get("sequence_level")
    if isinstance(seq_data, list):
        abprop_seq = [float(item.get("perplexity", float("nan"))) for item in seq_data]
        abprop_seq = [x for x in abprop_seq if np.isfinite(x)]
    baseline_seq = [float(x) for x in baseline.per_item.get("perplexity", []) if np.isfinite(x)]
    if not baseline_seq or not abprop_seq:
        return []
    stat, p_value = stats.ttest_ind(baseline_seq, abprop_seq, equal_var=False)
    diff = float(np.mean(abprop_seq) - np.mean(baseline_seq))
    return [
        SignificanceResult(
            metric="perplexity",
            diff=diff,
            p_value=float(p_value),
            baseline_value=float(np.mean(baseline_seq)),
            abprop_value=float(np.mean(abprop_seq)),
        )
    ]


def significance_cdr(baseline: BaselineResult, abprop_metrics: Dict[str, object]) -> List[SignificanceResult]:
    abprop_conf = abprop_metrics.get("classification", {}).get("confusion_matrix", {})
    if not abprop_conf:
        return []
    base_conf = baseline.extra.get("confusion", {})
    tp_b = int(base_conf.get("tp", 0))
    fp_b = int(base_conf.get("fp", 0))
    tn_b = int(base_conf.get("tn", 0))
    fn_b = int(base_conf.get("fn", 0))
    tp_a = int(abprop_conf.get("tp", 0))
    fp_a = int(abprop_conf.get("fp", 0))
    tn_a = int(abprop_conf.get("tn", 0))
    fn_a = int(abprop_conf.get("fn", 0))

    def two_proportion_z(success1: int, total1: int, success2: int, total2: int) -> Optional[float]:
        if total1 == 0 or total2 == 0:
            return None
        p1 = success1 / total1
        p2 = success2 / total2
        pooled = (success1 + success2) / (total1 + total2)
        denom = pooled * (1 - pooled) * (1 / total1 + 1 / total2)
        if denom <= 0:
            return None
        z = (p1 - p2) / math.sqrt(denom)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return float(p_value)

    acc_b = (tp_b + tn_b) / max(1, tp_b + tn_b + fp_b + fn_b)
    acc_a = (tp_a + tn_a) / max(1, tp_a + tn_a + fp_a + fn_a)
    p_value = two_proportion_z(tp_b + tn_b, tp_b + tn_b + fp_b + fn_b, tp_a + tn_a, tp_a + tn_a + fp_a + fn_a)
    return [
        SignificanceResult(
            metric="accuracy",
            diff=acc_a - acc_b,
            p_value=p_value,
            baseline_value=acc_b,
            abprop_value=acc_a,
        )
    ]


def significance_regression(baseline: BaselineResult, abprop_metrics: Dict[str, object]) -> List[SignificanceResult]:
    # Without per-sample AbProp errors we cannot compute a robust test.
    return []


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple baselines.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--benchmarks", nargs="*", default=["perplexity", "cdr_classification", "liability"])
    parser.add_argument("--splits", nargs="*", default=["val"])
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--abprop-results", type=Path, default=None, help="Path to AbProp evaluation outputs (e.g., outputs/eval).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_cfg = load_yaml_config(args.data_config)
    dataset_path = resolve_dataset_path(data_cfg)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, object]] = []
    results_json: Dict[str, object] = {}

    abprop_metrics_root = args.abprop_results.resolve() if args.abprop_results else None

    for split in args.splits:
        train_split = load_split(dataset_path, "train")
        eval_split = load_split(dataset_path, split)
        split_results: Dict[str, object] = {}
        abprop_metrics = gather_abprop_metrics(abprop_metrics_root, split) if abprop_metrics_root else {}

        if "perplexity" in args.benchmarks:
            baseline_results = evaluate_perplexity(train_split, eval_split)
            enriched = []
            for result in baseline_results:
                sig = significance_perplexity(result, abprop_metrics) if abprop_metrics else []
                enriched.append(
                    {
                        "baseline": result.to_dict(),
                        "significance": [entry.to_dict() for entry in sig],
                    }
                )
                row = {
                    "split": split,
                    "benchmark": "perplexity",
                    "baseline": result.name,
                    **result.metrics,
                }
                summary_records.append(row)
            split_results["perplexity"] = enriched

        if "cdr_classification" in args.benchmarks:
            baseline_results = evaluate_cdr(train_split, eval_split)
            enriched = []
            for result in baseline_results:
                sig = significance_cdr(result, abprop_metrics) if abprop_metrics else []
                enriched.append(
                    {
                        "baseline": result.to_dict(),
                        "significance": [entry.to_dict() for entry in sig],
                    }
                )
                row = {
                    "split": split,
                    "benchmark": "cdr_classification",
                    "baseline": result.name,
                    **result.metrics,
                }
                summary_records.append(row)
            split_results["cdr_classification"] = enriched

        if "liability" in args.benchmarks:
            baseline_results = evaluate_liability(train_split, eval_split)
            enriched = []
            for result in baseline_results:
                sig = significance_regression(result, abprop_metrics) if abprop_metrics else []
                enriched.append(
                    {
                        "baseline": result.to_dict(),
                        "significance": [entry.to_dict() for entry in sig],
                    }
                )
                row = {
                    "split": split,
                    "benchmark": "liability",
                    "baseline": result.name,
                    **result.metrics,
                }
                summary_records.append(row)
            split_results["liability"] = enriched

        results_json[split] = split_results

    with open(output_root / "baseline_results.json", "w", encoding="utf-8") as handle:
        json.dump(results_json, handle, indent=2)

    df = pd.DataFrame(summary_records)
    df.to_csv(output_root / "baseline_summary.csv", index=False)
    print(f"Saved baseline results to {output_root}")


if __name__ == "__main__":
    main()
