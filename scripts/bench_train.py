#!/usr/bin/env python
"""Train-time benchmarking utility for AbProp."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from abprop.data import build_collate_fn
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import load_yaml_config


@dataclass
class BenchConfig:
    batch_size: int
    seq_len: int
    steps: int


class SyntheticDataset:
    def __init__(self, sequence: str, size: int) -> None:
        self.samples = [
            {"sequence": sequence, "chain": "H", "liability_ln": {}}
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_model(cfg: Dict) -> AbPropModel:
    config = TransformerConfig(
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 1024),
        dropout=cfg.get("dropout", 0.1),
    )
    model = AbPropModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def run_benchmark(model_cfg: Dict, bench_cfg: BenchConfig) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    seq = "A" * bench_cfg.seq_len
    dataset = SyntheticDataset(seq, bench_cfg.batch_size * (bench_cfg.steps + 1))
    collate = build_collate_fn(generate_mlm=True)
    dataloader = DataLoader(dataset, batch_size=bench_cfg.batch_size, collate_fn=collate)
    iterator = iter(dataloader)

    samples_processed = 0
    padding_ratios: List[float] = []
    step_times: List[float] = []
    peak_mem = 0.0

    for _ in range(bench_cfg.steps):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_labels = batch.get("labels")
        if mlm_labels is not None:
            mlm_labels = mlm_labels.to(device)

        start = time.perf_counter()

        outputs = model(input_ids, attention_mask, mlm_labels=mlm_labels, tasks=("mlm",))
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.perf_counter() - start
        step_times.append(elapsed)

        samples_processed += input_ids.size(0)
        padding = 1.0 - attention_mask.float().mean().item()
        padding_ratios.append(padding)

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_mem = max(peak_mem, peak)
            torch.cuda.reset_peak_memory_stats(device)

    return {
        "samples_per_s": samples_processed / sum(step_times),
        "avg_padding": sum(padding_ratios) / len(padding_ratios),
        "avg_step_time": sum(step_times) / len(step_times),
        "peak_mem_mb": peak_mem,
    }


def format_markdown(results: List[Dict[str, float]], configs: List[BenchConfig]) -> str:
    header = "| Batch | Seq Len | Samples/s | Padding | Step Time (s) | Peak Mem (MB) |\n"
    header += "| --- | --- | --- | --- | --- | --- |\n"
    rows = []
    for cfg, metrics in zip(configs, results):
        rows.append(
            f"| {cfg.batch_size} | {cfg.seq_len} | {metrics['samples_per_s']:.2f} | "
            f"{metrics['avg_padding']:.3f} | {metrics['avg_step_time']:.3f} | {metrics['peak_mem_mb']:.1f} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AbProp training throughput.")
    parser.add_argument("--config", type=Path, default=Path("configs/bench.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks.md"))
    args = parser.parse_args()

    model_cfg = load_yaml_config(args.model_config)
    bench_yaml = load_yaml_config(args.config)
    benches = [BenchConfig(**entry) for entry in bench_yaml.get("experiments", [])]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for bench in benches:
        print(f"Running: batch={bench.batch_size}, seq_len={bench.seq_len}, steps={bench.steps}")
        metrics = run_benchmark(model_cfg, bench)
        results.append(metrics)

    markdown = format_markdown(results, benches)
    args.output.write_text(markdown)
    print("\nBenchmark summary written to", args.output)
    print(markdown)


if __name__ == "__main__":
    main()

