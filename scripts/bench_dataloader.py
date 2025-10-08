#!/usr/bin/env python
"""Benchmark AbProp dataloader throughput and padding efficiency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

from abprop.data import BucketBatchSampler, OASDataset, build_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OAS dataloader throughput.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to processed Parquet dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to benchmark (train/val/test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Maximum batch size.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional maximum tokens per batch (after padding).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Length bins used by the bucket sampler.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of iterations to benchmark.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle batches within length buckets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = OASDataset(args.data_dir, split=args.split)
    sampler = BucketBatchSampler(
        dataset.lengths,
        batch_size=args.batch_size,
        bins=args.bins,
        max_tokens=args.max_tokens,
        shuffle=args.shuffle,
    )
    collate_fn = build_collate_fn(generate_mlm=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    total_samples = 0
    padding_ratios = []
    start_time = time.perf_counter()

    for step, batch in enumerate(dataloader, start=1):
        total_samples += batch["input_ids"].size(0)
        mask = batch["attention_mask"]
        padding_ratio = 1.0 - mask.float().mean().item()
        padding_ratios.append(padding_ratio)
        if step >= args.steps:
            break

    elapsed = time.perf_counter() - start_time
    samples_per_sec = total_samples / max(elapsed, 1e-6)
    print(f"Samples processed: {total_samples}")
    print(f"Elapsed time     : {elapsed:.2f}s")
    print(f"Throughput       : {samples_per_sec:.2f} samples/s")
    if padding_ratios:
        print(f"Average padding  : {mean(padding_ratios):.2%}")


if __name__ == "__main__":
    main()

