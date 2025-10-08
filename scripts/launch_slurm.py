#!/usr/bin/env python
"""Helper to launch AbProp jobs on Slurm."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Slurm launch commands for AbProp distributed training."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to request.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="Number of GPUs per node.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="Slurm partition or queue to submit the job to.",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="04:00:00",
        help="Walltime limit (HH:MM:SS).",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("scripts/train.py"),
        help="Training script to execute under torchrun.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Training configuration file to pass to the script.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> str:
    total_tasks = args.nodes * args.gpus_per_node
    cmd = (
        f"srun --nodes={args.nodes} --ntasks={total_tasks} "
        f"--gpus-per-node={args.gpus_per_node} --partition={args.partition} "
        f"--time={args.time} "
        f"torchrun --nproc_per_node={args.gpus_per_node} {args.script} "
        f"--config-path {args.config}"
    )
    return cmd


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print("Suggested Slurm launch command:")
    print(command)


if __name__ == "__main__":
    main()
