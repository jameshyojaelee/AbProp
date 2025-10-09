"""Launch Slurm jobs programmatically."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit Slurm jobs for distributed AbProp training.")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--time", type=str, default="08:00:00")
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--job-name", type=str, default="abprop-ddp")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--time-min", type=str, default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser


def build_sbatch_contents(args: argparse.Namespace, repo_root: Path) -> str:
    config_path = Path(args.config).resolve()
    train_script = (repo_root / "scripts" / "train.py").resolve()
    env_nccl = (repo_root / "slurm" / "env_nccl.sh").resolve()

    extra = " ".join(shlex.quote(token) for token in args.extra) if args.extra else ""
    torchrun_cmd = (
        f"torchrun --nnodes=${{SLURM_JOB_NUM_NODES}} "
        f"--nproc_per_node={args.gpus_per_node} "
        f"--rdzv_backend=c10d --rdzv_endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} "
        f"{shlex.quote(str(train_script))} --distributed ddp --config-path {shlex.quote(str(config_path))} "
        f"--fsdp off --grad_ckpt false {extra}".strip()
    )

    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --ntasks-per-node={args.gpus_per_node}",
        f"#SBATCH --gpus-per-node={args.gpus_per_node}",
        f"#SBATCH --time={args.time}",
        "#SBATCH --output=logs/%x-%j.out",
    ]
    if args.time_min:
        header.append(f"#SBATCH --time-min={args.time_min}")

    body = f"""
set -euo pipefail

module load PyTorch/2.1.2 CUDA/12.1
source {env_nccl}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=${{MASTER_PORT:-6000}}
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * {args.gpus_per_node}))

echo "Executing: srun {torchrun_cmd}"
srun {torchrun_cmd}
""".strip()

    return "\n".join(header + [body]) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    sbatch_contents = build_sbatch_contents(args, repo_root)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(sbatch_contents)

    print(f"Submitting job with temporary sbatch file: {tmp_path}")
    try:
        subprocess.run(["sbatch", str(tmp_path)], check=True)
    finally:
        print(f"Sbatch file retained at: {tmp_path}")


__all__ = ["main", "build_parser"]

