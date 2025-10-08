#!/bin/bash
#SBATCH --job-name=abprop-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load PyTorch/2.1.2 CUDA/12.1

# NCCL environment tuning (adjust for your system)
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^lo,docker

# Torch distributed environment
export MASTER_ADDR="$(hostname -s)"
export MASTER_PORT="${MASTER_PORT:-6000}"
export WORLD_SIZE=$SLURM_NTASKS

echo "Launching single-node AbProp training on ${SLURM_JOB_NODELIST}"

srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    scripts/train.py \
    --config-path configs/train.yaml \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml

