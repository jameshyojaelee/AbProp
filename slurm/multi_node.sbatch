#!/bin/bash
#SBATCH --job-name=abprop-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load PyTorch/2.1.2 CUDA/12.1

# NCCL environment suggestions for multi-node runs
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_NET_GDR_LEVEL=PHB

# Slurm provides MASTER_ADDR/MASTER_PORT via srun; derive from first node
export MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-6200}"
export WORLD_SIZE=$SLURM_NTASKS

echo "Launching multi-node AbProp training on ${SLURM_JOB_NODELIST}"

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    scripts/train.py \
    --config-path configs/train.yaml \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml

