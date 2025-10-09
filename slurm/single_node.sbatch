#!/bin/bash
#SBATCH --job-name=abprop-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load Miniconda3/23.10.0-1
module load PyTorch/2.1.2-foss-2023b
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

# Source shared NCCL environment configuration
source slurm/env_nccl.sh
export NCCL_DEBUG=${NCCL_DEBUG:-warn}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# Torch distributed environment
export MASTER_ADDR="$(hostname -s)"
export MASTER_PORT="${MASTER_PORT:-6000}"
export WORLD_SIZE=$SLURM_NTASKS

echo "Launching single-node AbProp training on ${SLURM_JOB_NODELIST}"
echo "srun --gres=gpu:l40s:${SLURM_GPUS_PER_NODE} --ntasks-per-node=${SLURM_GPUS_PER_NODE} torchrun ..."

srun --gres=gpu:l40s:${SLURM_GPUS_PER_NODE} --ntasks-per-node=${SLURM_GPUS_PER_NODE} \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    scripts/train.py \
    --config-path configs/train.yaml \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml \
    --distributed ddp
