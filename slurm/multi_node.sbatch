#!/bin/bash
#SBATCH --job-name=abprop-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=1               # only one Slurm task
#SBATCH --cpus-per-task=12       # adjust CPUs per GPU
#SBATCH --gpus-per-node=3
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

module load Miniconda3/23.10.0-1
module load PyTorch/2.1.2-foss-2023b
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
source slurm/env_nccl.sh
export NCCL_DEBUG=${NCCL_DEBUG:-warn}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-6200}

echo "Launching multi-node AbProp training on ${SLURM_JOB_NODELIST}"
echo "Command: torchrun --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${SLURM_GPUS_PER_NODE} ..."

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    python -m abprop.commands.train \
    --distributed ddp \
    --config-path configs/train.yaml
