#!/bin/bash
# Source this file inside your Slurm scripts to configure NCCL + TorchDistributed.

# NCCL tuning knobs (adjust to match fabric)
export NCCL_DEBUG=${NCCL_DEBUG:-warn}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}      # set to 1 if InfiniBand unavailable
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}

# Torch distributed rendezvous defaults
export MASTER_PORT=${MASTER_PORT:-6000}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "NCCL environment configured:"
echo "  NCCL_DEBUG=${NCCL_DEBUG}"
echo "  NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
echo "  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "  NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
echo "  MASTER_PORT=${MASTER_PORT}"

