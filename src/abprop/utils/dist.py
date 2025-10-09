"""Distributed training utilities."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed(mode: str = "ddp") -> dict:
    """Initialise torch.distributed if requested."""
    mode = (mode or "none").lower()
    if mode == "none" or mode == "cpu":
        return {
            "is_distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return {
        "is_distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    if not torch.distributed.is_initialized() or not torch.cuda.is_available():
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def is_rank_zero() -> bool:
    return get_rank() == 0


@contextmanager
def rank_zero_only() -> Iterator[None]:
    if is_rank_zero():
        yield
    else:
        yield


def cleanup() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

