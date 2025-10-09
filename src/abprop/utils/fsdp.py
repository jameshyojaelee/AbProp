"""Helpers for configuring PyTorch Fully Sharded Data Parallel (FSDP)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

try:  # PyTorch 2.0+
    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    _HAS_FSDP = True
except ImportError:  # pragma: no cover - environment without FSDP support
    FSDP = None  # type: ignore
    transformer_auto_wrap_policy = None  # type: ignore
    _HAS_FSDP = False

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # noqa: E402
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    _HAS_CKPT = True
except ImportError:  # pragma: no cover - activation checkpointing unavailable
    CheckpointImpl = None  # type: ignore
    apply_activation_checkpointing = None  # type: ignore
    checkpoint_wrapper = None  # type: ignore
    _HAS_CKPT = False


@dataclass
class FSDPConfig:
    """Configuration for wrapping models with FSDP."""

    sharding: str = "full_shard"  # currently supports "full_shard" or "no_shard"
    cpu_offload: bool = False
    use_mixed_precision: bool = True
    mixed_precision_dtype: Optional[torch.dtype] = None
    activation_checkpointing: bool = False
    limit_all_gathers: bool = True


def _infer_mixed_precision_dtype(config: FSDPConfig) -> Optional[torch.dtype]:
    if not config.use_mixed_precision or not torch.cuda.is_available():
        return None
    if config.mixed_precision_dtype is not None:
        return config.mixed_precision_dtype
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _build_mixed_precision(config: FSDPConfig) -> Optional[MixedPrecision]:
    dtype = _infer_mixed_precision_dtype(config)
    if dtype is None:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=torch.float32)


def _default_auto_wrap_policy() -> callable:
    if not _HAS_FSDP:
        raise RuntimeError("FSDP is not available in this environment.")
    return transformer_auto_wrap_policy({nn.TransformerEncoderLayer})


def enable_activation_checkpointing(model: nn.Module) -> None:
    """Apply activation checkpointing to transformer blocks."""

    if not _HAS_CKPT:
        raise RuntimeError("Activation checkpointing requested but not supported in this PyTorch build.")
    wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    check_fn = lambda module: isinstance(module, nn.TransformerEncoderLayer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=check_fn,
    )


def wrap_fsdp_model(model: nn.Module, config: FSDPConfig) -> nn.Module:
    """Wrap the provided model with FSDP using a transformer-aware policy."""

    if not _HAS_FSDP:
        raise RuntimeError("FSDP is not available in this environment.")
    if not dist.is_initialized():  # pragma: no cover - guarded by caller
        raise RuntimeError("Distributed process group must be initialised before enabling FSDP.")

    if config.activation_checkpointing:
        enable_activation_checkpointing(model)

    sharding_strategy = (
        ShardingStrategy.FULL_SHARD if config.sharding == "full_shard" else ShardingStrategy.NO_SHARD
    )

    mixed_precision = _build_mixed_precision(config)
    cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else None

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=_default_auto_wrap_policy(),
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        device_id=device_id,
        limit_all_gathers=config.limit_all_gathers,
    )
    return fsdp_model


__all__ = ["FSDPConfig", "wrap_fsdp_model", "enable_activation_checkpointing"]
