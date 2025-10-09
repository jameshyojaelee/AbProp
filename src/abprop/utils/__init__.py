"""Utility helpers for configuration, logging, and reproducibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .dist import (
    barrier,
    cleanup,
    get_rank,
    get_world_size,
    init_distributed,
    is_rank_zero,
    rank_zero_only,
    seed_all,
    wrap_ddp,
)
from .fsdp import FSDPConfig, enable_activation_checkpointing, wrap_fsdp_model
from .mlflowx import (
    default_tags as mlflow_default_tags,
    log_artifact as mlflow_log_artifact,
    log_dict as mlflow_log_dict,
    log_metrics as mlflow_log_metrics,
    log_params as mlflow_log_params,
    mlflow_run,
)
from .liabilities import find_motifs, normalize_by_length


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")


def load_yaml_config(path: Path | str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "load_yaml_config",
    "find_motifs",
    "normalize_by_length",
    "init_distributed",
    "wrap_ddp",
    "seed_all",
    "barrier",
    "cleanup",
    "get_rank",
    "get_world_size",
    "is_rank_zero",
    "rank_zero_only",
    "FSDPConfig",
    "wrap_fsdp_model",
    "enable_activation_checkpointing",
    "mlflow_run",
    "mlflow_log_params",
    "mlflow_log_metrics",
    "mlflow_log_artifact",
    "mlflow_log_dict",
    "mlflow_default_tags",
]
