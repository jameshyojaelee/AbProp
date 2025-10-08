"""Utility helpers for configuration, logging, and reproducibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

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
]
