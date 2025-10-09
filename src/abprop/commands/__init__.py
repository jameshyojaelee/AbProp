"""Command-line entrypoints for AbProp."""

from __future__ import annotations

from .etl import main as etl_main
from .train import main as train_main
from .eval import main as eval_main
from .launch import main as launch_main

__all__ = ["etl_main", "train_main", "eval_main", "launch_main"]

