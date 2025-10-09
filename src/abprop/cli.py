"""CLI entrypoints for AbProp console scripts."""

from __future__ import annotations

from .commands.etl import main as run_etl
from .commands.train import main as run_train
from .commands.eval import main as run_eval
from .commands.launch import main as run_launch

__all__ = ["run_etl", "run_train", "run_eval", "run_launch"]
