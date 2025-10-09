"""AbProp: Antibody sequence property modeling toolkit."""

from __future__ import annotations

from importlib import metadata

from . import cli, commands, data, eval, models, tokenizers, train, utils


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("abprop")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)


__all__ = [
    "cli",
    "commands",
    "data",
    "eval",
    "models",
    "tokenizers",
    "train",
    "utils",
    "__version__",
]
