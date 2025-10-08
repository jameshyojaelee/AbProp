"""AbProp: Antibody sequence property modeling toolkit."""

from importlib import metadata


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("abprop")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)


__all__ = ["__version__"]

