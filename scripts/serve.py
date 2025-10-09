#!/usr/bin/env python
"""Run the AbProp FastAPI server."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from abprop.server.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the AbProp inference API.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.checkpoint, args.model_config, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

