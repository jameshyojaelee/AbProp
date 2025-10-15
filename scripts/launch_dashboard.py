#!/usr/bin/env python3
"""Launch the AbProp Streamlit dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"), help="Root directory for dashboard assets.")
    parser.add_argument("--config", type=Path, help="Optional JSON config overriding asset directories.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit server port.")
    parser.add_argument("--address", type=str, default="localhost", help="Server address (default: localhost).")
    parser.add_argument("--headless", action="store_true", help="Run without opening a browser window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_path = Path(__file__).resolve().parents[1] / "src" / "abprop" / "viz" / "dashboard.py"
    if not dashboard_path.is_file():
        raise SystemExit(f"Dashboard entry point not found: {dashboard_path}")

    env = os.environ.copy()
    env["ABPROP_DASHBOARD_ROOT"] = str(args.root.resolve())
    if args.config:
        env["ABPROP_DASHBOARD_CONFIG"] = str(args.config.resolve())

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])

    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()

