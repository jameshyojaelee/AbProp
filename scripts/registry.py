#!/usr/bin/env python3
"""CLI for the AbProp model registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from abprop.registry import ModelRegistry
from abprop.utils import load_yaml_config


def parse_key_value(pairs: list[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid metric spec: {item}")
        key, value = item.split("=", 1)
        metrics[key] = float(value)
    return metrics


def load_config(path: Path | None) -> Dict[str, object]:
    if path is None:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml_config(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return {k: float(v) for k, v in data.items()}
    raise ValueError("Metrics file must contain a JSON object.")


def cmd_register(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry)
    config = load_config(args.config)
    metrics = load_metrics(args.metrics_file)
    metrics.update(parse_key_value(args.metric or []))
    record = registry.register(
        model_id=args.id,
        checkpoint=args.checkpoint,
        config=config,
        metrics=metrics,
        tags=args.tags.split(",") if args.tags else [],
        notes=args.notes or "",
    )
    print(json.dumps(record.to_dict(), indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry)
    records = registry.list()
    if not records:
        print("<empty>")
        return
    print("ID\tCheckpoint\tMetric Keys\tTags\tCreated")
    for record in records:
        print(
            f"{record.model_id}\t{record.checkpoint}\t"
            f"{','.join(record.metrics.keys())}\t{','.join(record.tags)}\t{record.created_at}"
        )


def cmd_show(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry)
    record = registry.get(args.id)
    if not record:
        raise SystemExit(f"Model id '{args.id}' not found")
    print(json.dumps(record.to_dict(), indent=2))


def cmd_delete(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry)
    registry.delete(args.id)
    print(f"Deleted {args.id}")


def cmd_best(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry)
    record = registry.best(args.metric, higher_is_better=args.higher)
    if not record:
        raise SystemExit(f"No model contains metric '{args.metric}'")
    print(json.dumps(record.to_dict(), indent=2))


def cmd_export_card(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry)
    path = registry.export_card(args.id, args.output)
    print(f"Wrote card to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("models/registry.json"),
        help="Registry JSON path (default: models/registry.json).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_register = subparsers.add_parser("register", help="Register a new model entry.")
    p_register.add_argument("--id", required=True)
    p_register.add_argument("--checkpoint", required=True)
    p_register.add_argument("--config", type=Path)
    p_register.add_argument("--metrics-file", type=Path)
    p_register.add_argument("--metric", action="append", help="Inline metric key=value")
    p_register.add_argument("--tags", help="Comma-separated tags")
    p_register.add_argument("--notes")
    p_register.set_defaults(func=cmd_register)

    p_list = subparsers.add_parser("list", help="List registered models.")
    p_list.set_defaults(func=cmd_list)

    p_show = subparsers.add_parser("show", help="Show details for a model id.")
    p_show.add_argument("--id", required=True)
    p_show.set_defaults(func=cmd_show)

    p_delete = subparsers.add_parser("delete", help="Remove a model from the registry.")
    p_delete.add_argument("--id", required=True)
    p_delete.set_defaults(func=cmd_delete)

    p_best = subparsers.add_parser("best", help="Select the best model by a metric.")
    p_best.add_argument("--metric", required=True)
    p_best.add_argument("--higher", action="store_true", help="Treat higher values as better.")
    p_best.set_defaults(func=cmd_best)

    p_card = subparsers.add_parser("export-card", help="Export a Markdown model card.")
    p_card.add_argument("--id", required=True)
    p_card.add_argument("--output", type=Path, required=True)
    p_card.set_defaults(func=cmd_export_card)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

