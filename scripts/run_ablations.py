#!/usr/bin/env python3
"""Automated ablation runner for AbProp.

Reads a declarative plan, materialises per-experiment configuration files,
launches training jobs sequentially (or in dry-run mode), and records metadata
for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at the top level.")
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name.strip())
    return safe.strip("-_") or "experiment"


def load_experiment_file(path: Path) -> List[Dict[str, Any]]:
    data = load_yaml(path)
    experiments = data.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError(f"File {path} must contain an 'experiments' list.")
    validated: List[Dict[str, Any]] = []
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError(f"Experiment entry in {path} is not a mapping: {entry}")
        if "name" not in entry:
            raise ValueError(f"Experiment entry in {path} missing 'name': {entry}")
        validated.append(entry)
    return validated


def gather_experiments(plan: Dict[str, Any], plan_path: Path) -> List[Dict[str, Any]]:
    include_map = {}
    for key, rel_path in plan.get("includes", {}).items():
        include_map[key] = (plan_path.parent / rel_path).resolve()

    experiments: List[Dict[str, Any]] = []
    for item in plan.get("experiments", []):
        if isinstance(item, dict) and "include" in item:
            include_key = item["include"]
            if include_key not in include_map:
                raise ValueError(f"Include key '{include_key}' not defined in plan.")
            experiments.extend(load_experiment_file(include_map[include_key]))
        elif isinstance(item, dict):
            if "name" not in item:
                raise ValueError(f"Experiment entry missing 'name': {item}")
            experiments.append(item)
        else:
            raise ValueError(f"Invalid experiment specification: {item}")
    return experiments


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def build_command(
    python_exec: Path,
    train_script: Path,
    train_cfg: Path,
    model_cfg: Path,
    data_cfg: Path,
    output_dir: Path,
    extra_args: Iterable[str],
    seed: Optional[int],
) -> List[str]:
    cmd = [str(python_exec), str(train_script)]
    cmd.extend(["--config-path", str(train_cfg)])
    cmd.extend(["--model-config", str(model_cfg)])
    cmd.extend(["--data-config", str(data_cfg)])
    cmd.extend(["--output-dir", str(output_dir)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    cmd.extend(extra_args)
    return cmd


def summarise_metrics(log_dir: Path) -> Dict[str, Any]:
    csv_path = log_dir / "abprop-train.csv"
    if not csv_path.exists():
        return {}
    final_step = -1
    final_metrics: Dict[str, float] = {}
    loss_min = float("inf")
    loss_sum = 0.0
    loss_count = 0
    step_times: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("tag") != "train":
                continue
            try:
                step = int(row["step"])
                metric = row["metric"]
                value = float(row["value"])
            except (KeyError, ValueError):
                continue
            if metric == "loss":
                loss_sum += value
                loss_count += 1
                if value < loss_min:
                    loss_min = value
            if metric == "step_time":
                step_times.append(value)
            if step >= final_step:
                final_step = step
                final_metrics.setdefault("metrics", {})
                if "metrics" not in final_metrics:
                    final_metrics["metrics"] = {}
                final_metrics["metrics"][metric] = value
    if final_step >= 0:
        final_metrics["final_step"] = final_step
    if loss_count > 0:
        final_metrics["loss_min"] = loss_min
        final_metrics["loss_mean"] = loss_sum / loss_count
    if step_times:
        final_metrics["step_time_avg"] = sum(step_times) / len(step_times)
    return final_metrics


def resolve_path(path_like: Any, repo_root: Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def run_experiment(
    plan_name: str,
    defaults: Dict[str, Any],
    experiment: Dict[str, Any],
    plan_path: Path,
    repo_root: Path,
    dry_run: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    python_exec = resolve_path(defaults.get("python_executable", sys.executable), repo_root)
    train_script = resolve_path(defaults.get("train_script", "scripts/train.py"), repo_root)
    base_output_dir = resolve_path(defaults.get("base_output_dir", "outputs/ablations"), repo_root)
    base_train_cfg = load_yaml(resolve_path(defaults.get("train_config", "configs/train.yaml"), repo_root))
    base_model_cfg = load_yaml(resolve_path(defaults.get("model_config", "configs/model.yaml"), repo_root))
    base_data_cfg = load_yaml(resolve_path(defaults.get("data_config", "configs/data.yaml"), repo_root))

    exp_name = sanitize_name(experiment["name"])
    run_root = base_output_dir / plan_name / exp_name
    configs_dir = run_root / "configs"
    artifacts_dir = run_root / "artifacts"
    ensure_dir(configs_dir)
    ensure_dir(artifacts_dir)

    train_cfg = json.loads(json.dumps(base_train_cfg))
    model_cfg = json.loads(json.dumps(base_model_cfg))
    data_cfg = json.loads(json.dumps(base_data_cfg))

    train_overrides = experiment.get("train_overrides", {})
    model_overrides = experiment.get("model_overrides", {})
    data_overrides = experiment.get("data_overrides", {})
    combined_overrides = experiment.get("overrides", {})

    if isinstance(combined_overrides, dict):
        train_overrides = deep_update(train_overrides, combined_overrides.get("train", {}))
        model_overrides = deep_update(model_overrides, combined_overrides.get("model", {}))
        data_overrides = deep_update(data_overrides, combined_overrides.get("data", {}))

    deep_update(train_cfg, train_overrides or {})
    deep_update(model_cfg, model_overrides or {})
    deep_update(data_cfg, data_overrides or {})

    train_cfg["output_dir"] = str(artifacts_dir)
    train_cfg["log_dir"] = str(artifacts_dir / "logs")
    train_cfg["checkpoint_dir"] = str(artifacts_dir / "checkpoints")
    train_cfg.setdefault("mlflow", {})
    if not isinstance(train_cfg["mlflow"], dict):
        train_cfg["mlflow"] = {}
    tags = train_cfg["mlflow"].get("tags", {})
    if not isinstance(tags, dict):
        tags = {}
    tags.update(
        {
            "ablation_plan": plan_name,
            "ablation_experiment": exp_name,
        }
    )
    if "tags" in experiment:
        for key, value in experiment["tags"].items():
            tags[str(key)] = value
    train_cfg["mlflow"]["tags"] = tags

    seed = experiment.get("seed", defaults.get("seed"))
    if seed is not None:
        train_cfg["seed"] = int(seed)

    write_yaml(configs_dir / "train.yaml", train_cfg)
    write_yaml(configs_dir / "model.yaml", model_cfg)
    write_yaml(configs_dir / "data.yaml", data_cfg)

    extra_args = list(chain(defaults.get("extra_args", []), experiment.get("extra_args", [])))
    output_dir = artifacts_dir
    cmd = build_command(
        python_exec,
        train_script,
        configs_dir / "train.yaml",
        configs_dir / "model.yaml",
        configs_dir / "data.yaml",
        output_dir,
        extra_args,
        seed=int(seed) if seed is not None else None,
    )

    metadata = {
        "plan": plan_name,
        "experiment": exp_name,
        "description": experiment.get("description", ""),
        "command": cmd,
        "status": "pending",
        "start_time": datetime.utcnow().isoformat() + "Z",
        "run_root": str(run_root),
        "config_paths": {
            "train": str(configs_dir / "train.yaml"),
            "model": str(configs_dir / "model.yaml"),
            "data": str(configs_dir / "data.yaml"),
        },
        "tags": tags,
        "seed": seed,
    }

    metadata_path = run_root / "metadata.json"
    env = os.environ.copy()
    if defaults.get("env"):
        env.update({str(k): str(v) for k, v in defaults["env"].items()})
    if experiment.get("env"):
        env.update({str(k): str(v) for k, v in experiment["env"].items()})
    if env_overrides:
        env.update(env_overrides)

    if dry_run:
        metadata["status"] = "skipped"
        metadata["dry_run"] = True
        metadata["end_time"] = datetime.utcnow().isoformat() + "Z"
        metadata["duration_seconds"] = 0.0
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"[DRY RUN] {exp_name}: {' '.join(cmd)}")
        return metadata

    env.setdefault("PYTHONPATH", "src")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=False,
            capture_output=False,
        )
        metadata["returncode"] = result.returncode
        metadata["status"] = "completed" if result.returncode == 0 else "failed"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise
    finally:
        metadata["end_time"] = datetime.utcnow().isoformat() + "Z"
        start = datetime.fromisoformat(metadata["start_time"].rstrip("Z"))
        end = datetime.fromisoformat(metadata["end_time"].rstrip("Z"))
        metadata["duration_seconds"] = max(0.0, (end - start).total_seconds())
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    summary = summarise_metrics(Path(train_cfg["log_dir"]))
    if summary:
        with open(run_root / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        metadata["summary"] = summary

    return metadata


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Automated ablation sweep runner for AbProp.")
    parser.add_argument("--plan", type=Path, required=True, help="Path to ablation plan YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--filter", nargs="*", help="Subset of experiment names to run.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing metadata.json.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of experiments executed.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan_path = args.plan.resolve()
    resolved_plan = plan_path.resolve()
    parents = resolved_plan.parents
    repo_root = parents[2] if len(parents) >= 3 else resolved_plan.parent
    plan = load_yaml(plan_path)
    plan_name = sanitize_name(plan.get("plan_name", plan_path.stem))
    defaults = plan.get("defaults", {})
    experiments = gather_experiments(plan, plan_path)

    if args.filter:
        filters = {sanitize_name(name) for name in args.filter}
        experiments = [exp for exp in experiments if sanitize_name(exp["name"]) in filters]
        if not experiments:
            print(f"No experiments match filters: {args.filter}")
            return

    results: List[Dict[str, Any]] = []
    executed = 0
    for exp in experiments:
        if args.max_runs is not None and executed >= args.max_runs:
            break
        exp_name = sanitize_name(exp["name"])
        base_output_dir = Path(defaults.get("base_output_dir", "outputs/ablations"))
        run_root = base_output_dir / plan_name / exp_name
        metadata_path = run_root / "metadata.json"
        if args.skip_existing and metadata_path.exists():
            print(f"[SKIP] {exp_name} (metadata already exists)")
            continue
        metadata = run_experiment(
            plan_name,
            defaults,
            exp,
            plan_path=plan_path,
            repo_root=repo_root,
            dry_run=args.dry_run,
        )
        results.append(metadata)
        if metadata.get("status") == "completed":
            executed += 1

    index_path = resolve_path(defaults.get("base_output_dir", "outputs/ablations"), repo_root) / plan_name / "plan_index.json"
    ensure_dir(index_path.parent)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "plan": plan_name,
                "plan_path": str(plan_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "experiments": results,
            },
            handle,
            indent=2,
        )

    print(f"Ablation plan '{plan_name}' processed. Experiments: {len(results)}")


if __name__ == "__main__":
    main()
