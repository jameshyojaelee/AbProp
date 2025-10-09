"""Lightweight MLflow wrappers with safe fallbacks."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import json
import subprocess


def _ensure_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    default_uri = str(Path("./mlruns").resolve())
    os.environ["MLFLOW_TRACKING_URI"] = default_uri
    return default_uri


def _import_mlflow():
    try:
        import mlflow  # type: ignore
    except ImportError:
        return None
    return mlflow


@contextmanager
def mlflow_run(run_name: str, tags: Optional[Mapping[str, Any]] = None):
    mlflow = _import_mlflow()
    if mlflow is None:
        yield None
        return

    _ensure_tracking_uri()
    run = mlflow.start_run(run_name=run_name)
    if tags:
        safe_tags = {str(k): str(v) for k, v in tags.items()}
        mlflow.set_tags(safe_tags)
    try:
        yield run
    finally:
        mlflow.end_run()


def log_params(params: Mapping[str, Any]) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    flat = {str(k): str(v) for k, v in params.items()}
    mlflow.log_params(flat)


def log_metrics(metrics: Mapping[str, float], step: Optional[int] = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    flat = {str(k): float(v) for k, v in metrics.items()}
    mlflow.log_metrics(flat, step=step)


def log_artifact(path: Path | str, artifact_path: Optional[str] = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_dict(data: Mapping[str, Any], artifact_file: str) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    mlflow.log_dict(data, artifact_file)


def default_tags() -> Dict[str, str]:
    tags: Dict[str, str] = {}
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        tags["slurm_job_id"] = slurm_job_id
    if os.getenv("SLURM_JOB_NODELIST"):
        tags["slurm_nodes"] = os.getenv("SLURM_JOB_NODELIST")
    if os.getenv("SLURM_GPUS_PER_NODE"):
        tags["slurm_gpus_per_node"] = os.getenv("SLURM_GPUS_PER_NODE")
    git_sha = _git_sha()
    if git_sha:
        tags["git_sha"] = git_sha
    return tags


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pragma: no cover - git optional
        return None


__all__ = [
    "mlflow_run",
    "log_params",
    "log_metrics",
    "log_artifact",
    "log_dict",
    "default_tags",
]

