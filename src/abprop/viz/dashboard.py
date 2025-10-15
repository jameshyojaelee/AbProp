"""Streamlit dashboard for AbProp exploration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from abprop.models import AbPropModel, TransformerConfig
from abprop.tokenizers.aa import collate_batch


@dataclass
class DashboardConfig:
    runs_dir: Path
    benchmarks_dir: Path
    attention_dir: Path
    embeddings_dir: Path
    eval_dir: Path
    checkpoints_dir: Path


def load_config() -> DashboardConfig:
    root = Path(os.environ.get("ABPROP_DASHBOARD_ROOT", "outputs")).resolve()
    config_path = Path(os.environ.get("ABPROP_DASHBOARD_CONFIG", ""))
    config_data: Dict[str, str] = {}
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    return DashboardConfig(
        runs_dir=Path(config_data.get("runs_dir", root / "runs")),
        benchmarks_dir=Path(config_data.get("benchmarks_dir", root / "benchmarks")),
        attention_dir=Path(config_data.get("attention_dir", root / "attention")),
        embeddings_dir=Path(config_data.get("embeddings_dir", "docs/figures/embeddings")).resolve(),
        eval_dir=Path(config_data.get("eval_dir", root / "eval")),
        checkpoints_dir=Path(config_data.get("checkpoints_dir", root / "checkpoints")),
    )


def page_overview(cfg: DashboardConfig) -> None:
    st.title("AbProp Dashboard")
    st.write("Quick entry points into the latest training/evaluation artifacts.")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Runs", len(list(cfg.runs_dir.glob("*/"))) if cfg.runs_dir.exists() else 0)
        st.metric("Checkpoints", len(list(cfg.checkpoints_dir.glob("*.pt"))) if cfg.checkpoints_dir.exists() else 0)
    with cols[1]:
        st.metric("Benchmark suites", len(list(cfg.benchmarks_dir.glob("*.json"))) if cfg.benchmarks_dir.exists() else 0)
        st.metric("Attention studies", len(list(cfg.attention_dir.glob("*/aggregated"))) if cfg.attention_dir.exists() else 0)
    with cols[2]:
        st.metric("Embedding drops", len(list(cfg.embeddings_dir.glob("*.json"))) if cfg.embeddings_dir.exists() else 0)
        st.metric("Evaluation reports", len(list(cfg.eval_dir.glob("*.json"))) if cfg.eval_dir.exists() else 0)

    st.subheader("How to use this dashboard")
    st.markdown(
        """
        - Select a workspace page from the sidebar.
        - Point `ABPROP_DASHBOARD_ROOT` to your experiment directory or override individual paths via `ABPROP_DASHBOARD_CONFIG`.
        - Drop PNG/HTML artifacts produced by the visualization scripts into the suggested directories for instant previews.
        - Review narrative context in [docs/case_studies/README.md](../../docs/case_studies/README.md) and link key examples here.
        """
    )


@st.cache_data(show_spinner=False)
def load_benchmark_table(path: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            with open(file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload.setdefault("benchmark", file.stem)
            records.append(payload)
    return pd.DataFrame(records)


def page_benchmarks(cfg: DashboardConfig) -> None:
    st.title("Benchmark Explorer")
    if not cfg.benchmarks_dir.exists():
        st.warning(f"No benchmark artifacts found in {cfg.benchmarks_dir}")
        return
    table = load_benchmark_table(cfg.benchmarks_dir)
    if table.empty:
        st.info("No benchmark JSON files yet. Run `scripts/run_benchmarks.py` to populate them.")
        return
    available_benchmarks = sorted(table["benchmark"].unique()) if "benchmark" in table.columns else []
    if available_benchmarks:
        selected = st.multiselect("Filter benchmarks", available_benchmarks, default=available_benchmarks)
        table = table[table["benchmark"].isin(selected)]
    st.dataframe(table, use_container_width=True)
    st.download_button(
        label="Download Benchmark Table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="benchmarks.csv",
        mime="text/csv",
    )


@st.cache_data(show_spinner=False)
def load_image_paths(root: Path) -> Dict[str, List[Path]]:
    gallery: Dict[str, List[Path]] = {}
    if not root.exists():
        return gallery
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        aggregated = run_dir / "aggregated"
        if aggregated.is_dir():
            gallery[run_dir.name] = sorted(list(aggregated.glob("*.png")))
    return gallery


def page_attention(cfg: DashboardConfig) -> None:
    st.title("Attention Explorer")
    gallery = load_image_paths(cfg.attention_dir)
    if not gallery:
        st.info("Drop attention runs under `outputs/attention/<label>/aggregated`.")
        return
    label = st.selectbox("Select run", sorted(gallery.keys()))
    images = gallery.get(label, [])
    cols = st.columns(2)
    for idx, image_path in enumerate(images):
        column = cols[idx % len(cols)]
        with column:
            st.image(str(image_path), caption=image_path.name)
            with open(image_path, "rb") as handle:
                st.download_button(
                    label=f"Download {image_path.name}",
                    data=handle.read(),
                    file_name=image_path.name,
                )


@st.cache_data(show_spinner=False)
def load_embedding_metrics(path: Path) -> Dict[str, object]:
    metrics_path = path / "embedding_metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def page_embeddings(cfg: DashboardConfig) -> None:
    st.title("Embedding Explorer")
    metrics = load_embedding_metrics(cfg.embeddings_dir)
    if not metrics:
        st.info("Run `scripts/visualize_embeddings.py` and ensure outputs land in docs/figures/embeddings.")
        return
    source = st.selectbox("Source", sorted(metrics.keys()))
    reducer = st.selectbox("Reducer", sorted(metrics[source].keys()))
    field = st.selectbox("Metric Field", sorted(metrics[source][reducer].keys()))
    values = metrics[source][reducer][field]
    st.metric("Silhouette", f"{values.get('silhouette', float('nan')):.3f}")
    st.metric("Nearest Neighbor Accuracy", f"{values.get('nearest_neighbor_accuracy', float('nan')):.3f}")

    comparison_dir = cfg.embeddings_dir / f"{reducer}_2d" / "comparison"
    if comparison_dir.exists():
        pngs = sorted(comparison_dir.glob("*.png"))
        if pngs:
            st.subheader("Comparison Overlays")
            for png in pngs:
                st.image(str(png), caption=png.name)


@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint: Path) -> Optional[AbPropModel]:
    if not checkpoint.exists():
        return None
    state = torch_load_safe(checkpoint)
    if state is None:
        return None
    cfg_dict = state.get("model_config", {})
    config = TransformerConfig(**cfg_dict) if cfg_dict else TransformerConfig()
    model = AbPropModel(config)
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def torch_load_safe(path: Path) -> Optional[Dict[str, object]]:
    try:
        import torch
    except ImportError:
        st.warning("PyTorch not available; prediction sandbox disabled.")
        return None
    return torch.load(path, map_location="cpu")


def page_prediction(cfg: DashboardConfig) -> None:
    st.title("Prediction Sandbox")
    checkpoint_files = sorted(cfg.checkpoints_dir.glob("*.pt"))
    if not checkpoint_files:
        st.info("Place checkpoints under outputs/checkpoints to enable inference.")
        return
    checkpoint = st.selectbox("Checkpoint", checkpoint_files)
    sequence = st.text_area("Input sequence", height=160)
    run_button = st.button("Predict")
    if run_button and sequence:
        model = load_model_cached(checkpoint)
        if model is None:
            st.error("Failed to load model. Verify PyTorch installation and checkpoint path.")
            return
        try:
            import torch
        except ImportError:
            st.error("PyTorch not available; cannot run inference.")
            return
        batch = collate_batch([sequence], add_special=True)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, tasks=("mlm", "cls", "reg"))
        st.json({
            "liability_regression": outputs.get("regression", torch.tensor([])).tolist(),
            "token_logits_shape": list(outputs.get("cls_logits", torch.tensor([])).shape),
        })


@st.cache_data(show_spinner=False)
def load_eval_reports(path: Path) -> Dict[str, Dict[str, object]]:
    reports: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return reports
    for report_path in path.glob("*.json"):
        with open(report_path, "r", encoding="utf-8") as handle:
            reports[report_path.stem] = json.load(handle)
    return reports


def page_errors(cfg: DashboardConfig) -> None:
    st.title("Error Browser")
    reports = load_eval_reports(cfg.eval_dir)
    if not reports:
        st.info("No evaluation reports located. Generate them with `abprop-eval`.")
        return
    run = st.selectbox("Evaluation run", sorted(reports.keys()))
    payload = reports[run]
    error_table = payload.get("errors") or []
    if error_table:
        st.dataframe(pd.DataFrame(error_table))
    else:
        st.write("No explicit error entries; showing raw payload.")
        st.json(payload)


def page_checkpoints(cfg: DashboardConfig) -> None:
    st.title("Checkpoint Comparison")
    checkpoints = sorted(cfg.checkpoints_dir.glob("*.pt"))
    if len(checkpoints) < 2:
        st.info("Need at least two checkpoints under outputs/checkpoints for comparison.")
        return
    col1, col2 = st.columns(2)
    choice_a = col1.selectbox("Checkpoint A", checkpoints, index=0, key="ckpt_a")
    choice_b = col2.selectbox("Checkpoint B", checkpoints, index=1, key="ckpt_b")
    if st.button("Compare"):
        payload_a = torch_load_safe(choice_a) or {}
        payload_b = torch_load_safe(choice_b) or {}
        step_a = payload_a.get("step", "unknown")
        step_b = payload_b.get("step", "unknown")
        metrics_a = payload_a.get("metrics", {})
        metrics_b = payload_b.get("metrics", {})
        diff_rows: List[Dict[str, object]] = []
        keys = set(metrics_a) | set(metrics_b)
        for key in sorted(keys):
            diff_rows.append(
                {
                    "metric": key,
                    "checkpoint_a": metrics_a.get(key),
                    "checkpoint_b": metrics_b.get(key),
                    "delta": (metrics_b.get(key) or 0) - (metrics_a.get(key) or 0),
                }
            )
        st.write(f"Step A: {step_a} · Step B: {step_b}")
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows))
        else:
            st.info("No metric payloads present in checkpoints.")


def main() -> None:
    cfg = load_config()
    st.sidebar.title("Navigation")
    pages = {
        "Overview": page_overview,
        "Benchmarks": page_benchmarks,
        "Attention Explorer": page_attention,
        "Embedding Explorer": page_embeddings,
        "Prediction Sandbox": page_prediction,
        "Error Browser": page_errors,
        "Checkpoint Comparison": page_checkpoints,
    }
    selection = st.sidebar.radio("Select page", list(pages.keys()))
    pages[selection](cfg)


if __name__ == "__main__":
    main()
