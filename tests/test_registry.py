import json
from pathlib import Path

import pytest

from abprop.registry import ModelRegistry


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "registry.json"


def test_register_and_list(registry_path: Path):
    registry = ModelRegistry(registry_path)
    registry.register(
        model_id="model-v1",
        checkpoint="/tmp/checkpoint.pt",
        config={"d_model": 384},
        metrics={"perplexity": 1.9},
        tags=["baseline"],
    )
    records = registry.list()
    assert len(records) == 1
    assert records[0].model_id == "model-v1"


def test_best(registry_path: Path):
    registry = ModelRegistry(registry_path)
    registry.register("model-a", "/tmp/a.pt", {"d": 1}, {"f1": 0.8})
    registry.register("model-b", "/tmp/b.pt", {"d": 1}, {"f1": 0.9})
    best = registry.best("f1", higher_is_better=True)
    assert best and best.model_id == "model-b"


def test_export_card(registry_path: Path, tmp_path: Path):
    registry = ModelRegistry(registry_path)
    registry.register(
        model_id="model-card",
        checkpoint="/tmp/c.pt",
        config={"param": 1},
        metrics={"rmse": 0.2},
    )
    dest = tmp_path / "card.md"
    registry.export_card("model-card", dest)
    assert dest.exists()
    content = dest.read_text()
    assert "Model Card: model-card" in content
