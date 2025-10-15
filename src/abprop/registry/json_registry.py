"""JSON-backed model registry implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ModelRecord:
    model_id: str
    checkpoint: str
    config: Dict[str, object]
    metrics: Dict[str, float]
    tags: List[str]
    created_at: str
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ModelRegistry:
    """Tiny JSON registry for storing AbProp checkpoints and metadata."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> List[Dict[str, object]]:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, records: Iterable[Dict[str, object]]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(list(records), handle, indent=2)

    def list(self) -> List[ModelRecord]:
        return [ModelRecord(**record) for record in self._read()]

    def get(self, model_id: str) -> Optional[ModelRecord]:
        for record in self.list():
            if record.model_id == model_id:
                return record
        return None

    def register(
        self,
        *,
        model_id: str,
        checkpoint: Path | str,
        config: Dict[str, object],
        metrics: Dict[str, float],
        tags: Optional[Iterable[str]] = None,
        notes: str = "",
    ) -> ModelRecord:
        if self.get(model_id):
            raise ValueError(f"Model id '{model_id}' already exists in registry.")
        record = ModelRecord(
            model_id=model_id,
            checkpoint=str(Path(checkpoint).resolve()),
            config=config,
            metrics=metrics,
            tags=sorted(set(tags or [])),
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            notes=notes,
        )
        records = self._read()
        records.append(record.to_dict())
        self._write(records)
        return record

    def update(self, model_id: str, **updates: object) -> ModelRecord:
        records = self._read()
        for idx, record in enumerate(records):
            if record["model_id"] == model_id:
                record.update(updates)
                records[idx] = record
                self._write(records)
                return ModelRecord(**record)
        raise KeyError(model_id)

    def delete(self, model_id: str) -> None:
        records = [record for record in self._read() if record["model_id"] != model_id]
        self._write(records)

    def best(self, metric: str, higher_is_better: bool = True) -> Optional[ModelRecord]:
        candidates = [record for record in self.list() if metric in record.metrics]
        if not candidates:
            return None
        key_fn = (lambda rec: rec.metrics[metric])
        return max(candidates, key=key_fn) if higher_is_better else min(candidates, key=key_fn)

    def export_card(self, model_id: str, destination: Path | str) -> Path:
        record = self.get(model_id)
        if record is None:
            raise KeyError(model_id)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Model Card: {record.model_id}",
            "",
            f"- **Checkpoint**: `{record.checkpoint}`",
            f"- **Created At**: {record.created_at}",
            f"- **Tags**: {', '.join(record.tags) if record.tags else 'none'}",
            "",
            "## Metrics",
        ]
        for key, value in record.metrics.items():
            lines.append(f"- {key}: {value}")
        lines.extend([
            "",
            "## Configuration",
            "```json",
            json.dumps(record.config, indent=2),
            "```",
        ])
        if record.notes:
            lines.extend(["", "## Notes", record.notes])
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination
