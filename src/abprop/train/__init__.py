"""Training utilities for AbProp models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim

from abprop.utils import DEFAULT_OUTPUT_DIR


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_steps: int = 1000
    log_every: int = 50
    output_dir: Path = DEFAULT_OUTPUT_DIR


class Trainer:
    """Minimal training loop for quick experiments."""

    def __init__(self, model: nn.Module, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        padding_mask = batch.get("padding_mask")
        if padding_mask is None and "attention_mask" in batch:
            attention_mask = batch["attention_mask"].bool()
            padding_mask = ~attention_mask
        if padding_mask is not None:
            padding_mask = padding_mask.bool()

        logits, _ = self.model(input_ids, padding_mask)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return {"loss": float(loss.detach().cpu())}


__all__ = ["Trainer", "TrainingConfig"]
