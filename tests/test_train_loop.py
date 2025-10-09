from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from abprop.data import build_collate_fn
from abprop.models import AbPropModel, TransformerConfig
from abprop.train.loop import LoopConfig, TrainLoop


class _ToyDataset(Dataset):
    def __init__(self, sequences, liability_keys):
        self.examples = []
        template = {key: 0.0 for key in liability_keys}
        for seq in sequences:
            length = len(seq)
            mask = [1 if idx >= length // 2 else 0 for idx in range(length)]
            self.examples.append(
                {
                    "sequence": seq,
                    "chain": "H",
                    "liability_ln": template.copy(),
                    "cdr_mask": mask,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA amp not required for CPU test")
def test_amp_flag_respected():
    config = LoopConfig(max_steps=1, precision="amp")
    model = AbPropModel()
    loop = TrainLoop(model, config)
    assert loop.use_amp == (loop.device.type == "cuda")


def test_train_loop_reduces_loss(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    transformer_cfg = TransformerConfig(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    )
    model = AbPropModel(transformer_cfg)

    motif = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [(motif * ((i % 3) + 1))[:40] for i in range(32)]
    dataset = _ToyDataset(sequences, transformer_cfg.liability_keys)
    collate = build_collate_fn(generate_mlm=True, mlm_probability=0.2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

    loop_config = LoopConfig(
        learning_rate=5e-4,
        weight_decay=0.0,
        gradient_clipping=1.0,
        grad_accumulation=1,
        max_steps=100,
        warmup_steps=10,
        lr_schedule="linear",
        eval_interval=0,
        checkpoint_interval=0,
        output_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpts",
        tasks=("mlm",),
        precision="fp32",
    )

    loop = TrainLoop(model, loop_config, log_run_name="test-loop")
    loop.save_run_metadata({"train": {}, "model": {}, "data": {}})
    loop.fit(dataloader)

    assert len(loop.train_history) == loop_config.max_steps
    assert loop.train_history[0] > loop.train_history[-1]
    assert (tmp_path / "outputs" / "config_snapshot.json").exists()
    assert (tmp_path / "logs" / "test-loop.csv").exists()
