import pytest
import torch
from torch.utils.data import DataLoader

from abprop.data import build_collate_fn
from abprop.models import AbPropModel, TransformerConfig
from abprop.train import LoopConfig, TrainLoop


class _MiniDataset:
    def __init__(self, sequences):
        self.samples = [{"sequence": seq, "chain": "H", "liability_ln": {}} for seq in sequences]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def test_single_train_step(tmp_path):
    config = TransformerConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    model = AbPropModel(config)

    sequences = ["ACDEFGHIK", "ACDGST", "ACDX", "ACDEFG"]
    dataset = _MiniDataset(sequences)
    collate = build_collate_fn(generate_mlm=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    loop_config = LoopConfig(
        learning_rate=5e-4,
        weight_decay=0.0,
        gradient_clipping=1.0,
        grad_accumulation=1,
        max_steps=1,
        warmup_steps=0,
        lr_schedule="linear",
        eval_interval=0,
        checkpoint_interval=0,
        output_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpts",
        tasks=("mlm",),
        precision="fp32",
    )

    loop = TrainLoop(
        model,
        loop_config,
        log_run_name="unit-train",
        device=torch.device("cpu"),
        is_rank_zero_run=True,
    )

    loop.fit(dataloader)
    assert len(loop.train_history) == 1
    assert loop.train_history[0] >= 0.0
