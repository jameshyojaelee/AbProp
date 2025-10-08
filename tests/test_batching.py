import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for batching tests.")

from abprop.models import SimpleTransformerEncoder
from abprop.train import Trainer


def test_trainer_handles_padding_mask():
    model = SimpleTransformerEncoder()
    trainer = Trainer(model)
    batch = {
        "input_ids": torch.randint(low=0, high=model.config.vocab_size, size=(2, 8)),
        "labels": torch.randint(low=0, high=model.config.vocab_size, size=(2, 8)),
        "padding_mask": torch.zeros(2, 8, dtype=torch.bool),
    }
    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0
