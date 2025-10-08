import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for batching tests.")

from abprop.models import SimpleTransformerEncoder
from abprop.train import Trainer
from abprop.tokenizers import TOKEN_TO_ID, collate_batch


def test_trainer_handles_attention_mask():
    sequences = ["ACD", "A", "ACDEFG"]
    batch = collate_batch(sequences)
    # Trainer expects labels; reuse input ids for language modeling style.
    batch["labels"] = batch["input_ids"].clone()

    model = SimpleTransformerEncoder()
    trainer = Trainer(model)
    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0

    pad_id = TOKEN_TO_ID["<pad>"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    assert torch.equal(attention_mask, input_ids != pad_id)
