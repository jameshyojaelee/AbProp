import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for model tests.")

from abprop.models import AbPropModel, TransformerConfig


@pytest.mark.parametrize("tasks", [("mlm", "cls", "reg"), ("mlm",)])
def test_abprop_model_forward(tasks):
    config = TransformerConfig(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    )
    model = AbPropModel(config)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    mlm_labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    mlm_labels[0, 0] = -100  # ignore example

    token_labels = [
        [0] * (seq_len - 2),
        [1] * (seq_len - 2),
    ]
    liability_targets = [
        {key: 0.1 * (idx + 1) for idx, key in enumerate(config.liability_keys)},
        {key: 0.0 for key in config.liability_keys},
    ]

    outputs = model(
        input_ids,
        attention_mask,
        mlm_labels=mlm_labels,
        token_labels=token_labels,
        liability_targets=liability_targets,
        tasks=tasks,
    )

    assert "loss" in outputs
    if "mlm" in tasks:
        assert "mlm_logits" in outputs
        assert "mlm_loss" in outputs["losses"]
        assert "mlm_perplexity" in outputs["metrics"]
    if "cls" in tasks:
        assert "cls_logits" in outputs
        assert "cls_loss" in outputs["losses"]
    if "reg" in tasks:
        assert "regression" in outputs
        assert "reg_loss" in outputs["losses"]
