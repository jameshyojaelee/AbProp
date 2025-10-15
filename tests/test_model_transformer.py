import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for model tests.")
nn = pytest.importorskip("torch.nn", reason="PyTorch is required for model tests.")

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


def test_attention_outputs_shape():
    config = TransformerConfig(d_model=32, nhead=2, num_layers=2, dim_feedforward=64)
    model = AbPropModel(config)

    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    outputs = model(
        input_ids,
        attention_mask,
        tasks=(),
        return_attentions=True,
    )

    assert "attentions" in outputs
    attentions = outputs["attentions"]
    assert isinstance(attentions, list)
    assert len(attentions) == config.num_layers
    for attn in attentions:
        assert attn.shape == (batch_size, config.nhead, seq_len, seq_len)
        assert torch.isfinite(attn).all()


def test_stochastic_forward_mc_samples_and_dropout_reset():
    config = TransformerConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    model = AbPropModel(config)
    model.eval()

    batch_size, seq_len = 2, 12
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    outputs = model.stochastic_forward(
        input_ids,
        attention_mask,
        tasks=("mlm",),
        mc_samples=3,
    )

    assert len(outputs) == 3
    # Dropout layers should remain disabled after stochastic inference.
    dropout_training_states = [
        module.training for module in model.modules() if isinstance(module, nn.Dropout)
    ]
    assert dropout_training_states, "Model should expose dropout modules for inference."
    assert all(state is False for state in dropout_training_states)
    assert model.inference_dropout_enabled() is False
