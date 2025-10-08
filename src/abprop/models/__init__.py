"""Model architectures for AbProp."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .transformer import (
    AbPropModel,
    LiabilityRegHead,
    MLMHead,
    SeqClassifierHead,
    SmallEncoder,
    TransformerConfig,
)


class SimpleTransformerEncoder(nn.Module):
    """Minimal Transformer encoder for sequence modeling."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embedding(input_ids)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        logits = self.lm_head(encoded)
        return logits, encoded


__all__ = [
    "TransformerConfig",
    "SmallEncoder",
    "MLMHead",
    "SeqClassifierHead",
    "LiabilityRegHead",
    "AbPropModel",
    "SimpleTransformerEncoder",
]
