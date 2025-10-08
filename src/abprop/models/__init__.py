"""Model architectures for AbProp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from abprop.tokenizers import SPECIAL_TOKENS, AMINO_ACIDS


@dataclass
class TransformerConfig:
    vocab_size: int = len(SPECIAL_TOKENS) + len(AMINO_ACIDS)
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    max_position_embeddings: int = 1024


class SimpleTransformerEncoder(nn.Module):
    """Minimal Transformer encoder for sequence modeling."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits and hidden states.

        Args:
            input_ids: Tensor of token IDs with shape (batch, seq_len).
            padding_mask: Optional mask with True for padded positions.
        """
        hidden = self.embedding(input_ids)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        logits = self.lm_head(encoded)
        return logits, encoded


__all__ = ["TransformerConfig", "SimpleTransformerEncoder"]

