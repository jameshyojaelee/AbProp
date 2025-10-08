"""Baseline Transformer model with auxiliary heads for AbProp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from abprop.tokenizers import AMINO_ACIDS, SPECIAL_TOKENS


@dataclass
class TransformerConfig:
    vocab_size: int = len(SPECIAL_TOKENS) + len(AMINO_ACIDS)
    d_model: int = 384
    nhead: int = 6
    num_layers: int = 3
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_position_embeddings: int = 1024
    liability_keys: Tuple[str, ...] = (
        "nglyc",
        "deamidation",
        "isomerization",
        "oxidation",
        "free_cysteines",
    )
    mlm_weight: float = 1.0
    cls_weight: float = 1.0
    reg_weight: float = 1.0


class SmallEncoder(nn.Module):
    """Lightweight Transformer encoder with learned positional embeddings."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = config.d_model**-0.5
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask.bool()
        positions = torch.arange(
            input_ids.size(1), device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        hidden = self.token_embedding(input_ids) * self.scale + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        key_padding_mask = ~attention_mask
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        return encoded


class MLMHead(nn.Module):
    """Masked language modeling head with tied embeddings."""

    def __init__(self, hidden_size: int, vocab_size: int, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight = embedding_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states) + self.bias
        return logits


class SeqClassifierHead(nn.Module):
    """Token-level classifier (e.g., framework vs CDR)."""

    def __init__(self, hidden_size: int, dropout: float, num_labels: int = 2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden_states)
        return self.classifier(x)


class LiabilityRegHead(nn.Module):
    """Sequence-level liability regression head."""

    def __init__(self, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return self.regressor(pooled)


class AbPropModel(nn.Module):
    """Wrapper model combining encoder with MLM, classification, and regression heads."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        self.encoder = SmallEncoder(self.config)
        self.mlm_head = MLMHead(
            hidden_size=self.config.d_model,
            vocab_size=self.config.vocab_size,
            embedding_weight=self.encoder.token_embedding.weight,
        )
        self.classifier = SeqClassifierHead(
            hidden_size=self.config.d_model,
            dropout=self.config.dropout,
        )
        self.regressor = LiabilityRegHead(
            hidden_size=self.config.d_model,
            output_size=len(self.config.liability_keys),
            dropout=self.config.dropout,
        )

        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.reg_loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        tasks = tuple(tasks or ("mlm", "cls", "reg"))
        attention_mask = attention_mask.bool()
        hidden_states = self.encoder(input_ids, attention_mask)

        outputs: Dict[str, object] = {"hidden_states": hidden_states}
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)
        loss_tracked = False

        if "mlm" in tasks:
            mlm_logits = self.mlm_head(hidden_states)
            outputs["mlm_logits"] = mlm_logits
            if mlm_labels is not None:
                mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
                losses["mlm_loss"] = mlm_loss
                metrics["mlm_perplexity"] = torch.exp(mlm_loss.detach())
                total_loss = total_loss + self.config.mlm_weight * mlm_loss
                loss_tracked = True

        if "cls" in tasks:
            token_logits = self.classifier(hidden_states)
            outputs["cls_logits"] = token_logits
            prepared_labels = self._prepare_token_labels(token_labels, attention_mask, device=input_ids.device)
            if prepared_labels is not None:
                cls_loss = self.cls_loss_fn(
                    token_logits.view(-1, token_logits.size(-1)),
                    prepared_labels.view(-1),
                )
                losses["cls_loss"] = cls_loss
                total_loss = total_loss + self.config.cls_weight * cls_loss
                loss_tracked = True
                predictions = token_logits.argmax(dim=-1)
                valid_mask = prepared_labels != -100
                if valid_mask.any():
                    accuracy = (
                        (predictions[valid_mask] == prepared_labels[valid_mask]).float().mean()
                    )
                    metrics["cls_accuracy"] = accuracy.detach()

        if "reg" in tasks:
            reg_targets = self._prepare_regression_targets(
                liability_targets,
                batch_size=input_ids.size(0),
                device=input_ids.device,
            )
            regression_logits = self.regressor(hidden_states, attention_mask)
            outputs["regression"] = regression_logits
            if reg_targets is not None:
                reg_loss = self.reg_loss_fn(regression_logits, reg_targets)
                losses["reg_loss"] = reg_loss
                total_loss = total_loss + self.config.reg_weight * reg_loss
                loss_tracked = True

        if loss_tracked:
            outputs["loss"] = total_loss
        else:
            outputs["loss"] = None

        outputs["losses"] = losses
        outputs["metrics"] = metrics
        return outputs

    def _prepare_token_labels(
        self,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]],
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if token_labels is None:
            return None
        if isinstance(token_labels, torch.Tensor):
            return token_labels.to(device)

        batch_size, seq_len = attention_mask.shape
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        for idx, sequence_labels in enumerate(token_labels):
            if sequence_labels is None:
                continue
            if isinstance(sequence_labels, torch.Tensor):
                seq_tensor = sequence_labels.to(device=device, dtype=torch.long)
            else:
                seq_tensor = torch.tensor(sequence_labels, dtype=torch.long, device=device)
            if seq_tensor.numel() == 0:
                continue
            max_copy = min(seq_tensor.size(0), seq_len - 2)
            labels[idx, 1 : 1 + max_copy] = seq_tensor[:max_copy]
        return labels

    def _prepare_regression_targets(
        self,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if liability_targets is None:
            return None
        if isinstance(liability_targets, torch.Tensor):
            return liability_targets.to(device=device, dtype=torch.float32)

        target_tensor = torch.zeros(
            (batch_size, len(self.config.liability_keys)),
            dtype=torch.float32,
            device=device,
        )
        for idx, entry in enumerate(liability_targets):
            if entry is None:
                continue
            for key_idx, key in enumerate(self.config.liability_keys):
                target_tensor[idx, key_idx] = float(entry.get(key, 0.0))
        return target_tensor


__all__ = [
    "TransformerConfig",
    "SmallEncoder",
    "MLMHead",
    "SeqClassifierHead",
    "LiabilityRegHead",
    "AbPropModel",
]
