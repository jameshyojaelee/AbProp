"""Baseline Transformer model with auxiliary heads for AbProp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from abprop.tokenizers import AMINO_ACIDS, SPECIAL_TOKENS
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS
from abprop.eval.metrics import classification_summary, regression_summary


@dataclass
class TransformerConfig:
    vocab_size: int = len(SPECIAL_TOKENS) + len(AMINO_ACIDS)
    d_model: int = 384
    nhead: int = 6
    num_layers: int = 3
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_position_embeddings: int = 1024
    liability_keys: Tuple[str, ...] = CANONICAL_LIABILITY_KEYS
    mlm_weight: float = 1.0
    cls_weight: float = 1.0
    reg_weight: float = 1.0


def _get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation '{name}'.")


class TransformerEncoderLayerWithAttention(nn.Module):
    """Minimal Transformer encoder layer that can optionally expose self-attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attentions,
            average_attn_weights=False,
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src, attn_weights


class SmallEncoder(nn.Module):
    """Lightweight Transformer encoder with learned positional embeddings."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = config.d_model**-0.5
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList(
            TransformerEncoderLayerWithAttention(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]] | torch.Tensor:
        attention_mask = attention_mask.bool()
        positions = torch.arange(
            input_ids.size(1), device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        hidden = self.token_embedding(input_ids) * self.scale + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        key_padding_mask = ~attention_mask
        attentions: Optional[List[torch.Tensor]] = [] if return_attentions else None
        for layer in self.layers:
            hidden, attn_weights = layer(
                hidden,
                src_key_padding_mask=key_padding_mask,
                return_attentions=return_attentions,
            )
            if return_attentions and attn_weights is not None:
                attentions.append(attn_weights)
        encoded = self.norm(hidden)
        if return_attentions:
            return encoded, attentions
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
        self._mc_dropout_enabled = False
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

        # Cache dropout modules for quick toggling during MC dropout inference.
        self._dropout_modules: Tuple[nn.Dropout, ...] = tuple(
            module for module in self.modules() if isinstance(module, nn.Dropout)
        )

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
        return_attentions: bool = False,
    ) -> Dict[str, object]:
        tasks = tuple(tasks or ("mlm", "cls", "reg"))
        attention_mask = attention_mask.bool()
        encoder_output = self.encoder(
            input_ids,
            attention_mask,
            return_attentions=return_attentions,
        )
        if return_attentions:
            hidden_states, attentions = encoder_output  # type: ignore[misc]
        else:
            hidden_states = encoder_output  # type: ignore[assignment]
            attentions = None

        outputs: Dict[str, object] = {"hidden_states": hidden_states}
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)
        loss_tracked = False

        if attentions is not None:
            outputs["attentions"] = attentions

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
                valid_mask = prepared_labels != -100
                if valid_mask.any():
                    valid_logits = token_logits[valid_mask]
                    valid_labels = prepared_labels[valid_mask]
                    predictions = valid_logits.argmax(dim=-1)
                    accuracy = (predictions == valid_labels).float().mean()
                    metrics["cls_accuracy"] = accuracy.detach()
                    summary = classification_summary(
                        int(((predictions == 1) & (valid_labels == 1)).sum()),
                        int(((predictions == 1) & (valid_labels == 0)).sum()),
                        int(((predictions == 0) & (valid_labels == 0)).sum()),
                        int(((predictions == 0) & (valid_labels == 1)).sum()),
                    )
                    metrics["cls_f1"] = torch.tensor(summary["f1"], device=input_ids.device)

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
                preds_det = regression_logits.detach()
                targets_det = reg_targets.detach()
                reg_summary = regression_summary(preds_det, targets_det)
                metrics["reg_mse"] = torch.tensor(reg_summary["mse"], device=input_ids.device)
                metrics["reg_r2"] = torch.tensor(reg_summary["r2"], device=input_ids.device)
                metrics["reg_spearman"] = torch.tensor(reg_summary["spearman"], device=input_ids.device)

        if loss_tracked:
            outputs["loss"] = total_loss
        else:
            outputs["loss"] = None

        outputs["losses"] = losses
        outputs["metrics"] = metrics
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
        return_attentions: bool = False,
    ) -> Dict[str, object]:
        return self._forward_impl(
            input_ids,
            attention_mask,
            mlm_labels=mlm_labels,
            token_labels=token_labels,
            liability_targets=liability_targets,
            tasks=tasks,
            return_attentions=return_attentions,
        )

    def stochastic_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mc_samples: int,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
        enable_dropout: bool = True,
        no_grad: bool = True,
        return_attentions: bool = False,
    ) -> List[Dict[str, object]]:
        if mc_samples <= 0:
            raise ValueError("mc_samples must be a positive integer.")

        original_training = self.training
        dropout_states = self._snapshot_dropout_states()
        previous_flag = self._mc_dropout_enabled

        self.eval()
        if enable_dropout:
            self.set_inference_dropout(True)

        outputs: List[Dict[str, object]] = []
        grad_ctx = torch.no_grad if no_grad else torch.enable_grad  # type: ignore[assignment]
        with grad_ctx():
            for _ in range(mc_samples):
                outputs.append(
                    self._forward_impl(
                        input_ids,
                        attention_mask,
                        mlm_labels=mlm_labels,
                        token_labels=token_labels,
                        liability_targets=liability_targets,
                        tasks=tasks,
                        return_attentions=return_attentions,
                    )
                )

        if enable_dropout:
            self._restore_dropout_states(dropout_states)
            self._mc_dropout_enabled = previous_flag

        if original_training:
            self.train()

        return outputs

    def set_inference_dropout(self, enabled: bool) -> None:
        """Toggle dropout layers while keeping the rest of the model in eval mode."""
        self._mc_dropout_enabled = enabled
        for module in self._dropout_modules:
            if self.training and not enabled:
                module.train(True)
            else:
                module.train(enabled)

    def inference_dropout_enabled(self) -> bool:
        return self._mc_dropout_enabled

    def _snapshot_dropout_states(self) -> Tuple[bool, ...]:
        return tuple(module.training for module in self._dropout_modules)

    def _restore_dropout_states(self, states: Sequence[bool]) -> None:
        for module, state in zip(self._dropout_modules, states):
            module.train(state)

    def _prepare_token_labels(
        self,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]],
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if token_labels is None:
            return None
        if isinstance(token_labels, torch.Tensor):
            return token_labels.to(device=device, dtype=torch.long)

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
