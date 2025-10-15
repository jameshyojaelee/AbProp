"""ESM-2 baseline model adapter for AbProp benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from abprop.eval.metrics import classification_summary, regression_summary
from abprop.tokenizers import ID_TO_TOKEN, TOKEN_TO_ID, VOCAB, decode
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS

try:  # pragma: no cover - dependency not available in minimal test env
    import esm  # type: ignore
except ImportError:  # pragma: no cover
    esm = None


@dataclass
class ESM2Config:
    """Configuration for the ESM-2 baseline adapter."""

    model_name: str = "esm2_t33_650M_UR50D"
    repr_layer: int = 33
    probe_dropout: float = 0.1
    liability_keys: Tuple[str, ...] = CANONICAL_LIABILITY_KEYS
    freeze_backbone: bool = True
    half_precision: bool = False
    tied_token_head: bool = True
    pad_id: int = TOKEN_TO_ID["<pad>"]
    bos_id: int = TOKEN_TO_ID["<bos>"]
    eos_id: int = TOKEN_TO_ID["<eos>"]
    mask_id: int = TOKEN_TO_ID["<mask>"]
    tasks: Tuple[str, ...] = ("mlm", "cls", "reg")
    sequence_pooling: str = "mean"  # mean | cls
    allow_missing_tokens: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


class ESM2Baseline(nn.Module):
    """Wrap a fair-esm backbone with linear probes that mimic AbProp outputs."""

    def __init__(
        self,
        config: ESM2Config | None = None,
        *,
        backbone: Optional[nn.Module] = None,
        alphabet: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.config = config or ESM2Config()

        if backbone is None or alphabet is None:
            if esm is None:
                raise ImportError(
                    "fair-esm is required for ESM2Baseline. Install with `pip install fair-esm`."
                )
            loader = getattr(esm.pretrained, self.config.model_name, None)
            if loader is None:
                raise ValueError(
                    f"Unknown ESM-2 model '{self.config.model_name}'. "
                    "Refer to fair-esm documentation for supported names."
                )
            backbone, alphabet = loader()

        self.backbone = backbone  # type: ignore[assignment]
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None and hasattr(self.backbone, "encoder"):
            embed_dim = getattr(self.backbone.encoder, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("Unable to infer embedding dimension from ESM backbone.")
        self.embed_dim = int(embed_dim)

        self.probe_dropout = nn.Dropout(self.config.probe_dropout)
        self.token_classifier = nn.Linear(self.embed_dim, 2)
        self.reg_head = nn.Linear(self.embed_dim, len(self.config.liability_keys))

        if self.config.tied_token_head and hasattr(self.backbone, "lm_head"):
            self.mlm_head = self.backbone.lm_head  # type: ignore[attr-defined]
        else:
            self.mlm_head = nn.Linear(self.embed_dim, len(self.alphabet.all_toks))

        if self.config.freeze_backbone:
            self.freeze_backbone()

        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.reg_loss_fn = nn.MSELoss()

        self._ab_vocab_size = len(VOCAB)
        self._token_mapping = self._build_token_mapping()

    # ------------------------------------------------------------------ utils
    def freeze_backbone(self) -> None:
        """Disable gradients for the ESM backbone."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Enable gradients for the ESM backbone."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def probe_parameters(self) -> Iterable[nn.Parameter]:
        """Return iterator over trainable probe parameters."""
        for module in (self.token_classifier, self.reg_head):
            yield from module.parameters()
        if not self.config.tied_token_head:
            yield from self.mlm_head.parameters()

    def to_config_dict(self) -> Dict[str, object]:
        return asdict(self.config)

    # ----------------------------------------------------------------- helpers
    def _build_token_mapping(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        alphabet = self.alphabet
        special_map = {
            "<pad>": getattr(alphabet, "padding_idx", None),
            "<bos>": getattr(alphabet, "cls_idx", None),
            "<eos>": getattr(alphabet, "eos_idx", None),
            "<mask>": getattr(alphabet, "mask_idx", None),
        }
        for token, idx in special_map.items():
            if idx is not None:
                mapping[token] = int(idx)

        for token in VOCAB:
            if token in special_map:
                continue
            try:
                mapping[token] = int(self.alphabet.get_idx(token))  # type: ignore[attr-defined]
            except KeyError:
                if not self.config.allow_missing_tokens:
                    raise
        return mapping

    def _decode_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[str]:
        sequences: List[str] = []
        ids_cpu = input_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        for idx in range(ids_cpu.size(0)):
            valid_length = int(mask_cpu[idx].sum().item())
            token_slice = ids_cpu[idx, :valid_length].tolist()
            sequences.append(decode(token_slice, strip_special=True))
        return sequences

    @staticmethod
    def _match_sequence_length(
        tensor: torch.Tensor,
        target_length: int,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        if tensor.size(1) == target_length:
            return tensor
        if tensor.size(1) > target_length:
            return tensor[:, :target_length, ...]
        pad_len = target_length - tensor.size(1)
        pad_shape = (0, 0) * (tensor.dim() - 2) + (0, pad_len)
        return F.pad(tensor, pad_shape, value=pad_value)

    @staticmethod
    def _align_length(
        tensor: torch.Tensor,
        target_length: int,
        pad_value: Optional[int | float | bool] = None,
    ) -> torch.Tensor:
        if tensor.size(1) == target_length:
            return tensor
        if tensor.size(1) > target_length:
            return tensor[:, :target_length, ...]
        pad_len = target_length - tensor.size(1)
        pad_shape = (tensor.size(0), pad_len) + tensor.shape[2:]
        pad_tensor = tensor.new_full(pad_shape, pad_value if pad_value is not None else 0)
        return torch.cat([tensor, pad_tensor], dim=1)

    def _esm_forward(
        self,
        sequences: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = [(f"seq-{idx}", seq or "X") for idx, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(device)

        kwargs = {
            "repr_layers": [self.config.repr_layer],
            "return_contacts": False,
        }
        if self.config.half_precision:
            self.backbone.half()
            tokens = tokens.half()
        outputs = self.backbone(tokens, **kwargs)
        representations = outputs["representations"][self.config.repr_layer]
        logits = outputs["logits"]
        padding_mask = tokens.eq(getattr(self.alphabet, "padding_idx", -1))
        return tokens, representations, logits, padding_mask

    def _convert_logits_to_abprop_vocab(
        self,
        logits: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        logits = self._match_sequence_length(logits, target_len, pad_value=-1e4)
        batch_size = logits.size(0)
        ab_logits = logits.new_full((batch_size, target_len, self._ab_vocab_size), fill_value=-1e4)
        for token, ab_idx in TOKEN_TO_ID.items():
            esm_idx = self._token_mapping.get(token)
            if esm_idx is not None and esm_idx < logits.size(-1):
                ab_logits[:, :, ab_idx] = logits[:, :, esm_idx]
        return ab_logits

    def _pool_sequence(
        self,
        representations: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.sequence_pooling == "cls":
            return representations[:, 0, :]

        mask = ~padding_mask
        mask = mask.to(representations.dtype)
        masked_rep = representations[:, 1:, :]  # drop CLS token
        mask = mask[:, 1:]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (masked_rep * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    # ----------------------------------------------------------------- forward
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
        tasks = tuple(tasks or self.config.tasks)
        device = input_ids.device
        sequences = self._decode_sequences(input_ids, attention_mask)

        tokens, representations, logits, padding_mask = self._esm_forward(sequences, device)

        target_len = input_ids.size(1)
        representations = self._match_sequence_length(representations, target_len, pad_value=0.0)
        logits = self._convert_logits_to_abprop_vocab(logits, target_len)
        tokens = self._align_length(
            tokens,
            target_len,
            pad_value=getattr(self.alphabet, "padding_idx", 0),
        )
        padding_mask = self._align_length(padding_mask.unsqueeze(-1).float(), target_len, pad_value=1.0).squeeze(-1).bool()

        outputs: Dict[str, object] = {"hidden_states": representations}
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=device)
        loss_used = False

        if "mlm" in tasks:
            outputs["mlm_logits"] = logits
            if mlm_labels is not None:
                mlm_loss = self.mlm_loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
                losses["mlm_loss"] = mlm_loss
                metrics["mlm_perplexity"] = torch.exp(mlm_loss.detach())
                total_loss = total_loss + mlm_loss
                loss_used = True

        if "cls" in tasks:
            token_features = self.probe_dropout(representations)
            cls_logits = self.token_classifier(token_features)
            outputs["cls_logits"] = cls_logits
            prepared_labels = None
            if token_labels is not None:
                prepared_labels = self._prepare_token_labels(
                    token_labels,
                    attention_mask,
                    device=device,
                )
            if prepared_labels is not None:
                cls_loss = self.cls_loss_fn(
                    cls_logits.view(-1, cls_logits.size(-1)),
                    prepared_labels.view(-1),
                )
                losses["cls_loss"] = cls_loss
                total_loss = total_loss + cls_loss
                loss_used = True
                mask = prepared_labels != -100
                if mask.any():
                    preds = cls_logits.argmax(dim=-1)
                    valid_preds = preds[mask]
                    valid_labels = prepared_labels[mask]
                    cm = {
                        "tp": int(((valid_preds == 1) & (valid_labels == 1)).sum()),
                        "fp": int(((valid_preds == 1) & (valid_labels == 0)).sum()),
                        "tn": int(((valid_preds == 0) & (valid_labels == 0)).sum()),
                        "fn": int(((valid_preds == 0) & (valid_labels == 1)).sum()),
                    }
                    summary = classification_summary(cm["tp"], cm["fp"], cm["tn"], cm["fn"])
                    metrics["cls_accuracy"] = torch.tensor(summary["accuracy"], device=device)
                    metrics["cls_f1"] = torch.tensor(summary["f1"], device=device)

        if "reg" in tasks:
            pooled = self._pool_sequence(representations, tokens, padding_mask).to(device)
            pooled = self.probe_dropout(pooled)
            regression_logits = self.reg_head(pooled)
            outputs["regression"] = regression_logits

            reg_targets = self._prepare_regression_targets(
                liability_targets,
                batch_size=input_ids.size(0),
                device=device,
            )
            if reg_targets is not None:
                reg_loss = self.reg_loss_fn(regression_logits, reg_targets)
                losses["reg_loss"] = reg_loss
                total_loss = total_loss + reg_loss
                loss_used = True
                summary = regression_summary(regression_logits.detach(), reg_targets.detach())
                metrics["reg_mse"] = torch.tensor(summary["mse"], device=device)
                metrics["reg_r2"] = torch.tensor(summary["r2"], device=device)
                metrics["reg_spearman"] = torch.tensor(summary["spearman"], device=device)

        outputs["loss"] = total_loss if loss_used else None
        outputs["losses"] = losses
        outputs["metrics"] = metrics
        return outputs

    # -------------------------------------------------------------- label prep
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
            length = min(seq_tensor.numel(), seq_len - 2)
            if length <= 0:
                continue
            labels[idx, 1 : 1 + length] = seq_tensor[:length]
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
        targets = torch.zeros(
            (batch_size, len(self.config.liability_keys)),
            dtype=torch.float32,
            device=device,
        )
        for idx, entry in enumerate(liability_targets):
            if entry is None:
                continue
            for key_idx, key in enumerate(self.config.liability_keys):
                targets[idx, key_idx] = float(entry.get(key, 0.0))
        return targets

    # --------------------------------------------------------------- checkpoints
    def get_probe_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dict containing probe weights."""
        state = {
            "token_classifier": self.token_classifier.state_dict(),
            "reg_head": self.reg_head.state_dict(),
            "config": self.to_config_dict(),
        }
        if not self.config.tied_token_head:
            state["mlm_head"] = self.mlm_head.state_dict()
        return state

    def load_probe_state_dict(self, state_dict: Dict[str, object]) -> None:
        """Load probe weights (backbone is expected to stay frozen)."""
        token_state = state_dict.get("token_classifier")
        if isinstance(token_state, dict):
            self.token_classifier.load_state_dict(token_state)
        reg_state = state_dict.get("reg_head")
        if isinstance(reg_state, dict):
            self.reg_head.load_state_dict(reg_state)
        if not self.config.tied_token_head:
            mlm_state = state_dict.get("mlm_head")
            if isinstance(mlm_state, dict):
                self.mlm_head.load_state_dict(mlm_state)


__all__ = ["ESM2Baseline", "ESM2Config"]
