import math
from typing import Dict, List, Sequence, Tuple

import pytest
import torch
from torch import nn

from abprop.baselines import ESM2Baseline, ESM2Config
from abprop.tokenizers import TOKEN_TO_ID, collate_batch


class _FakeAlphabet:
    def __init__(self) -> None:
        self.padding_idx = 0
        self.cls_idx = 1
        self.eos_idx = 2
        self.mask_idx = 3
        self.all_toks: List[str] = ["<pad>", "<cls>", "<eos>", "<mask>"]
        for token in list("ACDEFGHIKLMNPQRSTVWY") + ["X"]:
            if token not in self.all_toks:
                self.all_toks.append(token)
        self._token_to_idx: Dict[str, int] = {token: idx for idx, token in enumerate(self.all_toks)}

    def get_idx(self, token: str) -> int:
        return self._token_to_idx[token]

    def get_batch_converter(self):
        def converter(batch: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[str], torch.Tensor]:
            if not batch:
                raise ValueError("batch must not be empty")
            labels = [label for label, _ in batch]
            strings = [sequence for _, sequence in batch]
            max_len = max(len(sequence) for sequence in strings) + 2  # cls + eos
            tokens = torch.full((len(batch), max_len), self.padding_idx, dtype=torch.long)
            for idx, sequence in enumerate(strings):
                tokens[idx, 0] = self.cls_idx
                for offset, residue in enumerate(sequence, start=1):
                    tokens[idx, offset] = self.get_idx(residue)
                tokens[idx, len(sequence) + 1] = self.eos_idx
            return labels, strings, tokens

        return converter


class _FakeBackbone(nn.Module):
    def __init__(self, embed_dim: int = 32, vocab: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._vocab = vocab

    def forward(self, tokens: torch.Tensor, **_: object) -> Dict[str, object]:
        batch, seq_len = tokens.shape
        device = tokens.device
        representations = torch.randn(batch, seq_len, self.embed_dim, device=device)
        logits = torch.randn(batch, seq_len, self._vocab, device=device)
        return {
            "representations": {1: representations},
            "logits": logits,
        }

    def half(self) -> "_FakeBackbone":
        return self


def _build_inputs(sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = collate_batch(sequences, add_special=True)
    return batch["input_ids"], batch["attention_mask"]


def test_esm2_baseline_forward_smoke():
    config = ESM2Config(
        repr_layer=1,
        freeze_backbone=True,
        tasks=("mlm", "cls", "reg"),
        liability_keys=("nglyc", "deamidation"),
        tied_token_head=False,
    )
    alphabet = _FakeAlphabet()
    backbone = _FakeBackbone(embed_dim=16, vocab=len(alphabet.all_toks))
    model = ESM2Baseline(config, backbone=backbone, alphabet=alphabet)

    input_ids, attention_mask = _build_inputs(["ACDE", "GGX"])
    mlm_labels = input_ids.clone()
    mlm_labels[attention_mask == 0] = -100
    token_labels = torch.zeros_like(input_ids)
    token_labels[:, 1:] = 1
    liability_targets = [
        {"nglyc": 0.5, "deamidation": 0.2},
        {"nglyc": 0.0, "deamidation": -0.1},
    ]

    outputs = model(
        input_ids,
        attention_mask,
        mlm_labels=mlm_labels,
        token_labels=token_labels,
        liability_targets=liability_targets,
        tasks=("mlm", "cls", "reg"),
    )

    assert outputs["mlm_logits"].shape[:2] == input_ids.shape
    assert outputs["cls_logits"].shape[:2] == input_ids.shape
    assert outputs["regression"].shape[0] == input_ids.size(0)
    assert outputs["loss"] is not None
    assert "mlm_loss" in outputs["losses"]
    assert "cls_loss" in outputs["losses"]
    assert "reg_loss" in outputs["losses"]


def test_probe_state_roundtrip():
    config = ESM2Config(repr_layer=1, liability_keys=("a",), tied_token_head=True)
    alphabet = _FakeAlphabet()
    backbone = _FakeBackbone(embed_dim=8, vocab=len(alphabet.all_toks))
    model = ESM2Baseline(config, backbone=backbone, alphabet=alphabet)

    state = model.get_probe_state_dict()
    # Modify weights to ensure load restores them
    with torch.no_grad():
        for param in model.token_classifier.parameters():
            param.add_(1.0)

    model.load_probe_state_dict(state)
    restored = model.get_probe_state_dict()

    for head in ("token_classifier", "reg_head"):
        for sub_key, tensor in state[head].items():
            assert torch.allclose(tensor, restored[head][sub_key])
