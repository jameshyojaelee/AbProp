"""Amino acid tokenizer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<mask>"]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]

PAD_TOKEN_ID = 0
_SPECIAL_OFFSET = len(SPECIAL_TOKENS)

VOCAB: List[str] = SPECIAL_TOKENS + AMINO_ACIDS
TOKEN_TO_ID: Dict[str, int] = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {idx: token for token, idx in TOKEN_TO_ID.items()}


def encode(sequence: str, add_special: bool = True) -> torch.Tensor:
    """Encode a sequence into token ids."""
    sequence = (sequence or "").upper()
    ids: List[int] = []
    if add_special:
        ids.append(TOKEN_TO_ID["<bos>"])
    for residue in sequence:
        ids.append(TOKEN_TO_ID.get(residue, TOKEN_TO_ID["X"]))
    if add_special:
        ids.append(TOKEN_TO_ID["<eos>"])
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: Sequence[int], strip_special: bool = True) -> str:
    """Decode token ids back to a sequence."""
    tokens = [ID_TO_TOKEN.get(int(idx), "X") for idx in ids]
    if strip_special:
        tokens = [tok for tok in tokens if tok not in {"<bos>", "<eos>", "<pad>"}]
    return "".join(tokens)


def collate_batch(
    sequences: Sequence[str],
    add_special: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert a batch of sequences into padded tensors."""
    if not sequences:
        raise ValueError("collate_batch expects at least one sequence.")

    encoded = [encode(seq, add_special=add_special) for seq in sequences]
    max_len = max(item.size(0) for item in encoded)
    batch_size = len(encoded)

    input_ids = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for idx, tensor in enumerate(encoded):
        length = tensor.size(0)
        input_ids[idx, :length] = tensor
        attention_mask[idx, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


__all__ = [
    "SPECIAL_TOKENS",
    "AMINO_ACIDS",
    "VOCAB",
    "TOKEN_TO_ID",
    "ID_TO_TOKEN",
    "PAD_TOKEN_ID",
    "encode",
    "decode",
    "collate_batch",
]
