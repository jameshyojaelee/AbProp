"""Tokenization utilities for antibody sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .aa import (
    AMINO_ACIDS,
    ID_TO_TOKEN,
    SPECIAL_TOKENS,
    TOKEN_TO_ID,
    VOCAB,
    collate_batch,
    decode,
    encode,
)


@dataclass
class AminoAcidTokenizer:
    """Simple tokenizer wrapper retaining legacy API."""

    vocab: List[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def build_default(cls) -> "AminoAcidTokenizer":
        return cls(vocab=VOCAB, token_to_id=TOKEN_TO_ID, id_to_token=ID_TO_TOKEN)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        return encode(sequence, add_special=add_special_tokens).tolist()

    def decode(self, ids: Sequence[int], remove_special_tokens: bool = True) -> str:
        return decode(ids, strip_special=remove_special_tokens)


__all__ = [
    "AminoAcidTokenizer",
    "SPECIAL_TOKENS",
    "AMINO_ACIDS",
    "encode",
    "decode",
    "collate_batch",
    "VOCAB",
    "TOKEN_TO_ID",
    "ID_TO_TOKEN",
]
