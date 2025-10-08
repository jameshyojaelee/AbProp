"""Tokenization utilities for antibody sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<mask>"]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]


@dataclass
class AminoAcidTokenizer:
    """Simple character-level tokenizer for antibody sequences."""

    vocab: List[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def build_default(cls) -> "AminoAcidTokenizer":
        vocab = SPECIAL_TOKENS + AMINO_ACIDS
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return cls(vocab=vocab, token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        bos = [self.token_to_id["<bos>"]] if add_special_tokens else []
        eos = [self.token_to_id["<eos>"]] if add_special_tokens else []
        ids = [self.token_to_id.get(char, self.token_to_id["X"]) for char in sequence]
        return bos + ids + eos

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token.get(idx, "X") for idx in ids]
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in {"<bos>", "<eos>", "<pad>"}]
        return "".join(tokens)


__all__ = ["AminoAcidTokenizer", "SPECIAL_TOKENS", "AMINO_ACIDS"]

