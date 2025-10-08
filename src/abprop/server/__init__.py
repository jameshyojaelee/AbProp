"""Minimal REST server scaffolding for AbProp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from abprop.tokenizers import AminoAcidTokenizer
from abprop.models import SimpleTransformerEncoder, TransformerConfig


@dataclass
class InferenceEngine:
    tokenizer: AminoAcidTokenizer
    model: SimpleTransformerEncoder

    @classmethod
    def build(cls) -> "InferenceEngine":
        tokenizer = AminoAcidTokenizer.build_default()
        model = SimpleTransformerEncoder(TransformerConfig())
        model.eval()
        return cls(tokenizer=tokenizer, model=model)

    def predict(self, sequence: str) -> Dict[str, float]:
        """Return placeholder predictions for a given sequence."""
        _ = self.tokenizer.encode(sequence)
        # Placeholder scores until the model is trained.
        return {"mlm_perplexity": 0.0, "cdr_frame_prob": 0.5}


__all__ = ["InferenceEngine"]

