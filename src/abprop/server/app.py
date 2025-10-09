"""FastAPI server for AbProp inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException

from abprop.data import build_collate_fn
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import load_yaml_config


class ModelWrapper:
    def __init__(self, checkpoint: Path, model_config: Path, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = load_yaml_config(model_config)
        self.model_cfg = TransformerConfig(
            vocab_size=cfg.get("vocab_size", TransformerConfig.vocab_size),
            d_model=cfg.get("d_model", TransformerConfig.d_model),
            nhead=cfg.get("nhead", TransformerConfig.nhead),
            num_layers=cfg.get("num_layers", TransformerConfig.num_layers),
            dim_feedforward=cfg.get("dim_feedforward", TransformerConfig.dim_feedforward),
            dropout=cfg.get("dropout", TransformerConfig.dropout),
            liability_keys=tuple(cfg.get("liability_keys", list(TransformerConfig.liability_keys))),
        )
        self.model = AbPropModel(self.model_cfg).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        model_state = state.get("model_state", state)
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()
        self.collate = build_collate_fn(generate_mlm=True)

    def score_perplexity(self, sequences: List[str]) -> List[float]:
        batch = self.collate([{"sequence": seq} for seq in sequences])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, tasks=("mlm",))
            logits = outputs["mlm_logits"]
        labels = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
            reduction="none",
        ).view(labels.size())
        mask = (labels != 0).float()
        per_seq_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.exp(per_seq_loss).tolist()

    def score_liabilities(self, sequences: List[str]) -> List[Dict[str, float]]:
        batch = self.collate([{"sequence": seq} for seq in sequences])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, tasks=("reg",))
            preds = outputs["regression"].cpu().tolist()
        return [dict(zip(self.model_cfg.liability_keys, row)) for row in preds]


def create_app(checkpoint: Path, model_config: Path, device: Optional[str] = None) -> FastAPI:
    wrapper = ModelWrapper(checkpoint, model_config, device)
    app = FastAPI(title="AbProp Inference API")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "device": str(wrapper.device)}

    @app.post("/score/perplexity")
    def score_perplexity(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        scores = wrapper.score_perplexity(sequences)
        return {"perplexity": scores}

    @app.post("/score/liabilities")
    def score_liabilities(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        scores = wrapper.score_liabilities(sequences)
        return {"liabilities": scores}

    return app


__all__ = ["create_app"]

