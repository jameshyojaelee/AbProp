"""FastAPI server for AbProp inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from abprop.data import build_collate_fn
from abprop.eval.uncertainty import sequence_perplexity_from_logits
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import load_yaml_config


class ModelWrapper:
    def __init__(
        self,
        checkpoint: Union[Path, List[Path]],
        model_config: Path,
        device: Optional[str] = None,
        ensemble_mode: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ensemble_mode = ensemble_mode

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

        # Load single or multiple models
        if isinstance(checkpoint, list):
            self.models = []
            for ckpt in checkpoint:
                model = AbPropModel(self.model_cfg).to(self.device)
                state = torch.load(ckpt, map_location=self.device, weights_only=False)
                model_state = state.get("model_state_dict", state.get("model_state", state))
                model.load_state_dict(model_state, strict=False)
                model.eval()
                self.models.append(model)
            self.ensemble_mode = True
        else:
            self.model = AbPropModel(self.model_cfg).to(self.device)
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model_state = state.get("model_state_dict", state.get("model_state", state))
            self.model.load_state_dict(model_state, strict=False)
            self.model.eval()
            self.models = [self.model]

        self.collate = build_collate_fn(generate_mlm=True)

    def score_perplexity(
        self, sequences: List[str], return_std: bool = False
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Score perplexity with optional ensemble std deviation."""
        batch = self.collate([{"sequence": seq} for seq in sequences])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        all_perplexities = []

        for model in self.models:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, tasks=("mlm",))
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
            perplexities = torch.exp(per_seq_loss)
            all_perplexities.append(perplexities)

        # Average across models if ensemble
        stacked = torch.stack(all_perplexities)
        mean_perplexity = stacked.mean(dim=0).tolist()

        if return_std and self.ensemble_mode and len(self.models) > 1:
            std_perplexity = stacked.std(dim=0).tolist()
            return {"mean": mean_perplexity, "std": std_perplexity}

        return mean_perplexity

    def score_liabilities(
        self, sequences: List[str], return_std: bool = False
    ) -> Union[List[Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
        """Score liabilities with optional ensemble std deviation."""
        batch = self.collate([{"sequence": seq} for seq in sequences])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        all_preds = []

        for model in self.models:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, tasks=("reg",))
                preds = outputs["regression"]
                all_preds.append(preds)

        # Average across models if ensemble
        stacked = torch.stack(all_preds)
        mean_preds = stacked.mean(dim=0).cpu().numpy()
        mean_results = [
            dict(zip(self.model_cfg.liability_keys, row)) for row in mean_preds
        ]

        if return_std and self.ensemble_mode and len(self.models) > 1:
            std_preds = stacked.std(dim=0).cpu().numpy()
            std_results = [
                dict(zip(self.model_cfg.liability_keys, row)) for row in std_preds
            ]
            return {"mean": mean_results, "std": std_results}

        return mean_results

    def score_perplexity_uncertainty(
        self,
        sequences: List[str],
        *,
        mc_samples: int = 20,
        dropout: bool = True,
        pad_token_id: int = 0,
    ) -> Dict[str, List[float]]:
        """Return mean and variance of perplexity estimates via MC dropout / ensembles."""
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive.")
        batch = self.collate([{"sequence": seq} for seq in sequences])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        sample_perplexities: List[torch.Tensor] = []
        mc_passes = mc_samples if dropout else 1

        with torch.no_grad():
            for model in self.models:
                passes = model.stochastic_forward(
                    input_ids,
                    attention_mask,
                    tasks=("mlm",),
                    mc_samples=mc_passes,
                    enable_dropout=dropout,
                    no_grad=True,
                )
                for out in passes:
                    sample_perplexities.append(
                        sequence_perplexity_from_logits(out["mlm_logits"].detach(), input_ids, pad_token_id=pad_token_id)
                        .detach()
                        .cpu()
                    )

        samples = torch.stack(sample_perplexities, dim=0)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0, unbiased=False)
        std = variance.clamp_min(0.0).sqrt()
        return {
            "mean": mean.tolist(),
            "variance": variance.tolist(),
            "std": std.tolist(),
        }

    def score_liabilities_uncertainty(
        self,
        sequences: List[str],
        *,
        mc_samples: int = 20,
        dropout: bool = True,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Return mean and variance of liability predictions."""
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive.")
        batch = self.collate([{"sequence": seq} for seq in sequences])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        sample_preds: List[torch.Tensor] = []
        mc_passes = mc_samples if dropout else 1

        with torch.no_grad():
            for model in self.models:
                passes = model.stochastic_forward(
                    input_ids,
                    attention_mask,
                    tasks=("reg",),
                    mc_samples=mc_passes,
                    enable_dropout=dropout,
                    no_grad=True,
                )
                for out in passes:
                    sample_preds.append(out["regression"].detach().cpu())

        stacked = torch.stack(sample_preds, dim=0)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)
        std = variance.clamp_min(0.0).sqrt()

        def _to_dict(tensor: torch.Tensor) -> List[Dict[str, float]]:
            results = []
            for row in tensor:
                results.append(
                    {
                        key: float(value)
                        for key, value in zip(self.model_cfg.liability_keys, row.tolist())
                    }
                )
            return results

        return {
            "mean": _to_dict(mean),
            "variance": _to_dict(variance),
            "std": _to_dict(std),
        }


def create_app(
    checkpoint: Union[Path, List[Path]],
    model_config: Path,
    device: Optional[str] = None,
    ensemble_mode: bool = False,
) -> FastAPI:
    """
    Create FastAPI app with single or ensemble model inference.

    Args:
        checkpoint: Path to single checkpoint or list of checkpoints for ensemble
        model_config: Path to model configuration YAML
        device: Device to run inference on (default: auto-detect)
        ensemble_mode: Whether to enable ensemble mode with multiple models

    Returns:
        FastAPI application instance
    """
    wrapper = ModelWrapper(checkpoint, model_config, device, ensemble_mode)
    app = FastAPI(title="AbProp Inference API")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": str(wrapper.device),
            "ensemble_mode": wrapper.ensemble_mode,
            "n_models": len(wrapper.models),
        }

    @app.post("/score/perplexity")
    def score_perplexity(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        return_std = payload.get("return_std", False)
        scores = wrapper.score_perplexity(sequences, return_std=return_std)
        return {"perplexity": scores}

    @app.post("/score/liabilities")
    def score_liabilities(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        return_std = payload.get("return_std", False)
        scores = wrapper.score_liabilities(sequences, return_std=return_std)
        return {"liabilities": scores}

    @app.post("/score/perplexity/uncertainty")
    def score_perplexity_uncertainty(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        mc_samples = int(payload.get("mc_samples", 20))
        dropout = bool(payload.get("dropout", True))
        try:
            stats = wrapper.score_perplexity_uncertainty(sequences, mc_samples=mc_samples, dropout=dropout)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"perplexity": stats}

    @app.post("/score/liabilities/uncertainty")
    def score_liabilities_uncertainty(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        mc_samples = int(payload.get("mc_samples", 20))
        dropout = bool(payload.get("dropout", True))
        try:
            stats = wrapper.score_liabilities_uncertainty(sequences, mc_samples=mc_samples, dropout=dropout)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"liabilities": stats}

    return app


def create_ensemble_app_from_cv(
    cv_dir: Path, model_config: Path, device: Optional[str] = None
) -> FastAPI:
    """
    Create FastAPI app with ensemble inference from CV fold checkpoints.

    Args:
        cv_dir: Directory containing CV fold subdirectories
        model_config: Path to model configuration YAML
        device: Device to run inference on (default: auto-detect)

    Returns:
        FastAPI application instance with ensemble inference
    """
    # Discover all fold checkpoints
    fold_dirs = sorted([d for d in cv_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    checkpoints = []

    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "checkpoints" / "best.pt"
        if ckpt_path.exists():
            checkpoints.append(ckpt_path)

    if not checkpoints:
        raise ValueError(f"No fold checkpoints found in {cv_dir}")

    return create_app(checkpoints, model_config, device, ensemble_mode=True)


__all__ = ["create_app", "create_ensemble_app_from_cv", "ModelWrapper"]
