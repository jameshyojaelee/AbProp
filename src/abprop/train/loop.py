"""High-level training loop utilities for AbProp."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader

from abprop.utils import (
    barrier,
    mlflow_default_tags,
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_run,
)


def _cosine_decay(progress: float) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class LoopConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_clipping: Optional[float] = 1.0
    grad_accumulation: int = 1
    max_steps: int = 1000
    warmup_steps: int = 0
    lr_schedule: str = "cosine"  # cosine | linear
    eval_interval: int = 200
    checkpoint_interval: int = 200
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("outputs/logs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    best_metric: str = "eval_loss"
    maximize_metric: bool = False
    precision: str = "amp"  # amp | fp32
    tasks: Tuple[str, ...] = ("mlm",)
    report_interval: int = 1


class _MetricLogger:
    """Logging helper supporting MLflow or CSV fallback."""

    def __init__(self, log_dir: Path, run_name: str = "abprop") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.log_dir / f"{run_name}.csv"
        self._csv_fp = open(self._csv_file, "w", newline="", encoding="utf-8")
        self._writer = None
        self._mlflow = None
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            try:
                import mlflow
            except ImportError:
                tracking_uri = None
            else:
                self._mlflow = mlflow
                self._mlflow.set_tracking_uri(tracking_uri)
                self._mlflow_run = self._mlflow.start_run(run_name=run_name)
        if self._mlflow is None:
            self._writer = csv.writer(self._csv_fp)
            self._writer.writerow(["step", "tag", "metric", "value"])

    def log_params(self, params: Mapping[str, object]) -> None:
        if self._mlflow is None:
            params_path = self.log_dir / "params.json"
            with open(params_path, "w", encoding="utf-8") as handle:
                json.dump(params, handle, indent=2, default=str)
        else:
            flat_params = {key: str(value) for key, value in params.items()}
            self._mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Mapping[str, float], step: int, tag: str = "train") -> None:
        if self._mlflow is not None:
            prefixed = {f"{tag}_{k}": float(v) for k, v in metrics.items()}
            self._mlflow.log_metrics(prefixed, step=step)
            return
        for key, value in metrics.items():
            self._writer.writerow([step, tag, key, float(value)])
            self._csv_fp.flush()

    def close(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
        if self._csv_fp:
            self._csv_fp.close()


class _NullLogger:
    def log_params(self, params: Mapping[str, object]) -> None:
        return

    def log_metrics(self, metrics: Mapping[str, float], step: int, tag: str = "train") -> None:
        return

    def close(self) -> None:
        return


def build_optimizer(model: nn.Module, config: LoopConfig) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: optim.Optimizer, config: LoopConfig) -> optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(config.warmup_steps))
    total_steps = max(1, int(config.max_steps))
    schedule_type = config.lr_schedule.lower()

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if schedule_type == "linear":
            return 1.0 - progress
        return _cosine_decay(progress)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TrainLoop:
    """End-to-end training loop supporting AMP, evaluation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: LoopConfig,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        *,
        device: Optional[torch.device] = None,
        log_run_name: str = "abprop",
        is_rank_zero_run: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer or build_optimizer(model, config)
        self.scheduler = scheduler or build_scheduler(self.optimizer, config)
        self.use_amp = config.precision == "amp" and self.device.type == "cuda"
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        self.output_dir = Path(config.output_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.is_rank_zero = is_rank_zero_run
        self.mlflow_run_ctx = None
        if self.is_rank_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger = _MetricLogger(self.log_dir, run_name=log_run_name)
            self.logger.log_params({"device": str(self.device), **asdict(config)})
            self.mlflow_run_ctx = mlflow_run(log_run_name, tags=mlflow_default_tags())
            self._mlflow_run = self.mlflow_run_ctx.__enter__() if self.mlflow_run_ctx else None
            mlflow_log_params({"device": str(self.device), **asdict(config)})
        else:
            self.logger = _NullLogger()
            self._mlflow_run = None

        self._init_fsdp_metadata()

        self.train_history: List[float] = [] if self.is_rank_zero else []
        self.best_metric: Optional[float] = None
        self.best_checkpoint_path = self.checkpoint_dir / "best.pt"
        self.last_checkpoint_path = self.checkpoint_dir / "last.pt"

    def _init_fsdp_metadata(self) -> None:
        self._is_fsdp = False
        self._fsdp_full_state_config = None
        self._fsdp_state_type = None
        self._fsdp_module = None
        self._fsdp_state_dict_ctx = None
        try:  # pragma: no cover - optional dependency path
            from torch.distributed.fsdp import (  # type: ignore
                FullyShardedDataParallel as _FSDP,
                StateDictType,
                FullStateDictConfig,
                state_dict_type,
            )
        except ImportError:  # pragma: no cover
            return
        if isinstance(self.model, _FSDP):
            self._is_fsdp = True
            self._fsdp_module = _FSDP
            self._fsdp_state_type = StateDictType
            self._fsdp_full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            self._fsdp_state_dict_ctx = state_dict_type

    def _prepare_batch(self, batch: Mapping[str, object]) -> Dict[str, object]:
        prepared: Dict[str, object] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, (list, tuple)):
                prepared[key] = [
                    item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item
                    for item in value
                ]
            else:
                prepared[key] = value
        return prepared

    def _compute_loss(self, batch: Mapping[str, object]) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = self._prepare_batch(batch)

        forward_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "tasks": self.config.tasks,
        }
        if "labels" in batch:
            forward_kwargs["mlm_labels"] = batch["labels"]
        if "token_labels" in batch:
            forward_kwargs["token_labels"] = batch["token_labels"]
        if "liability_ln" in batch:
            forward_kwargs["liability_targets"] = batch["liability_ln"]

        outputs = self.model(**forward_kwargs)
        loss = outputs.get("loss")
        if loss is None:
            raise RuntimeError("Model forward did not return a 'loss' tensor.")
        metrics = {"loss": loss.detach().float().item()}
        for key, value in outputs.get("metrics", {}).items():
            if torch.is_tensor(value):
                metrics[key] = float(value.detach().cpu())
            else:
                metrics[key] = float(value)
        return loss, metrics

    def _run_eval(self, dataloader: Iterable[Mapping[str, object]], max_batches: int = 50) -> Dict[str, float]:
        if not self.is_rank_zero:
            raise RuntimeError("Evaluation should only be run on rank zero.")
        self.model.eval()
        losses = []
        aggregated: Dict[str, float] = {}
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                loss, metrics = self._compute_loss(batch)
                losses.append(float(loss.detach().cpu()))
                for key, value in metrics.items():
                    aggregated.setdefault(key, 0.0)
                    aggregated[key] += value
                if idx + 1 >= max_batches:
                    break
        count = max(1, len(losses))
        averaged = {key: value / count for key, value in aggregated.items()}
        averaged["loss"] = sum(losses) / count
        self.model.train()
        return averaged

    def _should_update_best(self, metric_value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.config.maximize_metric:
            return metric_value > self.best_metric
        return metric_value < self.best_metric

    def _save_checkpoint(self, path: Path, step: int, extra: Optional[Dict[str, object]] = None) -> None:
        save_to_disk = self.is_rank_zero

        if self._is_fsdp:
            assert self._fsdp_module is not None
            assert self._fsdp_state_dict_ctx is not None
            with self._fsdp_state_dict_ctx(
                self.model,
                self._fsdp_state_type.FULL_STATE_DICT,
                self._fsdp_full_state_config,
            ):
                model_state = self.model.state_dict()
            if not save_to_disk:
                return
            state = {
                "model_state": model_state,
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "step": step,
            }
            if extra:
                state.update(extra)
            torch.save(state, path)
            return

        if not save_to_disk:
            return

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "step": step,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def _git_sha(self) -> Optional[str]:
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            return sha
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def save_run_metadata(self, config_snapshot: Mapping[str, object]) -> None:
        if not self.is_rank_zero:
            return
        snapshot_path = self.output_dir / "config_snapshot.json"
        with open(snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(config_snapshot, handle, indent=2, default=str)
        git_sha = self._git_sha()
        if git_sha:
            with open(self.output_dir / "git_commit.txt", "w", encoding="utf-8") as handle:
                handle.write(git_sha + "\n")

    def fit(
        self,
        train_dataloader: Iterable[Mapping[str, object]],
        eval_dataloader: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> None:
        grad_accum = max(1, self.config.grad_accumulation)
        max_steps = int(self.config.max_steps)

        sampler = getattr(train_dataloader, "sampler", None)
        iterator = iter(train_dataloader)
        epoch = 0
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for step in range(1, max_steps + 1):
            step_start = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                iterator = iter(train_dataloader)
                batch = next(iterator)

            with amp.autocast(enabled=self.use_amp):
                loss, metrics = self._compute_loss(batch)
                scaled_loss = loss / grad_accum
            self.scaler.scale(scaled_loss).backward()

            if step % grad_accum == 0:
                if self.config.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            if self.is_rank_zero:
                elapsed = time.perf_counter() - step_start
                metrics["step_time"] = elapsed
                if self.device.type == "cuda":
                    peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    metrics["gpu_mem_mb"] = peak_mem
                self.train_history.append(metrics["loss"])
                if (
                    step % self.config.report_interval == 0
                    or step == 1
                    or step == max_steps
                ):
                    self.logger.log_metrics(metrics, step=step, tag="train")
                    mlflow_log_metrics({f"train/{k}": v for k, v in metrics.items()}, step=step)

            if (
                eval_dataloader is not None
                and self.config.eval_interval > 0
                and step % self.config.eval_interval == 0
            ):
                should_save_best = False
                eval_metrics = None
                if self.is_rank_zero:
                    eval_metrics = self._run_eval(eval_dataloader)
                    self.logger.log_metrics(eval_metrics, step=step, tag="eval")
                    mlflow_log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)
                    metric_value = eval_metrics.get(self.config.best_metric, eval_metrics["loss"])
                    if self._should_update_best(metric_value):
                        self.best_metric = metric_value
                        should_save_best = True
                if dist.is_initialized():
                    flag_device = self.device if self.device.type == "cuda" else torch.device("cpu")
                    flag = torch.tensor(1 if should_save_best else 0, device=flag_device)
                    dist.broadcast(flag, src=0)
                    should_save_best = bool(flag.item())
                if should_save_best:
                    self._save_checkpoint(self.best_checkpoint_path, step, extra={"metrics": eval_metrics or {}})
                barrier()

            if (
                self.config.checkpoint_interval > 0
                and step % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoint(self.last_checkpoint_path, step)

        if self.is_rank_zero or self._is_fsdp:
            self._save_checkpoint(self.last_checkpoint_path, max_steps)
        if self.is_rank_zero:
            self.logger.close()
        if self.mlflow_run_ctx is not None:
            self.mlflow_run_ctx.__exit__(None, None, None)


__all__ = ["LoopConfig", "TrainLoop", "build_optimizer", "build_scheduler"]
