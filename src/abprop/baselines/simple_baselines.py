"""Simple heuristic baselines for AbProp benchmarks."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from abprop.data import OASDataset
from abprop.eval.metrics import classification_summary, regression_summary
from abprop.tokenizers import AMINO_ACIDS


VOCAB = list(AMINO_ACIDS)
VOCAB_SIZE = len(VOCAB)
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(VOCAB)}


@dataclass
class SequenceSplit:
    sequences: List[str]
    chains: List[str]
    liabilities: List[Dict[str, float]]
    cdr_masks: List[Optional[List[int]]]


@dataclass
class BaselineResult:
    name: str
    metrics: Dict[str, float]
    per_item: Dict[str, List[float]] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "per_item": self.per_item,
            "extra": self.extra,
        }


def load_split(dataset_path: Path, split: str) -> SequenceSplit:
    dataset = OASDataset(dataset_path, split=split)
    sequences: List[str] = []
    chains: List[str] = []
    liabilities: List[Dict[str, float]] = []
    cdr_masks: List[Optional[List[int]]] = []
    for item in dataset:
        sequences.append(str(item.get("sequence", "")))
        chains.append(str(item.get("chain", "")))
        liabilities.append(dict(item.get("liability_ln") or {}))
        mask = item.get("cdr_mask")
        if mask is None:
            cdr_masks.append(None)
        else:
            cdr_masks.append(list(mask))
    return SequenceSplit(sequences=sequences, chains=chains, liabilities=liabilities, cdr_masks=cdr_masks)


def _sequence_tokens(sequence: str) -> List[str]:
    return [aa if aa in AA_TO_INDEX else "X" for aa in sequence]


# ---------------------------------------------------------------------------
# Perplexity baselines
# ---------------------------------------------------------------------------


class UniformPerplexityBaseline:
    name = "uniform"

    def fit(self, train: SequenceSplit) -> None:
        return

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        per_seq_perplexities: List[float] = []
        per_seq_nll: List[float] = []
        vocab_size = float(VOCAB_SIZE)
        log_prob = math.log(vocab_size)
        for sequence in split.sequences:
            tokens = _sequence_tokens(sequence)
            if not tokens:
                continue
            neg_log_likelihood = log_prob * len(tokens)
            ce = neg_log_likelihood / len(tokens)
            per_seq_nll.append(neg_log_likelihood)
            per_seq_perplexities.append(math.exp(ce))
        metrics = {
            "perplexity": float(np.mean(per_seq_perplexities)) if per_seq_perplexities else float("nan"),
            "cross_entropy": float(np.mean(per_seq_nll) / np.mean([len(_sequence_tokens(seq)) for seq in split.sequences if seq])) if per_seq_nll else float(
                "nan"
            ),
        }
        return BaselineResult(
            name=self.name,
            metrics=metrics,
            per_item={
                "perplexity": per_seq_perplexities,
                "negative_log_likelihood": per_seq_nll,
            },
        )


class UnigramPerplexityBaseline:
    name = "unigram"

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.log_probs: Dict[str, float] = {}

    def fit(self, train: SequenceSplit) -> None:
        counts: Counter[str] = Counter()
        total = 0.0
        for sequence in train.sequences:
            for token in _sequence_tokens(sequence):
                counts[token] += 1
                total += 1
        vocab = set(VOCAB)
        denom = total + self.smoothing * len(vocab)
        for token in vocab:
            count = counts[token]
            prob = (count + self.smoothing) / denom
            self.log_probs[token] = math.log(prob)

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        per_seq_perplexities: List[float] = []
        per_seq_nll: List[float] = []
        for sequence in split.sequences:
            tokens = _sequence_tokens(sequence)
            if not tokens:
                continue
            neg_log_likelihood = 0.0
            for token in tokens:
                neg_log_likelihood -= self.log_probs.get(token, math.log(1.0 / VOCAB_SIZE))
            ce = neg_log_likelihood / len(tokens)
            per_seq_nll.append(neg_log_likelihood)
            per_seq_perplexities.append(math.exp(ce))
        metrics = {
            "perplexity": float(np.mean(per_seq_perplexities)) if per_seq_perplexities else float("nan"),
            "cross_entropy": float(np.mean(per_seq_nll) / np.mean([len(_sequence_tokens(seq)) for seq in split.sequences if seq])) if per_seq_nll else float(
                "nan"
            ),
        }
        return BaselineResult(
            name=self.name,
            metrics=metrics,
            per_item={
                "perplexity": per_seq_perplexities,
                "negative_log_likelihood": per_seq_nll,
            },
        )


class BigramPerplexityBaseline:
    name = "bigram"

    def __init__(self, smoothing: float = 0.1) -> None:
        self.smoothing = smoothing
        self.bigram_log_probs: Dict[Tuple[str, str], float] = {}
        self.unigram_log_probs: Dict[str, float] = {}

    def fit(self, train: SequenceSplit) -> None:
        unigram_counts: Counter[str] = Counter()
        bigram_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for sequence in train.sequences:
            tokens = ["<s>"] + _sequence_tokens(sequence)
            for prev, current in zip(tokens[:-1], tokens[1:]):
                unigram_counts[current] += 1
                bigram_counts[prev][current] += 1

        total = sum(unigram_counts.values()) + self.smoothing * VOCAB_SIZE
        for token in VOCAB:
            prob = (unigram_counts[token] + self.smoothing) / total
            self.unigram_log_probs[token] = math.log(prob)

        for prev, counter in bigram_counts.items():
            denom = sum(counter.values()) + self.smoothing * VOCAB_SIZE
            for token in VOCAB:
                prob = (counter[token] + self.smoothing) / denom
                self.bigram_log_probs[(prev, token)] = math.log(prob)

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        per_seq_perplexities: List[float] = []
        per_seq_nll: List[float] = []
        for sequence in split.sequences:
            tokens = _sequence_tokens(sequence)
            if not tokens:
                continue
            tokens_with_start = ["<s>"] + tokens
            neg_log_likelihood = 0.0
            for prev, current in zip(tokens_with_start[:-1], tokens_with_start[1:]):
                neg_log_likelihood -= self.bigram_log_probs.get(
                    (prev, current), self.unigram_log_probs.get(current, math.log(1.0 / VOCAB_SIZE))
                )
            ce = neg_log_likelihood / len(tokens)
            per_seq_nll.append(neg_log_likelihood)
            per_seq_perplexities.append(math.exp(ce))
        metrics = {
            "perplexity": float(np.mean(per_seq_perplexities)) if per_seq_perplexities else float("nan"),
            "cross_entropy": float(np.mean(per_seq_nll) / np.mean([len(_sequence_tokens(seq)) for seq in split.sequences if seq])) if per_seq_nll else float(
                "nan"
            ),
        }
        return BaselineResult(
            name=self.name,
            metrics=metrics,
            per_item={
                "perplexity": per_seq_perplexities,
                "negative_log_likelihood": per_seq_nll,
            },
        )


# ---------------------------------------------------------------------------
# CDR classification baselines
# ---------------------------------------------------------------------------


class RandomCDRBaseline:
    name = "random"

    def __init__(self, positive_prob: float = 0.5) -> None:
        self.positive_prob = positive_prob

    def fit(self, train: SequenceSplit) -> None:
        return

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        tp = fp = tn = fn = 0
        per_sequence_accuracy: List[float] = []
        rng = random.Random(0)
        for mask in split.cdr_masks:
            if not mask:
                continue
            preds = [1 if rng.random() < self.positive_prob else 0 for _ in mask]
            correct = sum(int(p == t) for p, t in zip(preds, mask))
            per_sequence_accuracy.append(correct / len(mask))
            for p, t in zip(preds, mask):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 0:
                    tn += 1
                elif p == 0 and t == 1:
                    fn += 1
        metrics = classification_summary(tp, fp, tn, fn)
        metrics["accuracy"] = (tp + tn) / max(1, tp + tn + fp + fn)
        return BaselineResult(
            name=self.name,
            metrics={k: float(v) for k, v in metrics.items()},
            per_item={"accuracy": per_sequence_accuracy},
            extra={"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}},
        )


class FrequencyCDRBaseline:
    name = "frequency"

    def __init__(self) -> None:
        self.positive_prob = 0.0

    def fit(self, train: SequenceSplit) -> None:
        positives = 0
        total = 0
        for mask in train.cdr_masks:
            if not mask:
                continue
            positives += sum(mask)
            total += len(mask)
        if total == 0:
            self.positive_prob = 0.0
        else:
            self.positive_prob = positives / total

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        tp = fp = tn = fn = 0
        per_sequence_accuracy: List[float] = []
        for mask in split.cdr_masks:
            if not mask:
                continue
            pred_label = 1 if self.positive_prob >= 0.5 else 0
            preds = [pred_label] * len(mask)
            correct = sum(int(p == t) for p, t in zip(preds, mask))
            per_sequence_accuracy.append(correct / len(mask))
            for p, t in zip(preds, mask):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 0:
                    tn += 1
                elif p == 0 and t == 1:
                    fn += 1
        metrics = classification_summary(tp, fp, tn, fn)
        metrics["accuracy"] = (tp + tn) / max(1, tp + tn + fp + fn)
        return BaselineResult(
            name=self.name,
            metrics={k: float(v) for k, v in metrics.items()},
            per_item={"accuracy": per_sequence_accuracy},
            extra={"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}},
        )


def _sequence_vector(sequence: str) -> np.ndarray:
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for token in _sequence_tokens(sequence):
        idx = AA_TO_INDEX.get(token)
        if idx is not None:
            vec[idx] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec


class KNNCdrBaseline:
    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.vectors: List[np.ndarray] = []
        self.masks: List[List[int]] = []

    def fit(self, train: SequenceSplit) -> None:
        self.vectors = []
        self.masks = []
        for seq, mask in zip(train.sequences, train.cdr_masks):
            if not mask:
                continue
            self.vectors.append(_sequence_vector(seq))
            self.masks.append(mask)

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        if not self.vectors:
            return BaselineResult(self.name, metrics={"accuracy": float("nan")})
        train_vectors = np.stack(self.vectors)
        tp = fp = tn = fn = 0
        per_sequence_accuracy: List[float] = []
        for seq, mask in zip(split.sequences, split.cdr_masks):
            if not mask:
                continue
            vec = _sequence_vector(seq)
            distances = np.linalg.norm(train_vectors - vec, axis=1)
            topk_idx = np.argsort(distances)[: self.k]
            votes = np.zeros(len(mask), dtype=np.float32)
            counts = np.zeros(len(mask), dtype=np.float32)
            for idx in topk_idx:
                neighbor_mask = self.masks[idx]
                for pos in range(min(len(neighbor_mask), len(mask))):
                    votes[pos] += neighbor_mask[pos]
                    counts[pos] += 1.0
            preds = []
            correct = 0
            for pos, target in enumerate(mask):
                if counts[pos] == 0:
                    pred = 1 if self.k == 0 else 0
                else:
                    pred = 1 if (votes[pos] / counts[pos]) >= 0.5 else 0
                preds.append(pred)
                if pred == target:
                    correct += 1
            per_sequence_accuracy.append(correct / len(mask))
            for p, t in zip(preds, mask):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 0:
                    tn += 1
                elif p == 0 and t == 1:
                    fn += 1
        metrics = classification_summary(tp, fp, tn, fn)
        metrics["accuracy"] = (tp + tn) / max(1, tp + tn + fp + fn)
        return BaselineResult(
            name=self.name,
            metrics={k: float(v) for k, v in metrics.items()},
            per_item={"accuracy": per_sequence_accuracy},
            extra={"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}},
        )


# ---------------------------------------------------------------------------
# Liability regression baselines
# ---------------------------------------------------------------------------


class MeanLiabilityBaseline:
    name = "mean"

    def __init__(self) -> None:
        self.mean_vector: Dict[str, float] = {}

    def fit(self, train: SequenceSplit) -> None:
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for liability in train.liabilities:
            for key, value in liability.items():
                sums[key] += float(value)
                counts[key] += 1
        self.mean_vector = {k: sums[k] / max(1, counts[k]) for k in sums}

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        preds = []
        targets = []
        squared_errors: List[float] = []
        for liability in split.liabilities:
            target_vector = [float(liability.get(key, 0.0)) for key in self.mean_vector]
            pred_vector = [self.mean_vector[key] for key in self.mean_vector]
            preds.append(pred_vector)
            targets.append(target_vector)
            sq_err = [(p - t) ** 2 for p, t in zip(pred_vector, target_vector)]
            squared_errors.extend(sq_err)
        if not preds:
            return BaselineResult(self.name, metrics={})
        pred_tensor = torch.tensor(preds, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        summary = regression_summary(pred_tensor, target_tensor)
        return BaselineResult(
            name=self.name,
            metrics={k: float(v) for k, v in summary.items()},
            per_item={"squared_error": squared_errors},
        )


class NearestNeighborLiabilityBaseline:
    name = "nearest_neighbor"

    def __init__(self, k: int = 1) -> None:
        self.k = k
        self.vectors: List[np.ndarray] = []
        self.liability_vectors: List[np.ndarray] = []
        self.keys: List[str] = []

    def fit(self, train: SequenceSplit) -> None:
        self.vectors = []
        self.liability_vectors = []
        keys = set()
        for liability in train.liabilities:
            keys.update(liability.keys())
        self.keys = sorted(keys)
        for seq, liability in zip(train.sequences, train.liabilities):
            self.vectors.append(_sequence_vector(seq))
            self.liability_vectors.append(np.array([float(liability.get(key, 0.0)) for key in self.keys], dtype=np.float32))
        if self.vectors:
            self.vectors = list(self.vectors)

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        if not self.vectors:
            return BaselineResult(self.name, metrics={})
        train_vectors = np.stack(self.vectors)
        train_liabilities = np.stack(self.liability_vectors)
        preds = []
        targets = []
        squared_errors: List[float] = []
        for seq, liability in zip(split.sequences, split.liabilities):
            vec = _sequence_vector(seq)
            distances = np.linalg.norm(train_vectors - vec, axis=1)
            topk_idx = np.argsort(distances)[: self.k]
            pred_vector = np.mean(train_liabilities[topk_idx], axis=0)
            target_vector = np.array([float(liability.get(key, 0.0)) for key in self.keys], dtype=np.float32)
            preds.append(pred_vector.tolist())
            targets.append(target_vector.tolist())
            squared_errors.extend(((pred_vector - target_vector) ** 2).tolist())
        pred_tensor = torch.tensor(preds, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        summary = regression_summary(pred_tensor, target_tensor)
        return BaselineResult(
            name=self.name,
            metrics={k: float(v) for k, v in summary.items()},
            per_item={"squared_error": squared_errors},
        )


class MotifLiabilityBaseline:
    name = "motif"

    def __init__(self) -> None:
        self.keys: List[str] = []

    def fit(self, train: SequenceSplit) -> None:
        keys = set()
        for liability in train.liabilities:
            keys.update(liability.keys())
        self.keys = sorted(keys)

    def _motif_score(self, sequence: str) -> Dict[str, float]:
        seq_upper = sequence.upper()
        scores = {}
        scores["nglyc"] = 1.0 if any(seq_upper[i] == "N" and seq_upper[i + 1] in ("S", "T") for i in range(len(seq_upper) - 1)) else 0.0
        scores["deamidation"] = 1.0 if "NG" in seq_upper or "DG" in seq_upper else 0.0
        scores["isomerization"] = 1.0 if "DG" in seq_upper else 0.0
        scores["oxidation"] = 1.0 if "M" in seq_upper or "W" in seq_upper else 0.0
        scores["free_cysteines"] = 1.0 if seq_upper.count("C") % 2 == 1 else 0.0
        return scores

    def evaluate(self, split: SequenceSplit) -> BaselineResult:
        preds = []
        targets = []
        squared_errors: List[float] = []
        for seq, liability in zip(split.sequences, split.liabilities):
            motif_scores = self._motif_score(seq)
            pred_vector = [motif_scores.get(key, 0.0) for key in self.keys]
            target_vector = [float(liability.get(key, 0.0)) for key in self.keys]
            preds.append(pred_vector)
            targets.append(target_vector)
            squared_errors.extend([(p - t) ** 2 for p, t in zip(pred_vector, target_vector)])
        if not preds:
            return BaselineResult(self.name, metrics={})
        pred_tensor = torch.tensor(preds, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        summary = regression_summary(pred_tensor, target_tensor)
        return BaselineResult(
            name=self.name,
            metrics={k: float(v) for k, v in summary.items()},
            per_item={"squared_error": squared_errors},
        )


__all__ = [
    "SequenceSplit",
    "BaselineResult",
    "load_split",
    "UniformPerplexityBaseline",
    "UnigramPerplexityBaseline",
    "BigramPerplexityBaseline",
    "RandomCDRBaseline",
    "FrequencyCDRBaseline",
    "KNNCdrBaseline",
    "MeanLiabilityBaseline",
    "NearestNeighborLiabilityBaseline",
    "MotifLiabilityBaseline",
]

