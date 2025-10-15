"""Dataset and sampling utilities for AbProp."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

try:
    import pyarrow.dataset as pa_dataset
except ImportError:  # pragma: no cover - pyarrow is a dependency, but guard just in case
    pa_dataset = None

from abprop.tokenizers import TOKEN_TO_ID, VOCAB, collate_batch


def _maybe_to_dict(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise TypeError(f"Expected dict-like liability entry, received {type(value)}")


def _maybe_to_list(value: object) -> Optional[List[int]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        tokens = [item for item in value.replace(" ", "").split(",") if item]
        if tokens:
            return [int(tok) for tok in tokens]
    return None


def _normalize_optional(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class OASDataset(Dataset):
    """Lightweight dataset backed by partitioned Parquet files."""

    def __init__(
        self,
        parquet_dir: Path | str,
        split: str,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.split = split
        available_columns: set[str] = set()
        if pa_dataset is not None:
            try:
                available_columns = set(pa_dataset.dataset(str(self.parquet_dir)).schema.names)
            except FileNotFoundError:
                pass

        default_columns = {"sequence", "chain", "liability_ln", "length"}
        metadata_columns = {"species", "germline_v", "germline_j", "cdr_mask"}
        default_columns.update(metadata_columns & available_columns)

        self.columns = set(columns or default_columns)
        required = {"sequence", "chain", "liability_ln", "length"}
        self.columns.update(required)

        if available_columns:
            missing_optional = {
                column
                for column in self.columns
                if column not in available_columns and column not in required and column != "split"
            }
            if missing_optional:
                self.columns -= missing_optional

        read_columns = None if columns is None else list(self.columns | {"split"})
        df = pd.read_parquet(
            self.parquet_dir,
            columns=read_columns,
            filters=[("split", "=", split)],
        )
        if df.empty:
            raise ValueError(f"No records found for split '{split}' in {self.parquet_dir}.")
        if "liability_ln" in df.columns:
            df["liability_ln"] = df["liability_ln"].apply(_maybe_to_dict)
        if "cdr_mask" in df.columns:
            df["cdr_mask"] = df["cdr_mask"].apply(_maybe_to_list)

        self.frame = df.reset_index(drop=True)
        self.lengths = self.frame["length"].astype(int).tolist()
        self.has_cdr = "cdr_mask" in self.frame.columns
        self._optional_fields = [
            column
            for column in self.frame.columns
            if column
            not in {
                "sequence",
                "chain",
                "liability_ln",
                "length",
                "cdr_mask",
                "split",
            }
        ]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.frame.iloc[idx]
        item: Dict[str, object] = {
            "sequence": row["sequence"],
            "chain": row["chain"],
            "liability_ln": row["liability_ln"],
            "length": int(row["length"]),
            "species": _normalize_optional(row.get("species")),
            "germline_v": _normalize_optional(row.get("germline_v")),
            "germline_j": _normalize_optional(row.get("germline_j")),
        }
        if self.has_cdr:
            cdr_mask = row["cdr_mask"]
            item["cdr_mask"] = cdr_mask
        for column in self._optional_fields:
            if column in item:
                continue
            item[column] = _normalize_optional(row[column])
        return item


class BucketBatchSampler(Sampler[List[int]]):
    """Length-aware sampler that forms batches with similar sequence lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bins: Sequence[int],
        *,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.bin_boundaries = sorted(int(b) for b in bins)
        if not self.bin_boundaries:
            raise ValueError("At least one bin boundary is required.")
        if any(b <= 0 for b in self.bin_boundaries):
            raise ValueError("Bin boundaries must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        self._buckets: Dict[int, List[int]] = self._assign_to_buckets()
        self._num_batches = None
        self._epoch = 0

    def _assign_to_buckets(self) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = {boundary: [] for boundary in self.bin_boundaries}
        buckets[self.bin_boundaries[-1]] = []
        overflow: List[int] = []
        for idx, length in enumerate(self.lengths):
            bucket = self._bucket_for_length(length)
            if bucket is None:
                overflow.append(idx)
            else:
                buckets[bucket].append(idx)
        if overflow:
            boundary = self.bin_boundaries[-1]
            buckets.setdefault(boundary, []).extend(overflow)
        return buckets

    def _bucket_for_length(self, length: int) -> Optional[int]:
        for boundary in self.bin_boundaries:
            if length <= boundary:
                return boundary
        return None

    def _iter_batches(self, *, shuffle: bool, rng: Optional[random.Random]) -> Iterator[List[int]]:
        for boundary in self.bin_boundaries:
            indices = self._buckets.get(boundary, [])
            if not indices:
                continue
            if shuffle and rng is not None:
                indices = indices.copy()
                rng.shuffle(indices)
            max_tokens = self.max_tokens
            if max_tokens:
                max_tokens = int(max_tokens)
                dynamic_bs = max(1, min(self.batch_size, max_tokens // max(1, boundary)))
            else:
                dynamic_bs = self.batch_size
            for start in range(0, len(indices), dynamic_bs):
                yield indices[start : start + dynamic_bs]

    def __iter__(self) -> Iterator[List[int]]:
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch) if self.shuffle else None
        return self._iter_batches(shuffle=self.shuffle, rng=rng)

    def __len__(self) -> int:
        if self._num_batches is None:
            count = 0
            for _ in self._iter_batches(shuffle=False, rng=None):
                count += 1
            self._num_batches = count
        return self._num_batches


def build_collate_fn(
    *,
    generate_mlm: bool = True,
    mlm_probability: float = 0.15,
    rng: Optional[random.Random] = None,
) -> callable:
    """Create a collate function that tokenizes sequences and applies MLM masking."""

    rng = rng or random.Random()

    pad_id = TOKEN_TO_ID["<pad>"]
    bos_id = TOKEN_TO_ID["<bos>"]
    eos_id = TOKEN_TO_ID["<eos>"]
    mask_id = TOKEN_TO_ID["<mask>"]
    vocab_size = len(VOCAB)

    def collate(examples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        sequences = [example["sequence"] for example in examples]
        batch = collate_batch(sequences, add_special=True)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        result: Dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chains": [example["chain"] for example in examples],
            "liability_ln": [example["liability_ln"] for example in examples],
            "lengths": [int(example.get("length", 0) or 0) for example in examples],
            "sequences": [example["sequence"] for example in examples],
        }
        optional_keys = set().union(*(example.keys() for example in examples))
        reserved_keys = {
            "sequence",
            "chain",
            "liability_ln",
            "cdr_mask",
            "length",
        }
        for key in sorted(optional_keys - reserved_keys):
            values = [example.get(key) for example in examples]
            if any(value is not None for value in values):
                result[key] = values

        cdr_values = [example.get("cdr_mask") for example in examples]
        if any(value is not None for value in cdr_values):
            token_labels = torch.full_like(input_ids, fill_value=-100)
            for idx, mask_list in enumerate(cdr_values):
                if mask_list is None:
                    continue
                mask_tensor = torch.tensor(mask_list, dtype=torch.long)
                length = min(mask_tensor.numel(), input_ids.size(1) - 2)
                if length <= 0:
                    continue
                token_labels[idx, 1 : 1 + length] = mask_tensor[:length]
            result["token_labels"] = token_labels

        if generate_mlm:
            labels = input_ids.clone()
            special_mask = (
                (input_ids == pad_id)
                | (input_ids == bos_id)
                | (input_ids == eos_id)
            )
            candidate_mask = attention_mask & ~special_mask
            prob_matrix = torch.full(input_ids.shape, mlm_probability, dtype=torch.float32)
            masked_indices = torch.bernoulli(prob_matrix).bool() & candidate_mask
            labels[~masked_indices] = -100

            # Apply 80% <mask>, 10% random, 10% original
            replace_prob = torch.full(input_ids.shape, 0.8, dtype=torch.float32)
            indices_replaced = torch.bernoulli(replace_prob).bool() & masked_indices
            input_ids[indices_replaced] = mask_id

            random_prob = torch.full(input_ids.shape, 0.5, dtype=torch.float32)
            indices_random = torch.bernoulli(random_prob).bool() & masked_indices & ~indices_replaced
            random_tokens = torch.randint(
                vocab_size,
                input_ids.shape,
                dtype=torch.long,
            )
            input_ids[indices_random] = random_tokens[indices_random]

            result["labels"] = labels
        else:
            result["labels"] = input_ids.clone()

        return result

    return collate


__all__ = ["OASDataset", "BucketBatchSampler", "build_collate_fn"]
