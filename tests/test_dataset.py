from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from abprop.data.dataset import BucketBatchSampler, OASDataset, build_collate_fn
from abprop.tokenizers import TOKEN_TO_ID


def _make_parquet_dataset(tmp_path: Path) -> Path:
    records = []
    lengths = [5, 8, 15, 22, 35, 48]
    for idx, length in enumerate(lengths):
        sequence = "A" * length
        records.append(
            {
                "sequence": sequence,
                "chain": "H" if idx % 2 == 0 else "L",
                "liability_ln": json.dumps({"nglyc": 0.0, "oxidation": 0.1}),
                "length": length,
                "split": "train",
            }
        )
    df = pd.DataFrame.from_records(records)
    path = tmp_path / "oas.parquet"
    df.to_parquet(path, index=False)
    return path


def test_bucket_batch_sampler_groups_similar_lengths(tmp_path: Path):
    parquet_path = _make_parquet_dataset(tmp_path)
    dataset = OASDataset(parquet_path, split="train")
    sampler = BucketBatchSampler(
        dataset.lengths,
        batch_size=2,
        bins=[8, 16, 32, 64],
        shuffle=False,
    )

    batches = list(iter(sampler))
    assert sum(len(batch) for batch in batches) == len(dataset)

    for batch in batches:
        lengths = [dataset.lengths[idx] for idx in batch]
        boundary = next(b for b in sampler.bin_boundaries if max(lengths) <= b)
        assert all(length <= boundary for length in lengths)


def test_collate_fn_shapes_and_mlm(tmp_path: Path):
    parquet_path = _make_parquet_dataset(tmp_path)
    dataset = OASDataset(parquet_path, split="train")

    examples = [dataset[idx] for idx in range(3)]
    collate = build_collate_fn(generate_mlm=True, mlm_probability=0.2)
    batch = collate(examples)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.shape == attention_mask.shape == labels.shape
    assert input_ids.shape[0] == len(examples)

    pad_id = TOKEN_TO_ID["<pad>"]
    assert torch.equal(attention_mask, input_ids != pad_id)

    # Ensure at least one token selected for MLM.
    masked_positions = labels != -100
    assert masked_positions.any()

    # Chains and liability metadata preserved.
    assert len(batch["chains"]) == len(examples)
    assert len(batch["liability_ln"]) == len(examples)
