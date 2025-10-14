from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from abprop.commands.train import build_oas_dataloaders
from abprop.data.dataset import BucketBatchSampler, OASDataset, build_collate_fn
from abprop.models import TransformerConfig
from abprop.tokenizers import TOKEN_TO_ID


def _make_parquet_dataset(tmp_path: Path) -> Path:
    records = []
    lengths = [5, 8, 15, 22, 35, 48]
    for idx, length in enumerate(lengths):
        sequence = "A" * length
        species = "human" if idx % 2 == 0 else "mouse"
        germline_v = f"IGHV{idx+1}"
        germline_j = f"IGHJ{idx+1}"
        cdr_mask = [0] * length
        records.append(
            {
                "sequence": sequence,
                "chain": "H" if idx % 2 == 0 else "L",
                "liability_ln": json.dumps({"nglyc": 0.0, "oxidation": 0.1}),
                "length": length,
                "split": "train",
                "species": species,
                "germline_v": germline_v,
                "germline_j": germline_j,
                "cdr_mask": json.dumps(cdr_mask),
                "custom_metric": float(idx),
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

    assert examples[0]["species"] == "human"
    assert examples[1]["germline_v"] == "IGHV2"
    assert "custom_metric" in examples[0]

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
    assert batch["lengths"] == [5, 8, 15]
    assert batch["species"] == ["human", "mouse", "human"]
    assert batch["germline_v"] == ["IGHV1", "IGHV2", "IGHV3"]
    assert batch["germline_j"] == ["IGHJ1", "IGHJ2", "IGHJ3"]
    assert batch["custom_metric"] == [0.0, 1.0, 2.0]


def test_build_oas_dataloaders_uses_configured_dataset(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    dataset_dir = processed_dir / "oas_real_full"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "sequence": "AAAAA",
            "chain": "H",
            "liability_ln": json.dumps({"nglyc": 0.0, "oxidation": 0.0}),
            "length": 5,
            "split": "train",
            "species": "human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "cdr_mask": json.dumps([0, 0, 0, 0, 0]),
        },
        {
            "sequence": "CCCCCC",
            "chain": "L",
            "liability_ln": json.dumps({"nglyc": 0.0, "oxidation": 0.0}),
            "length": 6,
            "split": "train",
            "species": "mouse",
            "germline_v": "IGKV1-5",
            "germline_j": "IGKJ1",
            "cdr_mask": json.dumps([0, 0, 0, 0, 0, 0]),
        },
        {
            "sequence": "GGGGGG",
            "chain": "H",
            "liability_ln": json.dumps({"nglyc": 0.0, "oxidation": 0.0}),
            "length": 6,
            "split": "val",
            "species": "human",
            "germline_v": "IGHV3-7",
            "germline_j": "IGHJ6",
            "cdr_mask": json.dumps([0, 0, 0, 0, 0, 0]),
        },
    ]
    parquet_path = dataset_dir / "oas_sequences.parquet"
    pd.DataFrame.from_records(records).to_parquet(parquet_path, index=False)

    data_cfg = {
        "processed_dir": str(processed_dir),
        "parquet": {
            "output_dir": "oas_real_full",
            "filename": "oas_sequences.parquet",
        },
        "length_bins": [8, 16, 32],
    }

    train_loader, eval_loader = build_oas_dataloaders(
        data_cfg,
        batch_size=2,
        model_config=TransformerConfig(),
        max_tokens=None,
        shuffle=True,
        num_workers=0,
        distributed=False,
        rank=0,
        world_size=1,
    )

    assert len(train_loader.dataset) == 2
    assert eval_loader is not None
    assert len(eval_loader.dataset) == 1

    train_batch = next(iter(train_loader))
    assert "species" in train_batch
    assert set(train_batch["species"]) <= {"human", "mouse"}
    assert set(train_batch["germline_v"]) <= {"IGHV1-2", "IGKV1-5"}
    assert train_batch["lengths"]


def test_build_oas_dataloaders_raises_for_missing_dataset(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    data_cfg = {
        "processed_dir": str(processed_dir),
        "parquet": {
            "output_dir": "missing_dataset",
            "filename": "does_not_exist.parquet",
        },
    }

    with pytest.raises(FileNotFoundError):
        build_oas_dataloaders(
            data_cfg,
            batch_size=2,
            model_config=TransformerConfig(),
            max_tokens=None,
            shuffle=True,
            num_workers=0,
            distributed=False,
            rank=0,
            world_size=1,
        )
