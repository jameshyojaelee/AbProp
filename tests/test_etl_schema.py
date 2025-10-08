from __future__ import annotations

import csv
from pathlib import Path

from abprop.data.etl import ETLConfig, run_etl
from abprop.data.schema import validate_parquet_dataset


def _write_synthetic_oas(path: Path) -> None:
    rows = [
        {
            "sequence": "ACDEFGHIK",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAAA",
        },
        {
            "sequence": "ACDEFGHIL",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAAA",
        },
        {
            "sequence": "NNNSTDGHI",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAB",
        },
        {
            "sequence": "ACDGNGSTY",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAC",
        },
        {
            "sequence": "ACDNMGSTY",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAD",
        },
        {
            "sequence": "ACDCGASTY",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAE",
        },
        {
            "sequence": "ACDDDGSTY",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": True,
            "cdr3": "CARAAF",
        },
        {
            "sequence": "ACDDDGSTY",
            "chain": "H",
            "species": "Human",
            "germline_v": "IGHV1-2",
            "germline_j": "IGHJ4",
            "is_productive": False,
            "cdr3": "CARAAF",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "chain",
                "species",
                "germline_v",
                "germline_j",
                "is_productive",
                "cdr3",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_oas_etl_creates_valid_dataset(tmp_path: Path) -> None:
    raw_path = tmp_path / "oas.tsv"
    _write_synthetic_oas(raw_path)
    out_dir = tmp_path / "processed" / "oas"

    config = ETLConfig(
        input_path=raw_path,
        output_dir=out_dir,
        splits=(0.6, 0.2, 0.2),
        seed=123,
    )

    df = run_etl(config)
    assert len(df) == 7
    assert (df["split"] == "train").sum() >= 1
    assert (df["split"] == "val").sum() >= 1
    assert (df["split"] == "test").sum() >= 1

    validated = validate_parquet_dataset(out_dir)
    assert set(validated.columns).issuperset(
        {
            "sequence",
            "chain",
            "species",
            "germline_v",
            "germline_j",
            "is_productive",
            "cdr3",
            "length",
            "liability_counts",
            "liability_ln",
            "split",
        }
    )

    same_clone_splits = validated.loc[validated["cdr3"] == "CARAAAA", "split"].unique()
    assert len(same_clone_splits) == 1

    first_row = validated.iloc[0]
    counts = first_row["liability_counts"]
    norm = first_row["liability_ln"]
    length = first_row["length"]
    for key, value in counts.items():
        if length:
            assert abs(norm[key] - value / length) < 1e-6
        else:
            assert norm[key] == 0.0
