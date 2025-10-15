# Data Provenance

| Dataset | Location | Source | Notes |
|---------|----------|--------|-------|
| OAS processed splits | `data/processed/oas_real_full` | Internal ETL via `scripts/process_real_data_etl.py` | Contains train/val/test parquet partitions with chain/species metadata. |
| Benchmark fixtures | `data/benchmarks/` | Derived from OAS + curated therapeutic set | Used for deterministic regression checks. |
| Demo sequences | `examples/attention_success.fa`, `examples/attention_failure.fa` | Synthetic heavy chains | Safe for public demos. |

## Regeneration

1. Download raw OAS data following the licensing terms.
2. Run `python scripts/process_real_data_etl.py --input <raw_csv> --output data/processed/oas_real_full`.
3. Log dataset version + checksum in this file for every refresh.
