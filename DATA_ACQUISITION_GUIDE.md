# AbProp Data Acquisition Guide

This guide explains how to acquire and prepare antibody sequence data for training AbProp models.

## Quick Start (Synthetic Data - Testing)

For immediate testing and development:

```bash
# Generate 10K synthetic sequences
python scripts/download_oas_data.py \
    --method synthetic \
    --num-sequences 10000 \
    --output data/raw/oas_synthetic.tsv

# Run ETL pipeline
python scripts/run_etl_standalone.py \
    --input data/raw/oas_synthetic.tsv \
    --out data/processed/oas_synthetic

# Verify output
ls -lh data/processed/oas_synthetic/
```

✅ **Status**: COMPLETED - You now have a working test dataset in `data/processed/oas_synthetic/`

---

## Production Data (Real Antibody Sequences)

### Method 1: OAS Website (Easiest for Small Datasets)

**Best for**: <100K sequences, specific filters, interactive exploration

1. **Navigate** to http://opig.stats.ox.ac.uk/webapps/oas/

2. **Filter** sequences:
   - Species: Human
   - Chain: Heavy (IGH) and/or Light (IGL/IGK)
   - Disease: Healthy, COVID-19, Cancer, etc.
   - Productive: Yes
   - Age group: Adult, Child, etc.

3. **Download** results:
   - Click "Download" button
   - Choose CSV or TSV format
   - Save to `data/raw/oas_human.tsv`

4. **Process**:
   ```bash
   python scripts/run_etl_standalone.py \
       --input data/raw/oas_human.tsv \
       --out data/processed/oas
   ```

### Method 2: AIRR Data Commons API (Programmatic)

**Best for**: Reproducible downloads, automated pipelines, specific repositories

```bash
# Fetch 50K human sequences
python scripts/fetch_airr_data.py \
    --species human \
    --sequences 50000 \
    --output data/raw/airr_human.tsv

# List available repertoires first
python scripts/fetch_airr_data.py --list-repertoires --species human

# Then process
python scripts/run_etl_standalone.py \
    --input data/raw/airr_human.tsv \
    --out data/processed/oas
```

**Note**: AIRR API may have rate limits. The script automatically falls back to synthetic data if the API is unavailable.

### Method 3: OAS Bulk Download (Large Scale)

**Best for**: >1M sequences, full dataset access, offline analysis

1. **Download** bulk files from OAS:
   - Visit: http://opig.stats.ox.ac.uk/webapps/oas/
   - Navigate to "Bulk Download" section
   - Download study files (CSV.gz format)
   - Save to `data/raw/`

2. **Combine** multiple files:
   ```bash
   python scripts/download_oas_data.py \
       --method combine \
       --input "data/raw/OAS_*.csv.gz" \
       --output data/raw/oas_combined.tsv
   ```

3. **Sample** if needed:
   ```bash
   python scripts/download_oas_data.py \
       --method sample \
       --input data/raw/oas_combined.tsv \
       --num-sequences 100000 \
       --output data/raw/oas_sample.tsv \
       --seed 42
   ```

4. **Process**:
   ```bash
   python scripts/run_etl_standalone.py \
       --input data/raw/oas_sample.tsv \
       --out data/processed/oas
   ```

---

## Data Format Requirements

The ETL pipeline expects TSV/CSV with these columns (various naming conventions supported):

| Required Column | Aliases | Example |
|----------------|---------|---------|
| **sequence** | sequence_aa, sequence_alignment_aa | EVQLVESGGGLVQPGG... |
| **chain** | chain_type, chain_id | H, L |
| **species** | species_common, organism | human, mouse |
| **germline_v** | v_gene, v_call | IGHV3-23*01 |
| **germline_j** | j_gene, j_call | IGHJ4*02 |
| **is_productive** | productive, productive_status | True, T, 1 |
| **cdr3** | junction_aa, cdr3_aa | CARDMGYYGMDV |

The ETL automatically handles:
- Column name variations
- Case normalization
- Deduplication
- Feature extraction (liabilities, CDR masks)
- Train/val/test splitting (stratified by clonotype)

---

## Output Structure

After ETL, data is stored in partitioned Parquet format:

```
data/processed/oas_synthetic/
├── species=human/
│   ├── chain=H/
│   │   ├── split=train/
│   │   │   └── *.parquet
│   │   ├── split=val/
│   │   │   └── *.parquet
│   │   └── split=test/
│   │       └── *.parquet
│   └── chain=L/
│       ├── split=train/
│       ├── split=val/
│       └── split=test/
```

Each Parquet file contains:
- `sequence`: Amino acid sequence
- `germline_v`: V gene
- `germline_j`: J gene
- `is_productive`: Productivity flag
- `cdr3`: CDR3 sequence
- `length`: Sequence length
- `liability_counts`: Raw liability motif counts (JSON)
- `liability_ln`: Length-normalized liabilities (JSON)
- `cdr_mask`: Token-level CDR labels (JSON array)

---

## Recommended Dataset Sizes

| Purpose | Sequences | Method |
|---------|-----------|--------|
| Quick test | 1K-10K | Synthetic |
| Development | 10K-50K | OAS website or AIRR API |
| Baseline model | 50K-100K | OAS download |
| Production model | 100K-1M+ | OAS bulk download |
| Research publication | 100K+ (real) | OAS bulk download |

---

## Troubleshooting

### "No module named 'abprop'"

The package isn't installed. Use the standalone script instead:
```bash
python scripts/run_etl_standalone.py --input <input> --out <output>
```

### "AIRR API returned no data"

The AIRR API may be temporarily unavailable. Options:
1. Try again later
2. Use OAS website download instead
3. Use synthetic data for testing: `--method synthetic`

### "Column 'X' not found"

The input file is missing required columns. Check:
1. File format (TSV vs CSV)
2. Column names (see aliases above)
3. File encoding (should be UTF-8)

Example inspection:
```bash
head -1 data/raw/your_file.tsv  # Check column names
```

### "Output directory already exists"

The ETL won't overwrite existing data. Either:
1. Delete the output directory: `rm -rf data/processed/oas`
2. Use a different output path: `--out data/processed/oas_v2`

### Memory issues with large files

For very large datasets (>1M sequences):
1. Sample first: use `--method sample`
2. Process in chunks (modify ETL script)
3. Use a machine with more RAM
4. Consider using cloud storage (S3, GCS) for data

---

## Validation

After ETL, validate your data:

```bash
# Check file structure
tree data/processed/oas_synthetic/ -L 4

# Count sequences per split
python -c "
import pandas as pd
splits = ['train', 'val', 'test']
for split in splits:
    df = pd.read_parquet(f'data/processed/oas_synthetic/species=human/chain=H/split={split}/')
    print(f'{split}: {len(df)} heavy chains')
"

# Inspect sample
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/oas_synthetic/species=human/chain=H/split=train/')
print(df.head())
print('\nData types:')
print(df.dtypes)
print('\nSequence length distribution:')
print(df['length'].describe())
"
```

---

## Next Steps

Once you have processed data in `data/processed/oas/`:

1. **Train a model**:
   ```bash
   python scripts/train.py \
       --config configs/train.yaml \
       --data-config configs/data.yaml \
       --model-config configs/model.yaml
   ```

2. **Update configs** to point to your dataset:
   ```yaml
   # configs/data.yaml
   processed_dir: ./data/processed/oas
   ```

3. **Run evaluation**:
   ```bash
   python scripts/eval.py \
       --checkpoint outputs/checkpoints/best.pt \
       --data-config configs/data.yaml
   ```

---

## Data Sources & Citation

### Observed Antibody Space (OAS)
```bibtex
@article{olsen2022observed,
  title={Observed Antibody Space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences},
  author={Olsen, Tobias H and Boyles, Fergus and Deane, Charlotte M},
  journal={Protein Science},
  volume={31},
  number={1},
  pages={141--146},
  year={2022}
}
```

### AIRR Data Commons
```bibtex
@article{corrie2018ireceptor,
  title={iReceptor: A platform for querying and analyzing antibody/B-cell and T-cell receptor repertoire data across federated repositories},
  author={Corrie, Brian D and others},
  journal={Immunological Reviews},
  volume={284},
  number={1},
  pages={24--41},
  year={2018}
}
```

---

## Tools Provided

| Script | Purpose |
|--------|---------|
| `scripts/download_oas_data.py` | Generate synthetic data or combine OAS files |
| `scripts/fetch_airr_data.py` | Programmatically fetch from AIRR API |
| `scripts/run_etl_standalone.py` | Run ETL pipeline |

---

## Additional Resources

- **OAS Documentation**: https://www.blopig.com/blog/2023/06/exploring-the-observed-antibody-space-oas/
- **AIRR Standards**: https://docs.airr-community.org/
- **IMGT Numbering**: http://www.imgt.org/
- **AbProp README**: See main project README for full documentation

---

**Last Updated**: 2025-10-12
**Status**: Ready for use with synthetic or real data
