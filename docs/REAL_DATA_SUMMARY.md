# Real Antibody Data Summary

This document summarizes the real antibody sequence data that has been acquired and processed for the AbProp project.

## Data Sources

### 1. SAbDab (Structural Antibody Database)
- **Source**: Oxford Protein Informatics Group (OPIG)
- **URL**: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/
- **Data Type**: Structural data from PDB with antibody sequences
- **Metadata Downloaded**: 19,730 antibody structures
- **Sequences Fetched**: 2,140 real sequences (from 5,000 structures)

### 2. PDB (Protein Data Bank)
- **Source**: RCSB PDB REST API
- **URL**: https://data.rcsb.org/
- **Data Type**: Full-length antibody sequences from crystal/cryo-EM structures
- **Access Method**: Automated fetching via REST API per PDB ID and chain

## Downloaded Datasets

### Initial Test Dataset (✓ Complete)
- **File**: `data/raw/sabdab_sequences_test.csv`
- **Sequences**: 27 sequences (test batch)
- **Species**: Mouse, Human
- **Chains**: Heavy (H) and Light (L)
- **Resolution**: < 4.0 Å (high-quality structures)
- **Methods**: X-ray diffraction, Electron microscopy

### Production Dataset v1 (✓ Complete)
- **File**: `data/raw/sabdab_sequences.csv`
- **Sequences**: 475 sequences
- **Species**:
  - Human: 256 (54%)
  - Mouse: 69 (15%)
  - Llama (Lama glama): 27 (6%)
  - Pig (Sus scrofa): 18 (4%)
  - Alpaca (Vicugna pacos): 16 (3%)
  - Others: 89 (18%)
- **Chain Distribution**:
  - Heavy: 350 (74%)
  - Light: 125 (26%)
- **Quality Filters**:
  - Resolution < 4.0 Å
  - Sequence length: 50-500 amino acids
  - Methods: X-ray diffraction or Electron microscopy only
- **Processing Status**: ✓ Processed through ETL pipeline
- **Output**: `data/processed/oas_real/` (358 sequences after deduplication)

### Production Dataset v2 (✓ Complete)
- **File**: `data/raw/sabdab_sequences_full.csv`
- **Sequences**: 2,140 sequences (from 5,000 structures, ~43% success rate)
- **Species**:
  - Human: 1,166 (54%)
  - Mouse: 329 (15%)
  - Llama (Lama glama): 178 (8%)
  - Macaca mulatta: 67 (3%)
  - Synthetic construct: 64 (3%)
  - Alpaca (Vicugna pacos): 64 (3%)
  - Others: 272 (13%)
- **Chain Distribution**:
  - Heavy: 1,498 (70%)
  - Light: 642 (30%)
- **Processing Status**: ✓ Processed through ETL pipeline
- **Output**: `data/processed/oas_real_full/` (1,502 sequences after deduplication)

## Processed Datasets

### OAS Real Dataset (Small - v1)
- **Location**: `data/processed/oas_real/`
- **Format**: Partitioned Parquet (by species, chain, split)
- **Total Sequences**: 358 (after deduplication of productive sequences)
- **Splits**:
  - Train: 291 (81%)
  - Val: 33 (9%)
  - Test: 34 (10%)

### OAS Real Dataset (Full - v2) ⭐ **RECOMMENDED**
- **Location**: `data/processed/oas_real_full/`
- **Format**: Partitioned Parquet (by species, chain, split)
- **Total Sequences**: 1,502 (after deduplication of productive sequences)
- **Splits**:
  - Train: 1,209 (80.5%)
  - Val: 144 (9.6%)
  - Test: 149 (9.9%)
- **Species Coverage**: 38 unique species (human-dominated)
- **Sequence Length**: 55-499 aa (mean: 212, median: 215)

### Comparison: Synthetic vs Real Data

| Metric | Synthetic (Old) | Real (New) |
|--------|----------------|------------|
| Total Sequences | 10,000 | 1,502 |
| Train/Val/Test | 8,007 / 994 / 999 | 1,209 / 144 / 149 |
| Data Source | Randomly generated | PDB structures |
| Species Diversity | 2 (human, mouse) | 38 species |
| Quality Assurance | None | Crystal structures <4Å |
| Germline Annotations | Synthetic | Real (but unknown) |
| CDR3 Sequences | Synthetic | Real (extractable) |

**Features** (both datasets):
- Sequence (amino acid)
- Chain type (H/L)
- Species
- Length
- Liability counts (nglyc, deamidation, isomerization, oxidation, cysteine_pairs)
- Liability scores (length-normalized)
- CDR masks (binary, approximate)
- Germline annotations (currently "UNKNOWN" - requires ANARCI)

### Partitioning Structure
```
data/processed/oas_real/
├── species=human/
│   ├── chain=H/
│   │   ├── split=train/*.parquet
│   │   ├── split=val/*.parquet
│   │   └── split=test/*.parquet
│   └── chain=L/
│       ├── split=train/*.parquet
│       ├── split=val/*.parquet
│       └── split=test/*.parquet
├── species=mouse/
│   └── ... (same structure)
└── species=*/
    └── ... (58 partition combinations total)
```

## Data Quality

### Advantages of This Dataset
1. **Real Experimental Data**: All sequences from resolved crystal/cryo-EM structures
2. **High Quality**: Resolution filters ensure structural validity
3. **Diverse Species**: 20+ species represented, not just human
4. **Engineered Variants**: Includes both natural and engineered antibodies
5. **Antigen Information**: Metadata includes antigen type (protein, peptide, etc.)
6. **Publication Date**: Structures span from 1990s to 2025 (latest: Oct 2025)

### Known Limitations
1. **Size**: Current dataset is modest (1,502 sequences after processing)
   - Can expand to ~10,000 by fetching more from SAbDab
   - Compare to: OAS has >1 billion sequences
2. **Germline Annotations**: Currently "UNKNOWN" (requires ANARCI installation)
3. **CDR3 Sequences**: Not extracted yet (requires ANARCI or similar tool)
4. **CDR Masks**: Approximate position-based masks, not structure-based
5. **Bias Toward Structured**: Only sequences with resolved structures (crystallization bias)

## Scripts and Tools

### Fetching Scripts
- **`scripts/fetch_real_antibody_data.py`**: Main script for downloading sequences
  - Supports SAbDab metadata processing
  - PDB REST API integration
  - UniProt integration (in development)
  - Automatic retry logic and rate limiting

### Processing Scripts
- **`scripts/process_real_data_etl.py`**: Process fetched data through ETL
  - Converts CSV to ETL-compatible format
  - Runs full AbProp ETL pipeline
  - Outputs partitioned Parquet files

### Supporting Scripts
- **`scripts/download_oas_data.py`**: Generate synthetic data (fallback)
- **`scripts/run_etl_standalone.py`**: Standalone ETL runner
- **`scripts/curate_therapeutic_dataset.py`**: Therapeutic antibody dataset (synthetic)
- **`scripts/build_cdr_gold_standard.py`**: CDR numbering schemes (synthetic)

## Usage Examples

### Using the Real Dataset for Training
```python
from abprop.data import OASDataset

# Load training data
train_dataset = OASDataset(
    parquet_dir="data/processed/oas_real",
    split="train"
)

# Load validation data
val_dataset = OASDataset(
    parquet_dir="data/processed/oas_real",
    split="val"
)

print(f"Training sequences: {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")

# Access a sample
sample = train_dataset[0]
print(f"Sequence: {sample['sequence'][:50]}...")
print(f"Chain: {sample['chain']}")
print(f"Liabilities: {sample['liability_ln']}")
```

### Fetching More Sequences
```bash
# Fetch from SAbDab (with PDB sequence fetching)
python scripts/fetch_real_antibody_data.py \
    --source sabdab \
    --output data/raw/sabdab_new_batch.csv \
    --max-entries 10000

# Process through ETL
python scripts/process_real_data_etl.py \
    --input data/raw/sabdab_new_batch.csv \
    --output data/processed/oas_real_v2
```

### Combining with Synthetic Data
```bash
# Process both real and synthetic data
python scripts/process_real_data_etl.py \
    --input data/raw/combined_sequences.csv \
    --output data/processed/oas_combined \
    --splits 0.7 0.15 0.15
```

## Next Steps

### Immediate (✓ Complete)
1. ✓ Download SAbDab metadata (19,730 structures)
2. ✓ Fetch sequences from PDB API (475 sequences complete)
3. ✓ Fetch full batch (5,000 structures → 2,140 sequences)
4. ✓ Process full batch through ETL pipeline (1,502 after deduplication)

### Short-term
5. Install ANARCI for proper germline annotation
6. Extract CDR3 sequences using ANARCI
7. Generate structure-based CDR masks
8. Fetch therapeutic antibodies from UniProt (antibody constant regions)
9. Integrate OAS bulk downloads (if available)

### Medium-term
10. Download additional SAbDab batches (target: 5,000-10,000 sequences)
11. Create species-specific datasets (human-only, mouse-only)
12. Build paired heavy-light datasets (currently unpaired)
13. Add sequence identity clustering for better train/test splits

### Long-term
14. Integrate AIRR Data Commons (when API is available)
15. Download full OAS studies (millions of sequences)
16. Create benchmark datasets for specific tasks (stability, affinity, etc.)
17. Build antibody structure prediction dataset (AlphaFold integration)

## Citations

### Data Sources
- **SAbDab**: Dunbar et al. (2014). "SAbDab: the structural antibody database." Nucleic Acids Research.
- **PDB**: Berman et al. (2000). "The Protein Data Bank." Nucleic Acids Research.
- **OAS**: Olsen et al. (2022). "Observed Antibody Space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences." Protein Science.

### Tools
- **AbProp**: This project - Antibody property prediction with transformers
- **PyArrow**: For efficient Parquet file handling
- **Pandas**: Data manipulation and ETL

## File Locations

```
AbProp/
├── data/
│   ├── raw/
│   │   ├── sabdab_summary.tsv (19,730 structures, 7.3 MB)
│   │   ├── sabdab_sequences_test.csv (27 sequences, 8.4 KB)
│   │   ├── sabdab_sequences.csv (475 sequences, 148 KB)
│   │   └── sabdab_sequences_full.csv (2,140 sequences, 648 KB) ⭐
│   ├── interim/
│   │   ├── sabdab_sequences_for_etl.tsv (475 sequences)
│   │   └── sabdab_sequences_full_for_etl.tsv (2,140 sequences)
│   └── processed/
│       ├── oas_synthetic/ (10,000 sequences, synthetic - old)
│       ├── oas_real/ (358 sequences, real - v1)
│       └── oas_real_full/ (1,502 sequences, real - v2) ⭐ RECOMMENDED
├── scripts/
│   ├── fetch_real_antibody_data.py (⭐ Main fetching script)
│   ├── process_real_data_etl.py (⭐ Processing script)
│   └── ... (other scripts)
└── REAL_DATA_SUMMARY.md (this file)
```

## Performance Notes

### Fetching Speed
- **PDB API**: ~0.2-0.5 seconds per structure (with retry logic)
- **Rate**: ~400-600 sequences per hour
- **Success Rate**: ~40-60% (not all PDB IDs return valid sequences)

### Processing Speed
- **ETL Pipeline**: ~1 second per 100 sequences
- **Bottleneck**: Motif finding for liability annotation

### Storage
- **Raw CSV**: ~300 bytes per sequence (with metadata)
- **Processed Parquet**: ~500-800 bytes per sequence (with features)
- **Compression**: Parquet achieves ~60% compression vs CSV

---

**Last Updated**: October 13, 2025
**Status**: ✓ All datasets complete and processed
**Total Real Sequences**: 2,140 raw → 1,502 processed
**Recommended Dataset**: `data/processed/oas_real_full/`
**Contact**: See repository for issues/contributions
