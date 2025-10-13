# AbProp Documentation

This directory contains supplementary documentation for the AbProp project.

## Documentation Files

### 📊 [REAL_DATA_SUMMARY.md](REAL_DATA_SUMMARY.md)
**Complete guide to real antibody sequence data**

Summary of all real antibody data that has been fetched and processed:
- Data sources (SAbDab, PDB)
- Dataset statistics (1,502 real sequences)
- Processing pipeline details
- Quality metrics and comparisons
- Usage examples
- Next steps for expansion

**Key Stats:**
- 2,140 sequences fetched from PDB
- 1,502 sequences after ETL processing
- 38 unique species (human, mouse, llama, etc.)
- Train: 1,209 | Val: 144 | Test: 149

---

### 📥 [DATA_ACQUISITION_GUIDE.md](DATA_ACQUISITION_GUIDE.md)
**Step-by-step guide for downloading antibody data**

Comprehensive guide for acquiring antibody sequence data:
- Quick start instructions
- Three methods: OAS website, AIRR API, bulk download
- Troubleshooting common issues
- Data format specifications
- ETL pipeline usage

**Use this when:** You need to download additional antibody sequences from public databases.

---

### 🎯 [EVALUATION_PROMPTS.md](EVALUATION_PROMPTS.md)
**Detailed prompts for building evaluation infrastructure**

A structured roadmap with 25+ detailed prompts organized into 7 phases:

**Phase 1: Data Acquisition & Benchmarking**
- Download real OAS dataset
- Create therapeutic antibody benchmark
- Build CDR identification gold standard

**Phase 2: Evaluation Infrastructure**
- Implement evaluation metrics suite
- Create task-specific evaluators
- Build benchmark runners

**Phase 3: Visualization & Analysis**
- Create evaluation dashboards
- Implement attention visualization
- Add interpretability tools

**Phase 4: Baseline Comparisons**
- Implement ESM baseline
- Add rule-based baselines
- Compare with published models

**Phase 5: Model Analysis**
- Per-species analysis
- Length-dependent analysis
- Error analysis tools

**Phase 6: Documentation & Reporting**
- Model cards
- Evaluation reports
- Interactive demos

**Phase 7: Advanced Evaluation**
- Antibody-specific benchmarks
- Cross-species evaluation
- Few-shot learning evaluation

**Use this when:** Planning next development tasks or implementing evaluation features.

---

## Quick Links

### Data Locations
- **Processed Real Data (Recommended)**: `data/processed/oas_real_full/`
- **Raw Data**: `data/raw/sabdab_sequences_full.csv`
- **Metadata**: `data/raw/sabdab_summary.tsv`

### Key Scripts
- **Fetch Data**: `scripts/fetch_real_antibody_data.py`
- **Process Data**: `scripts/process_real_data_etl.py`
- **Therapeutic Dataset**: `scripts/curate_therapeutic_dataset.py`
- **CDR Gold Standard**: `scripts/build_cdr_gold_standard.py`

### Dataset Comparison

| Dataset | Size | Source | Status |
|---------|------|--------|--------|
| `oas_synthetic` | 10,000 | Random generation | ✓ Complete (old) |
| `oas_real` | 358 | PDB structures | ✓ Complete (v1) |
| `oas_real_full` | 1,502 | PDB structures | ✓ Complete (v2) ⭐ |
| `therapeutic_benchmark` | 55 | Synthetic therapeutic | ✓ Complete |
| `cdr_gold_standard` | 100 | Synthetic with schemes | ✓ Complete |

---

## Getting Started with Real Data

### Load the Dataset
```python
from abprop.data import OASDataset

# Load training data
train_dataset = OASDataset(
    parquet_dir="data/processed/oas_real_full",
    split="train"
)

print(f"Training sequences: {len(train_dataset)}")

# Access a sample
sample = train_dataset[0]
print(f"Sequence: {sample['sequence'][:50]}...")
print(f"Chain: {sample['chain']}")
print(f"Species: {sample['species']}")
```

### Fetch More Data
```bash
# Fetch additional sequences from SAbDab/PDB
python scripts/fetch_real_antibody_data.py \
    --source sabdab \
    --output data/raw/sabdab_new_batch.csv \
    --max-entries 10000

# Process through ETL
python scripts/process_real_data_etl.py \
    --input data/raw/sabdab_new_batch.csv \
    --output data/processed/oas_real_v3
```

---

## Contributing

When adding new documentation:
1. Create the `.md` file in the `docs/` directory
2. Add an entry to this README with a brief description
3. Include relevant links and usage examples
4. Update the Quick Links section if needed

---

## Document Status

| Document | Last Updated | Status |
|----------|--------------|--------|
| REAL_DATA_SUMMARY.md | Oct 13, 2025 | ✓ Complete |
| DATA_ACQUISITION_GUIDE.md | Oct 12, 2025 | ✓ Complete |
| EVALUATION_PROMPTS.md | Oct 12, 2025 | ✓ Complete |

---

**For the main project README, see**: [`../README.md`](../README.md)
