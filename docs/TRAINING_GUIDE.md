# Training Guide for AbProp

This guide explains how to train AbProp models on real antibody sequence data.

## Prerequisites

### 1. Environment Setup

The AbProp package requires a properly configured Python environment with PyTorch and dependencies.

**Option A: Using Conda (Recommended for HPC)**

```bash
# Load required modules (adjust for your HPC site)
module load Miniconda3/23.10.0-1
module load CUDA/12.3.0

# Create conda environment
conda create -p $HOME/.conda/abprop python=3.10 -y
conda activate $HOME/.conda/abprop

# Install PyTorch with CUDA support
conda install pytorch=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install AbProp in development mode
cd /path/to/AbProp
pip install -e '.[dev]'
```

**Option B: Using Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.2
pip install -e '.[dev]'
```

### 2. Verify Installation

```bash
# Test import
python -c "import abprop; print(abprop.__version__)"

# Check available commands
abprop-train --help
```

### 3. Data Preparation

Ensure you have processed data ready:

```bash
# Check if real data exists
ls -lh data/processed/oas_real_full/

# If not, process your data:
python scripts/process_real_data_etl.py \
    --input data/raw/sabdab_sequences_full.csv \
    --output data/processed/oas_real_full
```

## Training Configuration

### Dataset Paths

Update `configs/data.yaml` to point to your real data:

```yaml
# configs/data_real.yaml
raw_dir: ./data/raw
interim_dir: ./data/interim
processed_dir: ./data/processed/oas_real_full  # Point to real data
splits: [train, val, test]
species: ["human", "mouse", "lama glama", "vicugna pacos"]  # Real data has 38 species
chain_types: ["H", "L"]
germline_metadata: true
liability_features:
  - nglyc
  - deamidation
  - isomerization
  - oxidation
  - free_cysteines
parquet:
  filename: oas_sequences.parquet
  partition_on: ["species", "chain", "split"]
```

### Training Parameters

Create a training config for real data (or modify `configs/train.yaml`):

```yaml
# configs/train_real.yaml
seed: 42
precision: amp  # Automatic mixed precision
gradient_clipping: 1.0
learning_rate: 1.0e-4
weight_decay: 1.0e-2
warmup_steps: 100
max_steps: 5000  # Adjust based on dataset size
batch_size: 16   # Adjust for your GPU memory
eval_interval: 250
checkpoint_interval: 500
output_dir: ./outputs/real_data_run
log_dir: ./outputs/real_data_run/logs
checkpoint_dir: ./outputs/real_data_run/checkpoints
grad_accumulation: 2  # Effective batch size = 16 * 2 = 32
lr_schedule: cosine
num_workers: 4
best_metric: eval_loss
maximize_metric: false
report_interval: 50
task_weights:
  mlm: 1.0      # Masked language modeling
  cls: 1.0      # CDR/framework classification
  reg: 1.0      # Liability regression
mlflow:
  tracking_uri: ./mlruns
  experiment_name: abprop_real_data
```

### Model Configuration

The default model config should work, but you can adjust:

```yaml
# configs/model.yaml
d_model: 256        # Model dimension
nhead: 8            # Attention heads
num_layers: 6       # Transformer layers
dim_feedforward: 1024
dropout: 0.1
activation: gelu
max_seq_len: 512
vocab_size: 30      # Character-level tokenization
pad_token_id: 0
mask_token_id: 1
```

## Training Commands

### Single GPU Training

For development and testing:

```bash
# Using command-line tool
abprop-train \
    --distributed none \
    --config-path configs/train_real.yaml \
    --data-config configs/data_real.yaml \
    --model-config configs/model.yaml

# Or using the script directly
python scripts/train.py \
    --config-path configs/train_real.yaml \
    --data-config configs/data_real.yaml \
    --model-config configs/model.yaml
```

### Multi-GPU Training (Single Node)

Using PyTorch DistributedDataParallel:

```bash
# With torchrun (recommended)
torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/train.py \
    --distributed ddp \
    --config-path configs/train_real.yaml \
    --data-config configs/data_real.yaml \
    --model-config configs/model.yaml

# Or with abprop-launch helper
abprop-launch \
    --gpus-per-node 4 \
    --config configs/train_real.yaml
```

### Multi-Node Training (Slurm)

Create a batch script:

```bash
#!/bin/bash
#SBATCH --job-name=abprop-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules
module load Miniconda3/23.10.0-1
module load CUDA/12.3.0
module load NCCL

# Activate environment
conda activate $HOME/.conda/abprop

# NCCL settings
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

# Training command
srun python scripts/train.py \
    --distributed ddp \
    --config-path configs/train_real.yaml \
    --data-config configs/data_real.yaml \
    --model-config configs/model.yaml
```

Submit with:
```bash
sbatch slurm/train_real_data.sbatch
```

## Dataset Statistics

**Current Real Dataset** (`data/processed/oas_real_full/`):
- Total: 1,502 sequences
- Train: 1,209 (80.5%)
- Val: 144 (9.6%)
- Test: 149 (9.9%)
- Species: 38 unique (human, mouse, llama, alpaca, etc.)
- Chains: H=1,086 (72%), L=416 (28%)
- Length: 55-499 aa (mean: 212)

**Training Recommendations:**
- **Batch size**: Start with 16-32, adjust for GPU memory
- **Learning rate**: 1e-4 is a good starting point
- **Steps**: ~5,000 steps for 1.5K dataset (3-4 epochs with batch size 32)
- **Evaluation**: Every 250-500 steps
- **Expected training time**:
  - Single GPU (V100): ~2-3 hours
  - 4 GPUs (V100): ~30-45 minutes

## Monitoring Training

### MLflow UI

AbProp logs all metrics to MLflow:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000

# Access at: http://localhost:5000
```

### Log Files

Training logs are saved to:
- `outputs/real_data_run/logs/training.log`
- `outputs/real_data_run/logs/metrics.csv`

Monitor in real-time:
```bash
tail -f outputs/real_data_run/logs/training.log
```

### Checkpoints

Checkpoints are saved to `outputs/real_data_run/checkpoints/`:
- `last.pt` - Most recent checkpoint
- `best.pt` - Best checkpoint by validation loss

## Evaluating Models

After training, evaluate on the test set:

```bash
# Evaluate best checkpoint
abprop-eval \
    --checkpoint outputs/real_data_run/checkpoints/best.pt \
    --data-config configs/data_real.yaml \
    --model-config configs/model.yaml \
    --split test \
    --output outputs/real_data_run/eval_results.json
```

## Comparing Datasets

To compare performance on real vs synthetic data:

```bash
# Train on synthetic data
python scripts/train.py \
    --config-path configs/train.yaml \
    --data-config configs/data.yaml \
    --output-dir outputs/synthetic_run

# Train on real data
python scripts/train.py \
    --config-path configs/train_real.yaml \
    --data-config configs/data_real.yaml \
    --output-dir outputs/real_run

# Compare metrics in MLflow
mlflow ui --backend-store-uri ./mlruns
```

## Troubleshooting

### GPU Memory Issues

If you run out of GPU memory:

1. **Reduce batch size**: Set `batch_size: 8` or `batch_size: 4`
2. **Increase gradient accumulation**: Set `grad_accumulation: 4`
3. **Reduce model size**: Set `d_model: 128`, `num_layers: 4`
4. **Enable gradient checkpointing** (if implemented)

### Slow Training

1. **Increase num_workers**: Set `num_workers: 4` or `num_workers: 8`
2. **Use mixed precision**: Ensure `precision: amp` is set
3. **Check data loading**: Monitor `DataLoader` time in logs
4. **Use multiple GPUs**: Switch to DDP mode

### NCCL Issues (Multi-GPU)

```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check GPU visibility
nvidia-smi

# Verify NCCL can communicate
python -c "import torch; print(torch.cuda.nccl.version())"
```

### Import Errors

```bash
# Verify installation
pip show abprop

# Reinstall if needed
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## Advanced Topics

### Learning Rate Scheduling

AbProp supports multiple LR schedules:

```yaml
lr_schedule: cosine     # Cosine annealing
# lr_schedule: linear   # Linear decay
# lr_schedule: constant # Constant LR
```

### Task Weighting

Adjust loss weights for multi-task learning:

```yaml
task_weights:
  mlm: 1.0  # Masked language modeling (reconstruction)
  cls: 2.0  # CDR classification (if you care more about this)
  reg: 0.5  # Liability prediction (if less important)
```

### Data Filtering

Filter by species or chain type:

```python
# In your training script or config
dataset = OASDataset(
    parquet_dir="data/processed/oas_real_full",
    split="train",
    species_filter=["human", "mouse"],  # Only human and mouse
    chain_filter=["H"]                   # Only heavy chains
)
```

### Transfer Learning

Start from a pretrained checkpoint:

```bash
python scripts/train.py \
    --resume-from outputs/pretrained_model/checkpoints/best.pt \
    --config-path configs/finetune_real.yaml
```

## Performance Expectations

Based on the current dataset size (1,502 sequences):

**Training Metrics** (after 5,000 steps):
- MLM Perplexity: 1.5-3.0 (lower is better)
- CDR Classification Accuracy: 75-85%
- Liability Regression MAE: 0.01-0.05

**Validation Metrics**:
- Should be close to training metrics (within 10-20%)
- Watch for overfitting if validation loss increases

**Test Metrics**:
- Final evaluation on held-out test set
- Should match validation performance

## Next Steps

1. **Expand Dataset**: Fetch more sequences to improve generalization
   ```bash
   python scripts/fetch_real_antibody_data.py \
       --source sabdab \
       --max-entries 15000 \
       --output data/raw/sabdab_15k.csv
   ```

2. **Install ANARCI**: For better germline and CDR annotations
   ```bash
   # See: http://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred/anarci/
   ```

3. **Benchmark on Therapeutic Dataset**:
   ```bash
   abprop-eval \
       --checkpoint outputs/real_data_run/checkpoints/best.pt \
       --benchmark therapeutic \
       --config configs/therapeutic_bench.yaml
   ```

4. **Deploy Model**: See REST API section in main README

## References

- **Dataset**: See [docs/REAL_DATA_SUMMARY.md](REAL_DATA_SUMMARY.md)
- **Data Acquisition**: See [docs/DATA_ACQUISITION_GUIDE.md](DATA_ACQUISITION_GUIDE.md)
- **Evaluation Prompts**: See [docs/EVALUATION_PROMPTS.md](EVALUATION_PROMPTS.md)
- **Main README**: See [../README.md](../README.md)

---

**Last Updated**: October 13, 2025
**Dataset**: 1,502 real sequences from PDB structures
**Status**: Ready for training once environment is configured
