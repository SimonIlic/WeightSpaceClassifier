# Snellius Training Scripts

SLURM job scripts to train 1000 CNNs per dataset on [Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius), the Dutch national supercomputer.

## Prerequisites

1. **Clone the repository to Snellius:**
   ```bash
   cd $HOME
   git clone <repo-url> WeightSpaceClassifier
   cd WeightSpaceClassifier
   ```

2. **Set up Python environment:**
   ```bash
   module load 2024
   module load Python/3.11.5-GCCcore-13.2.0
   module load CUDA/12.1.1

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Verify GPU access:**
   ```bash
   srun --partition=gpu --gpus=1 --time=00:05:00 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

## Quick Start

```bash
cd experiments/controlled_hyperparams/snellius

# Submit for all datasets (mnist, fashion_mnist, cifar10, svhn_cropped)
./submit_all.sh

# Submit for a single dataset
./submit_all.sh --dataset mnist

# Submit with cleanup (delete .keras checkpoints after aggregation)
./submit_all.sh --delete-checkpoints --copy-to-home
```

## Files

| File | Description |
|------|-------------|
| `train_job.sh` | SLURM job array: trains one CNN per array task (seed = task ID) |
| `aggregate_job.sh` | SLURM job: aggregates results into `weights.npy` + `metrics.csv` |
| `submit_all.sh` | Convenience script: submits training + aggregation with dependencies |

## Job Configuration

### Training Job (`train_job.sh`)

| Resource | Value | Rationale |
|----------|-------|-----------|
| Partition | `gpu` (A100) | GPU acceleration for TensorFlow |
| GPUs | 1 | Single GPU sufficient for 4970-param CNN |
| Time | 15 min | Generous for 86 epochs on MNIST |
| Memory | 4 GB | ~2GB TF/CUDA + ~200MB dataset + buffer |
| CPUs | 2 | For tf.data pipeline parallelism |
| Array | 1-1000%50 | 1000 seeds, max 50 concurrent jobs |

### Aggregation Job (`aggregate_job.sh`)

| Resource | Value | Rationale |
|----------|-------|-----------|
| Partition | `gpu` | Needs GPU for model loading/evaluation |
| GPUs | 1 | For per-class accuracy evaluation |
| Time | 1 hour | Processing 1000 checkpoints |
| Memory | 16 GB | Loading test data + models in memory |
| CPUs | 4 | For parallel data loading |

## Usage Examples

### Submit for a single dataset

```bash
sbatch --export=DATASET=mnist train_job.sh
```

### Submit a subset of seeds

```bash
# Seeds 1-100 only
sbatch --export=DATASET=mnist --array=1-100%50 train_job.sh

# Seeds 501-1000 (continuing a previous run)
sbatch --export=DATASET=mnist --array=501-1000%50 train_job.sh
```

### Submit aggregation after training completes

```bash
# Get the training job ID
TRAIN_JOB_ID=$(sbatch --parsable --export=DATASET=mnist train_job.sh)

# Submit aggregation with dependency
sbatch --dependency=afterok:$TRAIN_JOB_ID --export=DATASET=mnist aggregate_job.sh
```

### Use custom output directory

```bash
sbatch --export=DATASET=mnist,OUTPUT_BASE=/scratch-shared/$USER/custom_output train_job.sh
```

## Monitoring

```bash
# View your jobs
squeue -u $USER

# Detailed job info
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# View array task status
sacct -j <array_job_id> --format=JobID%20,State,ExitCode,Elapsed

# View logs
tail -f logs/train_cnn_<job_id>_<task_id>.out
```

## Output Structure

Results are stored in `/scratch-shared/$USER/controlled_hyperparams/`:

```
/scratch-shared/$USER/controlled_hyperparams/
├── mnist/
│   ├── seed_1/
│   │   ├── permanent_ckpt-86.keras    # Final trained model
│   │   └── results.json               # Training metrics
│   ├── seed_2/
│   │   └── ...
│   ├── ...
│   ├── weights.npy                    # Aggregated weights [1000, 4970]
│   └── metrics.csv                    # Per-model metrics
├── fashion_mnist/
│   └── ...
└── ...
```

## Troubleshooting

### Job fails with "module not found"

Ensure the `.venv` was created with the same Python module:

```bash
module load 2024
module load Python/3.11.5-GCCcore-13.2.0
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Jobs stuck in pending

Check queue status and your fair share:

```bash
squeue -p gpu
sshare -u $USER
```

Reduce concurrent jobs if needed:

```bash
./submit_all.sh --array 1-1000%20  # Max 20 concurrent instead of 50
```

### Out of memory errors

The 4GB default should be sufficient. If you see OOM errors:

```bash
# Edit train_job.sh: change --mem=4G to --mem=8G
```

### Training already completed

Jobs automatically skip if `results.json` exists. To re-run:

```bash
# On scratch (be careful!)
rm /scratch-shared/$USER/controlled_hyperparams/mnist/seed_*/results.json
```

## Copying Results

After aggregation, copy results back to your home directory or local machine:

```bash
# On Snellius: copy to home
cp /scratch-shared/$USER/controlled_hyperparams/mnist/weights.npy ~/WeightSpaceClassifier/model_zoo/mnist_controlled/
cp /scratch-shared/$USER/controlled_hyperparams/mnist/metrics.csv ~/WeightSpaceClassifier/model_zoo/mnist_controlled/

# From local machine: download via scp
scp snellius:/scratch-shared/$USER/controlled_hyperparams/mnist/weights.npy ./
```

Or use the `--copy-to-home` flag in `submit_all.sh`.

## Resource Estimates

| Dataset | Training (1000 seeds) | Aggregation |
|---------|----------------------|-------------|
| MNIST | ~5-10 min each | ~30 min |
| Fashion-MNIST | ~5-10 min each | ~30 min |
| CIFAR-10 | ~10-15 min each | ~45 min |
| SVHN | ~10-15 min each | ~45 min |

With `--array=1-1000%50` (50 concurrent jobs), all training completes in ~3-4 hours wall time per dataset.
