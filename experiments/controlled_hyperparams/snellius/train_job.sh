#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --array=1-1000%50
#SBATCH --cpus-per-task=9
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Train a single CNN with seed = SLURM_ARRAY_TASK_ID
#
# Usage:
#   sbatch --export=DATASET=mnist train_job.sh
#   sbatch --export=DATASET=mnist,START_SEED=501,END_SEED=1000 --array=501-1000%50 train_job.sh
#
# Environment variables:
#   DATASET      - Required: mnist, fashion_mnist, cifar10, svhn_cropped
#   OUTPUT_BASE  - Optional: Base output directory (default: /scratch-shared/$USER/controlled_hyperparams)
#   PROJECT_DIR  - Optional: Project directory (default: $HOME/WeightSpaceClassifier)

set -e

# Validate required environment variable
if [[ -z "$DATASET" ]]; then
    echo "ERROR: DATASET environment variable is required"
    echo "Usage: sbatch --export=DATASET=mnist train_job.sh"
    exit 1
fi

# Set defaults
PROJECT_DIR="${PROJECT_DIR:-$HOME/WeightSpaceClassifier}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/controlled_hyperparams}"
CONFIG_PATH="$PROJECT_DIR/configs/optimal_cnn_hyperparams.json"

# Validate dataset
case "$DATASET" in
    mnist|fashion_mnist|cifar10|svhn_cropped)
        ;;
    *)
        echo "ERROR: Invalid dataset '$DATASET'"
        echo "Must be one of: mnist, fashion_mnist, cifar10, svhn_cropped"
        exit 1
        ;;
esac

# Get seed from array task ID
SEED=$SLURM_ARRAY_TASK_ID

# Create output directories
DATASET_OUTPUT="$OUTPUT_BASE/$DATASET"
WORKDIR="$DATASET_OUTPUT/seed_$SEED"
mkdir -p "$WORKDIR"

# Skip if already completed
if [[ -f "$WORKDIR/results.json" ]]; then
    echo "[$DATASET] Seed $SEED: already completed, skipping"
    exit 0
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Output: $WORKDIR"
echo "=========================================="

# Load modules (Snellius 2024 software stack)
module purge
module load 2024
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

# Activate virtual environment
cd "$PROJECT_DIR"
source .venv/bin/activate

# Verify Python environment
echo "Python: $(which python)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Run training
echo "[$DATASET] Seed $SEED: starting training..."

python -m cnn_surgery.utils.train_network \
    --dataset "$DATASET" \
    --config "$CONFIG_PATH" \
    --random_seed "$SEED" \
    --workdir "$WORKDIR" \
    --nosave_intermediate_checkpoints \
    --verbose 0

echo "[$DATASET] Seed $SEED: training complete"
