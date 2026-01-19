#!/bin/bash
#SBATCH --job-name=aggregate_cnns
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Aggregate trained CNNs into model zoo format (weights.npy + metrics.csv)
#
# Usage:
#   sbatch --export=DATASET=mnist aggregate_job.sh
#   sbatch --export=DATASET=mnist,DELETE_CHECKPOINTS=1 aggregate_job.sh
#   sbatch --dependency=afterok:<training_job_id> --export=DATASET=mnist aggregate_job.sh
#
# Environment variables:
#   DATASET            - Required: mnist, fashion_mnist, cifar10, svhn_cropped
#   DELETE_CHECKPOINTS - Optional: Set to 1 to delete .keras files after aggregation
#   OUTPUT_BASE        - Optional: Base output directory (default: /scratch-shared/$USER/controlled_hyperparams)
#   PROJECT_DIR        - Optional: Project directory (default: $HOME/WeightSpaceClassifier)
#   COPY_TO_HOME       - Optional: Set to 1 to copy results to home directory after aggregation

set -e

# Validate required environment variable
if [[ -z "$DATASET" ]]; then
    echo "ERROR: DATASET environment variable is required"
    echo "Usage: sbatch --export=DATASET=mnist aggregate_job.sh"
    exit 1
fi

# Set defaults
PROJECT_DIR="${PROJECT_DIR:-$HOME/WeightSpaceClassifier}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/controlled_hyperparams}"
DELETE_CHECKPOINTS="${DELETE_CHECKPOINTS:-0}"
COPY_TO_HOME="${COPY_TO_HOME:-0}"

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

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: $DATASET"
echo "Output base: $OUTPUT_BASE"
echo "Delete checkpoints: $DELETE_CHECKPOINTS"
echo "Copy to home: $COPY_TO_HOME"
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

# Build aggregation command
AGGREGATE_ARGS="--dataset $DATASET --output-dir $OUTPUT_BASE"
if [[ "$DELETE_CHECKPOINTS" == "1" ]]; then
    AGGREGATE_ARGS="$AGGREGATE_ARGS --delete-checkpoints"
fi

# Run aggregation
echo "Running aggregation for $DATASET..."
python "$PROJECT_DIR/experiments/controlled_hyperparams/aggregate_results.py" $AGGREGATE_ARGS

echo "Aggregation complete!"

# Copy results to home directory if requested
if [[ "$COPY_TO_HOME" == "1" ]]; then
    HOME_OUTPUT="$PROJECT_DIR/experiments/controlled_hyperparams/output/$DATASET"
    mkdir -p "$HOME_OUTPUT"

    echo "Copying results to $HOME_OUTPUT..."
    cp "$OUTPUT_BASE/$DATASET/weights.npy" "$HOME_OUTPUT/"
    cp "$OUTPUT_BASE/$DATASET/metrics.csv" "$HOME_OUTPUT/"

    echo "Results copied to home directory"
fi

echo "Done!"
echo "  Weights: $OUTPUT_BASE/$DATASET/weights.npy"
echo "  Metrics: $OUTPUT_BASE/$DATASET/metrics.csv"
