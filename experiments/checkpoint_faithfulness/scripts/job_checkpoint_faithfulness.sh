#!/bin/bash
#SBATCH --job-name=checkpoint_faithfulness
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Disable XLA JIT compilation (avoids libdevice.10.bc issue)
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# Activate environment
cd ~/WeightSpaceClassifier
source .venv/bin/activate

echo "=========================================="
echo "Starting Checkpoint Faithfulness Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Phase 1: Training (skipped - already complete)
echo "Phase 1: Skipping training (meta-networks already trained)"
echo ""

# Phase 2: Evaluation (only CIFAR10 and SVHN - MNIST/Fashion-MNIST already done)
# Using --serial to avoid GPU memory contention with parallel jobs
echo "Phase 2: Running evaluation for CIFAR10 and SVHN (serial mode)..."
export DATASETS="cifar10 svhn_cropped"
./experiments/checkpoint_faithfulness/scripts/eval_all.sh --serial

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "End time: $(date)"
echo "=========================================="
