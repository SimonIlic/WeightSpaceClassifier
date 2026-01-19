#!/bin/bash -l
#SBATCH -J cifar10_early
#SBATCH -t 4:00:00
#SBATCH -p rome
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=moos@cwi.nl
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate
python src/cnn_surgery/utils/evaluate_per_class_accuracy.py --dataset cifar10 --stage early
