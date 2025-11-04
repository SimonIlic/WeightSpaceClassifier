#!/bin/bash
#SBATCH --job-name=wsc16pack
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # you will be charged for 16 anyway
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out.txt
#SBATCH --error=logs/%j.err.txt
set -euo pipefail

# Pin every proc to 1 thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

cd "$HOME/WeightSpace/WeightSpaceClassifier/"
source .venv/bin/activate

# Generate 10k “tickets” and run up to 16 at once
seq 1 10000 | xargs -I{} -P ${SLURM_CPUS_PER_TASK} bash -lc \
  'python run_experiment.py mnist 8'