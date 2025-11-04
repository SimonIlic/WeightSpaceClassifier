#!/bin/bash
#SBATCH --job-name=wsc_array
#SBATCH --partition=rome
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-10000%128
#SBATCH --output=logs/%A_%a.out.txt
#SBATCH --error=logs/%A_%a.err.txt

set -euo pipefail

cd "$HOME/WeightSpace/WeightSpaceClassifier/"
source .venv/bin/activate

# run in parallel
srun python run_experiment.py mnist 8