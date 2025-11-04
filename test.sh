#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=00:05:00

#Execute program located in $HOME
cd $HOME/WeightSpace/WeightSpaceClassifier/
source .venv/bin/activate
python run_experiment.py mnist 8
