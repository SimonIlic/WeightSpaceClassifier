# Checkpoint Faithfulness Experiment

This experiment tests whether training meta-networks on diverse checkpoints (early, middle, final) leads to better **faithfulness** during unlearning compared to training only on final checkpoints.

## Hypothesis

Training on early/middle checkpoints exposes the meta-network to weights that don't have the typical "SGD-optimized" look, expanding its trust region so predictions remain accurate when weights are modified significantly during unlearning.

## Experiment Design

### Conditions
| Condition | Training Data | Samples |
|-----------|--------------|---------|
| **final-only** | `metrics_merged_final.csv` only | 15K |
| **multi-stage** | early+middle+final combined | 45K |

### Parameters
- **Replicates**: 5 meta-networks per condition (seeds: 42, 123, 456, 789, 1011)
- **Datasets**: MNIST, Fashion-MNIST, CIFAR-10, SVHN
- **Evaluation**: 100 CNNs per condition (expandable via `--n-models`)
- **Total**: 40 meta-networks, 40,000 unlearning runs

### Faithfulness Metrics
1. **Final MAE**: `mean(|final_pred - accuracy_after|)` after unlearning
2. **mean_diff trajectory**: MAE at every step for post-hoc threshold analysis
3. **Initial MAE**: `mean(|init_pred - original_accuracy|)` (baseline control)

## Quick Start

```bash
# 1. Train all meta-networks (~2 hours)
python train_metanetworks.py --all

# 2. Run all evaluations (~17 hours, parallelizable)
python run_experiment.py --eval-only

# 3. Analyze results
# Open notebooks/checkpoint_faithfulness_analysis.ipynb
```

## Usage

### Train a single meta-network
```bash
python train_metanetworks.py --dataset mnist --condition multi-stage --seed 42
```

### Train all meta-networks
```bash
python train_metanetworks.py --all --verbose
```

### Run evaluation for a specific meta-network
```bash
python run_evaluation.py \
    --meta-network-path metanetworks/multi_stage/mnist_seed42.pt \
    --dataset mnist \
    --condition multi-stage \
    --seed 42 \
    --n-models 100
```

### Run full experiment
```bash
python run_experiment.py --all
```

### Run for specific dataset
```bash
python run_experiment.py --dataset mnist
```

## Output Structure

```
checkpoint_faithfulness/
├── metanetworks/
│   ├── final_only/
│   │   └── {dataset}_seed{seed}.pt
│   └── multi_stage/
│       └── {dataset}_seed{seed}.pt
└── results/
    ├── {dataset}_{condition}_seed{seed}.csv
    ├── all_results.csv
    └── summary.csv
```

## CSV Schema

| Column | Description |
|--------|-------------|
| model_idx | Index in validation set |
| target_class | Class being unlearned (0-9) |
| initial_mae | Mean |init_pred - original_accuracy| |
| final_mae | Mean |final_pred - accuracy_after| |
| mean_diff_trajectory | JSON list of MAE at every step |
| total_steps | Unlearning steps taken |
| distance_travelled | Cumulative L2 distance |

## Analysis

See `notebooks/checkpoint_faithfulness_analysis.ipynb` for:
- Box plots of final MAE by condition
- Trajectory visualizations
- Statistical tests (paired t-test, Wilcoxon)
- Unfaithfulness threshold analysis
