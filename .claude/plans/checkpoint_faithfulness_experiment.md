# Experiment: Meta-Network Faithfulness vs Training Checkpoint Diversity

## Research Question
Does training meta-networks on diverse checkpoints (early, middle, final) lead to better **faithfulness** during unlearning compared to training only on final checkpoints?

## Hypothesis
Training on early/middle checkpoints exposes the meta-network to weights that don't have the typical "SGD-optimized" look, expanding its trust region so predictions remain accurate when weights are modified significantly during unlearning.

---

## Experiment Design

### Conditions
| Condition | Training Data | Samples |
|-----------|--------------|---------|
| **final-only** | `metrics_merged_final.csv` only | 15K |
| **multi-stage** | `load_multi_stage_dataset()` (early+middle+final) | 45K |

### Parameters
- **Replicates**: 5 meta-networks per condition (different seeds: 42, 123, 456, 789, 1011)
- **Datasets**: MNIST, Fashion-MNIST, CIFAR-10, SVHN (all four)
- **Evaluation**: 100 CNNs per condition (expandable via `--n-models` flag)
- **Total**: 40 meta-networks, 40,000 unlearning runs

### Faithfulness Metrics
1. **Final MAE**: `mean(|final_pred - accuracy_after|)` after unlearning
2. **mean_diff trajectory**: Store `mean_diff` at every step for complete trajectory analysis
3. **Initial MAE**: `mean(|init_pred - original_accuracy|)` (baseline control)

> **Design decision**: Store raw mean_diff values at every step (~80KB per run, ~3.2GB total). This allows complete trajectory visualization and flexible post-hoc threshold analysis with no information loss.

---

## Implementation Plan

### Step 1: Modify `regressor_lens.py` - Add seed parameter
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/src/cnn_surgery/lenses/regressor_lens.py`

Add `seed` parameter to `train_torch_dnn()` and `get_regressor_lens()`:
```python
def train_torch_dnn(..., seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
```

### Step 2: Modify `unlearning.py` - Add step callback
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/src/cnn_surgery/unlearning.py`

Add optional callback to track predictions at each step:
```python
def unlearn(..., step_callback=None):
    # In loop:
    if step_callback is not None:
        step_callback(step, acc_pred.detach().clone(), weights.detach().clone())
```

### Step 3: Create meta-network training script
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/experiments/checkpoint_faithfulness/train_metanetworks.py`

```python
def train_and_save_metanetwork(dataset, condition, seed, output_dir):
    """Train meta-network with specified condition and save with metrics."""
    if condition == 'final-only':
        train, _, val = load_dataset(dataset, metrics_file='metrics_merged_final.csv',
                                      load_class_acc=True, stage='final')
    else:  # multi-stage
        data = load_multi_stage_dataset(dataset=dataset)
        train, val = data['train'], data['val']

    model, metrics = get_regressor_lens(..., seed=seed, return_metrics=True)
    # Save model and metrics JSON
```

CLI: `python train_metanetworks.py --dataset mnist --condition multi-stage --seed 42`

### Step 4: Create faithfulness evaluation module
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/experiments/checkpoint_faithfulness/evaluate_faithfulness.py`

```python
@dataclass
class FaithfulnessResult:
    model_idx: int
    target_class: int
    initial_mae: float           # |init_pred - original_accuracy|
    final_mae: float             # |final_pred - accuracy_after|
    mean_diff_trajectory: List[float]  # mean_diff at every step
    total_steps: int
    distance_travelled: float

class FaithfulnessCallback:
    """Track mean_diff at every step for complete trajectory analysis."""
    def __init__(self, original_accuracy):
        self.original_accuracy = original_accuracy
        self.mean_diff_trajectory = []

    def __call__(self, step, pred, weights):
        mean_diff = np.abs(pred.numpy() - self.original_accuracy).mean()
        self.mean_diff_trajectory.append(mean_diff)
```

### Step 5: Create evaluation runner
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/experiments/checkpoint_faithfulness/run_evaluation.py`

- Loads meta-network and validation CNNs
- For each model × target class:
  1. Create `FaithfulnessCallback`
  2. Run `unlearn()` with callback
  3. Evaluate actual accuracy via `test_network_accuracy()`
  4. Compute metrics, append to CSV

CLI: `python run_evaluation.py --meta-network-path ... --dataset mnist --n-models 100`

### Step 6: Create experiment orchestrator
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/experiments/checkpoint_faithfulness/run_experiment.py`

Main script that:
1. Trains all 40 meta-networks (if not already cached)
2. Runs all evaluations
3. Aggregates results into summary CSV

### Step 7: Create analysis notebook
**File**: `~/Desktop/PHD/snijzaal/WeightSpaceClassifier/notebooks/checkpoint_faithfulness_analysis.ipynb`

Analyses:
1. **Box plots**: Final MAE by condition (per dataset)
2. **Survival curves**: % faithful at each step (Kaplan-Meier style)
3. **Scatter**: distance_travelled vs final_mae (colored by condition)
4. **Statistical tests**: Paired t-test between conditions

---

## File Structure

```
experiments/checkpoint_faithfulness/
├── train_metanetworks.py       # Train meta-networks with seeds
├── evaluate_faithfulness.py    # Faithfulness metrics & callback
├── run_evaluation.py           # Single-condition evaluator
├── run_experiment.py           # Main orchestrator
├── scripts/
│   ├── train_all.sh           # Train all 40 meta-networks
│   └── eval_all.sh            # Run all evaluations
├── metanetworks/
│   ├── final_only/
│   │   └── {dataset}_seed{seed}.pt
│   └── multi_stage/
│       └── {dataset}_seed{seed}.pt
└── results/
    ├── {dataset}_{condition}_seed{seed}.csv
    └── summary.csv
```

---

## CSV Output Schema

### Per-run results: `{dataset}_{condition}_seed{seed}.csv`

| Column | Description |
|--------|-------------|
| model_idx | Index in validation set |
| target_class | Class being unlearned (0-9) |
| dataset | Dataset name |
| condition | 'final-only' or 'multi-stage' |
| seed | Random seed |
| original_accuracy | JSON list of 10 floats |
| accuracy_after | JSON list of 10 floats |
| init_pred | JSON list of 10 floats |
| final_pred | JSON list of 10 floats |
| **initial_mae** | Mean |init_pred - original_accuracy| |
| **final_mae** | Mean |final_pred - accuracy_after| |
| **mean_diff_trajectory** | JSON list of mean_diff at every step |
| total_steps | Unlearning steps taken |
| distance_travelled | Cumulative L2 distance |

---

## Computational Estimates

| Phase | Count | Time Each | Total |
|-------|-------|-----------|-------|
| Training | 40 meta-networks | ~3 min | ~2 hours |
| Evaluation | 40,000 unlearning runs | ~1.5 sec | ~17 hours |

Can parallelize across datasets (4x speedup).

---

## Critical Files to Modify

1. `src/cnn_surgery/lenses/regressor_lens.py` - Add seed parameter
2. `src/cnn_surgery/unlearning.py` - Add step_callback parameter

## New Files to Create

1. `experiments/checkpoint_faithfulness/train_metanetworks.py`
2. `experiments/checkpoint_faithfulness/evaluate_faithfulness.py`
3. `experiments/checkpoint_faithfulness/run_evaluation.py`
4. `experiments/checkpoint_faithfulness/run_experiment.py`
5. `notebooks/checkpoint_faithfulness_analysis.ipynb`
