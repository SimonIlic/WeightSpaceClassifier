# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Conventions

**Package Management**: Always use `uv` instead of `pip`. Use `uv pip install`, `uv add`, `uv sync`, etc.

**File Paths**: Always use `~` to represent the home directory (`/Users/moos`) for conciseness and portability. This applies to:
- File paths in code and documentation
- Permissions in settings.json
- Command-line arguments
- Examples: `~/Documents/...` instead of `/Users/moos/Documents/...` or `//Users/moos/Documents/...`

## Project Overview

This project (codenamed "project snijzaal") implements neural network unlearning techniques using meta-networks (also called "lenses" or "regressor lenses"). The core idea is to train a meta-network that predicts per-class accuracy from model weights, then use gradient descent in weight space to "unlearn" specific classes from trained CNNs.

The codebase uses the "Small CNN Zoo" dataset from Unterthiner et al. (2020) "Predicting Neural Network Accuracy from Weights", which contains 30k trained CNN instances across MNIST, Fashion-MNIST, CIFAR-10, and SVHN datasets.

It is scientific work, with the aim of publishing at a top ML conference. The code is structured for research experiments, reproducibility, and extensibility. 

## HPC Environment (Snellius)

Sometimes runs for this project are performed on a HPC cluster. Specifically on **Snellius**, the Dutch national supercomputer operated by SURF.

- **Documentation**: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius
- **Login nodes**: Access via SSH
- **Job scheduler**: Slurm

### Account & Budget

- **Account**: `vusr98271`
- **Budget**: 100,000 SBU (expires 2026-06-30)
- **Check budget**: Run `accinfo`

### Available Partitions

| Partition | Access | Description |
|-----------|--------|-------------|
| `gpu_a100` | Yes | NVIDIA A100 GPUs |
| `gpu_h100` | Yes | NVIDIA H100 GPUs |
| `gpu_mig` | Yes | Multi-Instance GPU partitions |
| `gpu_vis` | Yes | Visualization GPUs (1-day max) |
| `cbuild` | Yes | Build/compile nodes |
| `staging` | Yes | Data staging |
| `rome`, `genoa` | No | CPU compute (not in budget) |

## Installation and Environment

```bash
# Install uv package manager first, then:
uv sync
uv pip install -e .
```

Adding new packages:
```bash
uv add <package_name>
```

The project uses `ruff` for code formatting and linting (configured in `pyproject.toml`).

## Key Commands

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model_reconstruction.py -v

# Quick reconstruction verification
python tests/test_reconstruction_core.py
```

### Training and Evaluation
```bash
# Main unlearning evaluation script (for rigorous experiments)
python -m cnn_surgery.evaluate_models \
    -d mnist \
    -c 5 \
    --max-steps 10000 \
    --lr 0.1 \
    --loss-fn simple \
    --stopping-criterium acc_pred \
    --meta-network-path metanetworks/meta_network_mnist_5.pkl

# Per-class evaluation
./per_class_evaluation.sh

# Hyperparameter tuning for meta-network
python -m cnn_surgery.tuning
```

The `notebooks/unlearning.ipynb` notebook contains up-to-date exploratory experiments (primarily used by @moosmiddelkoop).

## Architecture

### Core Components

**Meta-Network (Regressor Lens)**
- Located in: `src/cnn_surgery/lenses/regressor_lens.py`
- A PyTorch MLP that takes flattened CNN weights as input and predicts per-class accuracies (10 outputs for 10 classes)
- Architecture: 5 hidden layers with 256 units each, ReLU activation, dropout, sigmoid output
- Default hyperparameters stored in `default_config` dict or `configs/best_metanetwork_hyperparams.json`
- Trained using `get_regressor_lens()` function
- Pre-trained models stored in `metanetworks/` directory, named like `meta_network_{dataset}_{class}.pkl`

**Unlearning Algorithm**
- Located in: `src/cnn_surgery/unlearning.py`
- Main function: `unlearn(model_weights, meta_network, target_class, ...)`
- Uses gradient descent in weight space guided by the meta-network
- Supports multiple loss functions: `simple_loss`, `boost_loss`, `improve_loss`
- Supports multiple stopping criteria: accuracy threshold, cosine similarity, max steps
- Returns modified weights and metrics (tracked via `UnlearnState` dataclass)

**CNN Architecture**
- 3 convolutional layers (16 filters each, 3x3 kernels, stride=2)
- Global average pooling
- 10-output dense layer
- Total: 4970 parameters
- Built in TensorFlow/Keras via `build_cnn()` in `train_network.py`

### Data Flow

1. **Model Zoo**: Pre-trained CNN weights stored in `model_zoo/{dataset}/weights.npy` (shape: [n_models, 4970])
2. **Metrics**: Per-class accuracies and training info in `model_zoo/{dataset}/metrics_merged_final.csv`
3. **Meta-Network Training**: Uses `load_dataset()` from `utils/load_dataset.py` to load weights + accuracies
4. **Unlearning**:
   - Load meta-network from `metanetworks/`
   - Run gradient descent on CNN weights using `unlearn()`
   - Evaluate unlearned CNN via `reconstruct_network()` and `evaluate_classifier()`

### Critical Implementation Details

**Weight Ordering**: The weights in `weights.npy` must match the order expected by `reconstruct_network()`:
- Storage order: `[bias, kernel, bias, kernel, ...]` (for each layer)
- This differs from Keras `model.get_weights()` which returns `[kernel, bias, kernel, bias, ...]`
- Conversion handled by `_flatten_weights_for_reconstruction()` in `process_models.py`
- See `tests/README_TESTS.md` for full explanation of this critical fix

**Network Reconstruction**:
- Use `reconstruct_network(weights, activation)` from `utils/reconstruct_network.py`
- Returns a Keras model (must compile before use)
- Uses fixed shapes in `SHAPES` dict and `reshape_weights()` function

**Dataset Loading**:
- Use `load_dataset(dataset, metrics_file, load_class_acc)` for training meta-networks
- Returns tuple: `(train_data, test_data, val_data)` where each is `(weights, metrics, configs)`
- Use `load_testset_data(dataset)` for evaluating reconstructed CNNs on actual images

## Directory Structure

```
src/cnn_surgery/
├── lenses/
│   └── regressor_lens.py       # Meta-network (PyTorch MLP)
├── utils/
│   ├── train_network.py        # CNN training (TensorFlow, from Unterthiner et al.)
│   ├── reconstruct_network.py  # Rebuild CNN from flat weights
│   ├── process_models.py       # Weight extraction/flattening
│   ├── load_dataset.py         # Data loading for meta-network training
│   ├── evaluate_per_class_accuracy.py  # CNN evaluation utilities
│   └── metrics.py              # Evaluation metrics
├── unlearning.py               # Main unlearning algorithm
├── evaluate_models.py          # Rigorous unlearning experiments
└── tuning.py                   # Meta-network hyperparameter search

model_zoo/{dataset}/
├── weights.npy                 # All CNN weights [n_models, 4970]
├── metrics_merged_final.csv    # Per-class accuracies + training config
└── README.md                   # Dataset documentation

metanetworks/
└── meta_network_{dataset}_{class}.pkl  # Trained meta-networks

configs/
├── best_configs.json           # Optimal CNN training hyperparameters
└── best_metanetwork_hyperparams.json  # Optimal meta-network hyperparameters

experiments/
├── ensemble_unlearning/        # Ensemble experiments
├── per_class_evaluation/       # Per-class evaluation results
└── good-bad/                   # Good vs bad model analysis

tests/
├── test_model_reconstruction.py  # Comprehensive reconstruction tests
└── test_reconstruction_core.py   # Quick reconstruction verification
```

## Common Patterns

**Evaluating Unlearning**:
1. Load meta-network: `torch.load()` or `pickle.load()`
2. Get original CNN weights from dataset
3. Run `unlearn()` to get modified weights
4. Reconstruct CNN: `reconstruct_network(modified_weights, activation)`
5. Compile and evaluate on test set: `evaluate_classifier(model, x_test, y_test)`

**Training a Meta-Network**:
```python
from cnn_surgery.lenses.regressor_lens import get_regressor_lens, default_config
train_data, _, val_data = load_dataset(dataset="mnist", metrics_file="metrics_merged_final.csv", load_class_acc=True)
model = get_regressor_lens(train_data, val_data, config=default_config)
```

**Loss Functions** (in `unlearning.py`):
- `simple_loss`: Only reduces target class accuracy
- `boost_loss_factory(beta)`: Reduces target class while boosting others
- `improve_loss`: Increases target class accuracy (inverse unlearning)

**Stopping Criteria**:
- `acc_pred_stop_factory(threshold)`: Stop when predicted accuracy < threshold
- `cosine_similarity_stop_factory(eps)`: Stop when gradient direction changes
- `step_stop_factory(max_steps)`: Stop after fixed number of steps

## Datasets

Supported datasets: `mnist`, `fashion_mnist`, `cifar10`, `svhn_cropped`

Each model zoo contains:
- ~270k trained CNN instances
- 9 checkpoints per training run (epochs: 0, 1, 2, 3, 20, 40, 60, 80, 86)
- Various hyperparameter combinations (activation, dropout, l2reg, learning_rate, optimizer, etc.)
- Per-class accuracies on test set

## Package Name

The package is installed as `cnn_surgery` (not `WeightSpaceClassifier`). Import as:
```python
from cnn_surgery.unlearning import unlearn
from cnn_surgery.lenses.regressor_lens import FCN
```
