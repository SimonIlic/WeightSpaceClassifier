# Metanetwork Models in `metanetworks/pt/`

This directory contains trained **regressor lens models** (also called "metanetworks") that predict CNN classification accuracies from their flattened weight vectors.

IF YOU WANT THESE MODELS, CONTACT MOOS (MOOS@CWI.NL). TOO BIG FOR GITHUB.

## Where These Models Came From

These models were created as part of the "good and bad CNNs" experiment documented in `notebooks/good_and_bad_CNNs.ipynb`. The experiment aimed to:

1. Train multiple metanetworks to evaluate CNNs
2. Analyze intra-CNN variance to understand metanetwork-invariance
3. Cluster good vs. bad CNNs for unlearning experiments

The models were trained using the `get_regressor_lens()` function from `src/cnn_surgery/lenses/regressor_lens.py`. This function trains a fully-connected neural network (FCN) to regress from CNN weight vectors to their per-class accuracies.

### Training Data

The models were trained on datasets created by `load_multi_stage_dataset()` which combines:
- **Early stage** models (from `metrics_merged_early.csv`)
- **Middle stage** models (from `metrics_merged_middle.csv`)  
- **Final stage** models (from `metrics_merged_final.csv`)

Each dataset contains:
- **Input**: Flattened CNN weight vectors (shape: `[n_models, n_weights]`)
- **Output**: Per-class accuracies (shape: `[n_models, 10]` for 10-class classification)

### Model Architecture

All models use the same architecture (defined in `regressor_lens.py`):
- **Type**: Fully-connected network (FCN)
- **Layers**: 5 hidden layers
- **Hidden units**: 256 per layer
- **Dropout**: 0.03
- **Output activation**: Sigmoid (to constrain predictions to [0, 1])
- **Optimizer**: Adam with learning rate 4e-4
- **Regularization**: L2 penalty of 2e-5

## Directory Structure

### `good_bad_experiment_2/` (Recommended)

This directory contains the most recent version of the models with associated metrics:

- **15 model files**: 3 datasets × 5 models per dataset
  - `{dataset}_metanetwork_{i}.pt` where:
    - `dataset` ∈ `{mnist, fashion_mnist, cifar10}`
    - `i` ∈ `{0, 1, 2, 3, 4}` (5 different random initializations)

- **15 metrics files**: `{dataset}_metanetwork_{i}_metrics.json`
  - Contains training/validation metrics: `mse_train`, `mae_train`, `mse_val`, `mae_val`, `r2_val`

### `good_bad_experiment/` (Older Version)

Contains an earlier version of the same models (15 `.pt` files) without metrics files. Use `good_bad_experiment_2/` instead.

### `main_regressor_lens_fashion_mnist.pt` (Standalone)

A single metanetwork model for Fashion-MNIST, likely created for a different experiment. Used as a default in `src/cnn_surgery/evaluate_models.py`.

## How They Were Saved

The models were saved using PyTorch's `state_dict()` format:

```python
torch.save(MetaNetwork.state_dict(), f"{dataset}_metanetwork_{i}.pt")
```

**Important**: Only the model weights are saved, not the full model object. This means you need to reconstruct the model architecture before loading.

Metrics were saved as JSON files alongside the models:

```python
metrics_dict = {
    "mse_train": metrics[0][0],
    "mae_train": metrics[0][1],
    "mse_val": metrics[1][0],
    "mae_val": metrics[1][1],
    "r2_val": metrics[2],
}
json.dump(metrics_dict, f)
```

## How to Load Them

To load a metanetwork model, you need to:

1. **Instantiate an FCN model** with the correct architecture parameters
2. **Load the state dictionary** from the `.pt` file
3. **Set the model to evaluation mode**

### Example Loading Code

```python
import torch
import torch.nn as nn
from cnn_surgery.lenses.regressor_lens import FCN, default_config
from cnn_surgery.utils.load_dataset import load_multi_stage_dataset

# Load dataset to get the correct input/output dimensions
dataset = "fashion_mnist"
train, val, _ = load_multi_stage_dataset(include_test=False, dataset=dataset).values()
weights_train = train[0]
accuracies_train = train[1]

# Create model with matching architecture
metanetwork = FCN(
    input_dim=weights_train.shape[1],  # Number of weight parameters
    n_layers=int(default_config["n_layers"]),  # 5
    n_hidden=int(default_config["n_hiddens"]),  # 256
    n_outputs=accuracies_train.shape[1],  # 10 (number of classes)
    dropout_p=float(default_config["dropout_rate"]),  # 0.03
    activation=nn.ReLU,
    last_activation="sigmoid",
)

# Load the saved state dictionary
model_path = "metanetworks/pt/good_bad_experiment_2/fashion_mnist_metanetwork_0.pt"
metanetwork.load_state_dict(torch.load(model_path, map_location="cpu"))

# Set to evaluation mode
metanetwork.eval()

# Optionally disable gradients for inference
for param in metanetwork.parameters():
    param.requires_grad = False
```

### Loading Metrics

```python
import json

metrics_path = "metanetworks/pt/good_bad_experiment_2/fashion_mnist_metanetwork_0_metrics.json"
with open(metrics_path, "r") as f:
    metrics = json.load(f)
    print(f"Validation R²: {metrics['r2_val']:.4f}")
    print(f"Validation MSE: {metrics['mse_val']:.6f}")
    print(f"Validation MAE: {metrics['mae_val']:.6f}")
```

### Reference Implementation

See `src/cnn_surgery/evaluate_models.py` (lines 100-114) for a production example of loading these models.

## Usage

These metanetworks are used to:
- **Predict CNN accuracies** from weight vectors without training
- **Evaluate unlearning experiments** by comparing predicted vs. actual accuracies after weight modifications
- **Analyze metanetwork variance** across different random initializations
- **Cluster CNNs** into good/bad categories for targeted unlearning

## Notes

- Models were trained on CPU (device="cpu" in the notebook)
- All models use the same hyperparameters defined in `default_config`
- The 5 models per dataset represent different random initializations, allowing analysis of metanetwork variance
- Models predict per-class accuracies (10 outputs) normalized to [0, 1] via sigmoid activation
