"""
Loop over the runs. In runs/, for each run:
1. Load the latest checkpoint
2. Get the weights
3. Add the flattened weights to a collection
4. Save the processed weights as weights.npy
"""

import json
from pathlib import Path
from typing import List, Optional

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from cnn_surgery.utils.load_dataset import find_project_root


# Thanks CLAUDE!
def flatten_weights_for_reconstruction(weights: List[np.ndarray]) -> np.ndarray:
    """
    Flatten model weights in the order expected by reconstruct_network.py.

    The model.get_weights() returns [kernel, bias, kernel, bias, ...]
    But reconstruct_network expects [bias, kernel, bias, kernel, ...]

    Args:
        weights: List of weight arrays from model.get_weights()

    Returns:
        Flattened weights in the order expected by reconstruction
    """
    reordered_weights = []

    # Process weights in pairs (kernel, bias)
    for i in range(0, len(weights), 2):
        if i + 1 < len(weights):
            # Add bias first, then kernel
            reordered_weights.append(weights[i + 1])  # bias
            reordered_weights.append(weights[i])  # kernel
        else:
            # Handle odd case (shouldn't happen with CNN)
            reordered_weights.append(weights[i])

    return np.concatenate([w.flatten() for w in reordered_weights])


def flatten_and_aggregate_model_weights(
    runs_dir: Optional[Path] = None,
    checkpoint_name: str = "permanent_ckpt-86.keras",
    output_filename: str = "weights.npy",
    verbose: bool = True,
) -> np.ndarray:
    """
    Aggregate flattened weights from multiple model checkpoints across different runs
    and saves them as a numpy array.

    This function processes model checkpoints from different training runs, extracts
    their weights, flattens them into 1D arrays, and combines them into a single
    2D numpy array where each row represents the flattened weights of one model.

    Args:
        runs_dir: Directory containing the run subdirectories. If None, uses
                 project_root/runs.
        checkpoint_name: Name of the checkpoint file to load from each run.
        output_filename: Name of the output file to save the aggregated weights.
        verbose: Whether to print progress and shape information.

    Returns:
        A 2D numpy array where:
        - Shape: (num_models, num_parameters_per_model)
        - Each row contains the flattened weights of one model
    """
    if runs_dir is None:
        root_dir = find_project_root()
        runs_dir = root_dir / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Initialize empty list to collect flattened weights from each model
    all_weights: List[np.ndarray] = []

    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            if verbose:
                print(f"Processing {run_dir}")

            checkpoint_path = run_dir / checkpoint_name

            model = keras.saving.load_model(checkpoint_path)
            assert model is not None, f"Failed to load model from {checkpoint_path}"

            weights = model.get_weights()  # returns a list of numpy ndarrays
            weights_flat = flatten_weights_for_reconstruction(weights)

            all_weights.append(weights_flat)

    if not all_weights:
        raise ValueError(f"No valid checkpoints found in {runs_dir}")

    all_weights_array = np.vstack(all_weights)

    if verbose:
        print(f"\nFinal weights array shape: {all_weights_array.shape}")
        print(f"Number of models: {all_weights_array.shape[0]}")
        print(f"Number of parameters per model: {all_weights_array.shape[1]}")

    # Save the aggregated weights
    weights_path = runs_dir / output_filename
    np.save(weights_path, all_weights_array)

    if verbose:
        print(f"Saved weights to: {weights_path}")

        # Verification
        loaded_weights = np.load(weights_path)
        print(f"Verification - loaded weights shape: {loaded_weights.shape}")

    return all_weights_array


def collect_metadata_runs(
    runs_dir: Optional[Path] = None,
    metadata_name: str = "results.json",
    output_filename: str = "metrics.csv",
    verbose: bool = True,
) -> None:
    if runs_dir is None:
        root_dir = find_project_root()
        runs_dir = root_dir / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Initialize empty list to collect metadata from each model
    all_metadata: List[pd.Series] = []

    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            if verbose:
                print(f"Processing {run_dir}")

            metadata_path = run_dir / metadata_name
            with open(metadata_path) as f:
                metadata = json.load(f)

            # we are excluding step and laststep (this was present in the original metadata)
            metrics_series = pd.Series(
                {
                    "config.activation": metadata["config.activation"],
                    "config.b_init": metadata["config.b_init"],
                    "config.dataset": metadata["config.dataset"],
                    "config.dnn_architecture": metadata["config.dnn_architecture"],
                    "config.dropout": metadata["config.dropout"],
                    "config.epochs": metadata["config.epochs"],
                    "config.epochs_between_checkpoints": metadata["config.epochs_between_checkpoints"],
                    "config.init_std": metadata["config.init_std"],
                    "config.l2reg": metadata["config.l2reg"],
                    "config.learning_rate": metadata["config.learning_rate"],
                    "config.num_layers": metadata["config.num_layers"],
                    "config.num_units": metadata["config.num_units"],
                    "config.optimizer": metadata["config.optimizer"],
                    "config.random_seed": metadata["config.random_seed"],
                    "config.train_fraction": metadata["config.train_fraction"],
                    "config.w_init": metadata["config.w_init"],
                    "config.exclude_class": metadata["config.exclude_class"],
                    "modeldir": run_dir,
                    "test_accuracy": metadata["test_accuracy"]["86"],
                    "test_loss": metadata["test_loss"]["86"],
                    "train_accuracy": metadata["train_accuracy"]["86"],
                    "train_loss": metadata["train_loss"]["86"],
                }
            )

            # add it to the dataframe of all metrics
            all_metadata.append(metrics_series)

    df = pd.DataFrame(all_metadata)
    df.to_csv(runs_dir / output_filename, index=False)


if __name__ == "__main__":
    # Example usage
    weights_array = flatten_and_aggregate_model_weights()
    collect_metadata_runs()
