"""
Loop over the runs. In runs/, for each run:
1. Load the latest checkpoint
2. Get the weights
3. Add the flattened weights to a collection
4. Save the processed weights as weights.npy
"""

from pathlib import Path
from typing import List, Optional

import keras
import numpy as np
import tensorflow as tf

from cnn_surgery.utils.load_dataset import find_project_root


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
            weights_flat = np.concatenate([w.flatten() for w in weights])

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


if __name__ == "__main__":
    # Example usage
    weights_array = flatten_and_aggregate_model_weights()
    print(f"Aggregated weights shape: {weights_array.shape}")
