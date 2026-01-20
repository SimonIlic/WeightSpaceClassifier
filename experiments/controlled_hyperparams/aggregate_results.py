#!/usr/bin/env python3
"""
Aggregate trained CNN results into model zoo format.

After training CNNs with train_controlled_cnns.sh, this script:
1. Loads each .keras checkpoint
2. Flattens weights using _flatten_weights_for_reconstruction()
3. Collects metrics from results.json
4. Evaluates per-class accuracy on test set
5. Saves weights.npy and metrics.csv per dataset

Usage:
    python aggregate_results.py --dataset mnist
    python aggregate_results.py --dataset mnist --delete-checkpoints
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.process_models import flatten_weights_for_reconstruction


def find_checkpoint_file(run_dir: Path) -> Path | None:
    """Find the final checkpoint file in a run directory.

    Looks for permanent_ckpt-{epoch}.keras files and returns the one with highest epoch.
    """
    checkpoint_files = list(run_dir.glob("permanent_ckpt-*.keras"))
    if not checkpoint_files:
        return None

    # Sort by epoch number and return the highest
    def get_epoch(path: Path) -> int:
        # Extract epoch from "permanent_ckpt-86.keras" -> 86
        return int(path.stem.split("-")[1])

    return max(checkpoint_files, key=get_epoch)


def aggregate_dataset(
    dataset: str,
    output_dir: Path,
    delete_checkpoints: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Aggregate all trained CNNs for a dataset into weights.npy and metrics.csv.

    Args:
        dataset: Dataset name (mnist, fashion_mnist, cifar10, svhn_cropped)
        output_dir: Directory containing seed_N subdirectories
        delete_checkpoints: If True, delete .keras files after successful aggregation
        verbose: Print progress information

    Returns:
        Tuple of (weights_array, metrics_dataframe)
    """
    dataset_dir = output_dir / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Find all seed directories
    seed_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")], key=lambda x: int(x.name.split("_")[1])
    )

    if not seed_dirs:
        raise ValueError(f"No seed_* directories found in {dataset_dir}")

    if verbose:
        print(f"Found {len(seed_dirs)} seed directories in {dataset_dir}")

    # Load test data for per-class accuracy evaluation
    if verbose:
        print(f"Loading test data for {dataset}...")
    x_test, y_test = load_testset_data(dataset)

    # Collect weights and metrics
    all_weights: list[np.ndarray] = []
    all_metrics: list[dict] = []
    successful_dirs: list[Path] = []

    for seed_dir in tqdm(seed_dirs, desc=f"Processing {dataset}", disable=not verbose):
        results_path = seed_dir / "results.json"

        # Skip if no results.json
        if not results_path.exists():
            if verbose:
                print(f"  Skipping {seed_dir.name}: no results.json")
            continue

        # Find checkpoint file
        checkpoint_path = find_checkpoint_file(seed_dir)
        if checkpoint_path is None:
            if verbose:
                print(f"  Skipping {seed_dir.name}: no checkpoint file")
            continue

        try:
            # Load model and extract weights
            model = keras.saving.load_model(checkpoint_path)
            assert model is not None, f"Failed to load model from {checkpoint_path}"
            weights = model.get_weights()
            weights_flat = flatten_weights_for_reconstruction(weights)

            # Load results.json
            with open(results_path) as f:
                results = json.load(f)

            # Get final epoch (highest epoch in test_accuracy dict)
            final_epoch = str(max(int(k) for k in results["test_accuracy"].keys()))

            # Compile model for evaluation (from_logits=True because CNN outputs logits)
            model.compile(
                optimizer="adam",
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )

            # Evaluate per-class accuracy
            overall_acc, per_class_acc = evaluate_classifier(model, x_test, y_test)

            # Build metrics dict matching model zoo format
            metrics = {
                "config.activation": results.get("config.activation", "relu"),
                "config.b_init": results.get("config.b_init", "zeros"),
                "config.dataset": results.get("config.dataset", dataset),
                "config.dnn_architecture": results.get("config.dnn_architecture", "cnn"),
                "config.dropout": results.get("config.dropout", 0.0),
                "config.epochs": results.get("config.epochs", 86),
                "config.epochs_between_checkpoints": results.get("config.epochs_between_checkpoints", 20),
                "config.init_std": results.get("config.init_std", 0.05),
                "config.l2reg": results.get("config.l2reg", 0.0),
                "config.learning_rate": results.get("config.learning_rate", 0.01),
                "config.num_layers": results.get("config.num_layers", 3),
                "config.num_units": results.get("config.num_units", 16),
                "config.optimizer": results.get("config.optimizer", "sgd"),
                "config.random_seed": results.get("config.random_seed", 0),
                "config.train_fraction": results.get("config.train_fraction", 1.0),
                "config.w_init": results.get("config.w_init", "glorot_normal"),
                "test_accuracy": results["test_accuracy"][final_epoch],
                "test_loss": results["test_loss"][final_epoch],
                "train_accuracy": results["train_accuracy"][final_epoch],
                "train_loss": results["train_loss"][final_epoch],
                "overall_accuracy": overall_acc,
            }

            # Add per-class accuracies
            for cls_idx, acc in enumerate(per_class_acc):
                metrics[f"accuracy_class_{cls_idx}"] = acc

            all_weights.append(weights_flat)
            all_metrics.append(metrics)
            successful_dirs.append(seed_dir)

        except Exception as e:
            if verbose:
                print(f"  Error processing {seed_dir.name}: {e}")
            continue

    if not all_weights:
        raise ValueError(f"No valid checkpoints processed for {dataset}")

    # Stack weights into 2D array
    weights_array = np.vstack(all_weights)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    if verbose:
        print(f"\nAggregated {len(all_weights)} models")
        print(f"Weights shape: {weights_array.shape}")
        print(f"Metrics columns: {list(metrics_df.columns)}")

    # Save outputs
    weights_path = dataset_dir / "weights.npy"
    metrics_path = dataset_dir / "metrics.csv"

    np.save(weights_path, weights_array)
    metrics_df.to_csv(metrics_path, index=False)

    if verbose:
        print(f"\nSaved weights to: {weights_path}")
        print(f"Saved metrics to: {metrics_path}")

    # Delete checkpoints if requested
    if delete_checkpoints:
        if verbose:
            print(f"\nDeleting .keras checkpoint files from {len(successful_dirs)} directories...")
        for seed_dir in successful_dirs:
            for keras_file in seed_dir.glob("*.keras"):
                keras_file.unlink()
            # Also try to remove the directory if it's now empty (except results.json)
            remaining_files = [f for f in seed_dir.iterdir() if f.name != "results.json"]
            if not remaining_files:
                # Only results.json remains, we could optionally delete the whole dir
                pass
        if verbose:
            print("Checkpoint files deleted.")

    return weights_array, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate trained CNN results into model zoo format")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion_mnist", "cifar10", "svhn_cropped"],
        help="Dataset to aggregate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Base output directory containing dataset subdirectories",
    )
    parser.add_argument(
        "--delete-checkpoints",
        action="store_true",
        help="Delete .keras files after successful aggregation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    weights, _ = aggregate_dataset(
        dataset=args.dataset,
        output_dir=args.output_dir,
        delete_checkpoints=args.delete_checkpoints,
        verbose=not args.quiet,
    )

    print(f"\nDone! Aggregated {len(weights)} models for {args.dataset}")
    print(f"  Weights: {args.output_dir / args.dataset / 'weights.npy'}")
    print(f"  Metrics: {args.output_dir / args.dataset / 'metrics.csv'}")


if __name__ == "__main__":
    main()
