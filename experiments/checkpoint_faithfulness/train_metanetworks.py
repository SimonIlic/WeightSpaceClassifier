"""
Train meta-networks for the checkpoint faithfulness experiment.

This script trains meta-networks under two conditions:
- final-only: Train on only final checkpoints (metrics_merged_final.csv)
- multi-stage: Train on early+middle+final checkpoints combined

Usage:
    python train_metanetworks.py --dataset mnist --condition multi-stage --seed 42 --output-dir metanetworks
"""

import argparse
import json
import os
from pathlib import Path
from time import time

import torch

from cnn_surgery.lenses.regressor_lens import default_config, get_regressor_lens
from cnn_surgery.utils.load_dataset import load_dataset, load_multi_stage_dataset

DATASETS = ["mnist", "fashion_mnist", "cifar10", "svhn_cropped"]
CONDITIONS = ["final-only", "multi-stage"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]


def train_and_save_metanetwork(
    dataset: str,
    condition: str,
    seed: int,
    output_dir: str,
    device: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Train a meta-network with the specified condition and save it with metrics.

    Args:
        dataset: Dataset name (mnist, fashion_mnist, cifar10, svhn_cropped)
        condition: Training condition ('final-only' or 'multi-stage')
        seed: Random seed for reproducibility
        output_dir: Base directory to save model and metrics
        device: Device to train on (auto-detected if None)
        verbose: Print training progress

    Returns:
        Dictionary containing training metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Training meta-network: {dataset} | {condition} | seed={seed}")
    print(f"{'=' * 60}")

    start_time = time()

    # Load data based on condition
    if condition == "final-only":
        train_data, _, val_data = load_dataset(
            dataset,
            metrics_file="metrics_merged_final.csv",
            load_class_acc=True,
            stage="final",
        )
        weights_train, accuracies_train, _ = train_data
        weights_val, accuracies_val, _ = val_data
        # For final-only, only use per-class accuracies (last 10 columns)
        accuracies_train = accuracies_train[:, -10:]
        accuracies_val = accuracies_val[:, -10:]
    else:  # multi-stage
        data = load_multi_stage_dataset(dataset=dataset)
        weights_train, accuracies_train, _ = data["train"]  # type: ignore
        weights_val, accuracies_val, _ = data["val"]  # type: ignore

    print(f"Training samples: {len(weights_train)}")
    print(f"Validation samples: {len(weights_val)}")

    # Train meta-network
    model, metrics = get_regressor_lens(  # type: ignore
        weights_train,
        accuracies_train,
        weights_val,
        accuracies_val,
        config=default_config,
        return_metrics=True,
        device=device,
        verbose=verbose,
        seed=seed,
    )

    training_time = time() - start_time

    # Unpack metrics: ((mse_train, mae_train), (mse_val, mae_val), r2)
    (mse_train, mae_train), (mse_val, mae_val), r2 = metrics

    # Prepare output paths
    condition_dir = "final_only" if condition == "final-only" else "multi_stage"
    model_dir = Path(output_dir) / condition_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{dataset}_seed{seed}.pt"
    metrics_path = model_dir / f"{dataset}_seed{seed}_metrics.json"

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save metrics
    metrics_dict = {
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "train_samples": len(weights_train),
        "val_samples": len(weights_val),
        "mse_train": float(mse_train),
        "mae_train": float(mae_train),
        "mse_val": float(mse_val),
        "mae_val": float(mae_val),
        "r2_val": float(r2),
        "training_time_seconds": float(training_time),
        "model_path": str(model_path),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return metrics_dict


def train_all_metanetworks(
    datasets: list[str] | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    output_dir: str = "metanetworks",
    device: str | None = None,
    verbose: bool = False,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Train all meta-networks for the experiment.

    Args:
        datasets: List of datasets to train on (default: all)
        conditions: List of conditions (default: both)
        seeds: List of random seeds (default: 5 seeds)
        output_dir: Base directory to save models
        device: Device to train on
        verbose: Print training progress
        skip_existing: Skip if model already exists

    Returns:
        List of training metrics for all models
    """
    datasets = datasets or DATASETS
    conditions = conditions or CONDITIONS
    seeds = seeds or DEFAULT_SEEDS

    all_metrics = []

    for dataset in datasets:
        for condition in conditions:
            for seed in seeds:
                # Check if model already exists
                condition_dir = "final_only" if condition == "final-only" else "multi_stage"
                model_path = Path(output_dir) / condition_dir / f"{dataset}_seed{seed}.pt"

                if skip_existing and model_path.exists():
                    print(f"Skipping existing: {model_path}")
                    continue

                metrics = train_and_save_metanetwork(
                    dataset=dataset,
                    condition=condition,
                    seed=seed,
                    output_dir=output_dir,
                    device=device,
                    verbose=verbose,
                )
                all_metrics.append(metrics)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Train meta-networks for checkpoint faithfulness experiment.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        help="Dataset to train on. If not specified, trains on all datasets.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=CONDITIONS,
        help="Training condition. If not specified, trains both conditions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. If not specified, uses all default seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metanetworks",
        help="Output directory for models and metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (auto-detected if not specified).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training progress.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing models.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 40 meta-networks (4 datasets x 2 conditions x 5 seeds).",
    )

    args = parser.parse_args()

    # Change to script directory for relative paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    if args.all:
        # Train all meta-networks
        train_all_metanetworks(
            output_dir=args.output_dir,
            device=args.device,
            verbose=args.verbose,
            skip_existing=not args.force,
        )
    elif args.dataset and args.condition and args.seed:
        # Train single meta-network
        train_and_save_metanetwork(
            dataset=args.dataset,
            condition=args.condition,
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device,
            verbose=args.verbose,
        )
    else:
        # Train subset based on provided args
        datasets = [args.dataset] if args.dataset else None
        conditions = [args.condition] if args.condition else None
        seeds = [args.seed] if args.seed else None

        train_all_metanetworks(
            datasets=datasets,
            conditions=conditions,
            seeds=seeds,
            output_dir=args.output_dir,
            device=args.device,
            verbose=args.verbose,
            skip_existing=not args.force,
        )


if __name__ == "__main__":
    main()
