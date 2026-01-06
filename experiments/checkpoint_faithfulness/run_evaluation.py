"""
Run faithfulness evaluation for a single meta-network.

This script evaluates how faithful a meta-network's predictions remain
during unlearning by tracking mean_diff at every step.

Usage:
    python run_evaluation.py --meta-network-path metanetworks/multi_stage/mnist_seed42.pt \
        --dataset mnist --condition multi-stage --seed 42 --n-models 100
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cnn_surgery.lenses.regressor_lens import FCN, default_config
from cnn_surgery.unlearning import unlearn, simple_loss, acc_pred_stop_factory, step_stop_factory
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.load_dataset import load_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network

from evaluate_faithfulness import (
    FaithfulnessCallback,
    FaithfulnessResult,
    create_faithfulness_result,
)


DATASETS = ["mnist", "fashion_mnist", "cifar10", "svhn_cropped"]
CONDITIONS = ["final-only", "multi-stage"]
N_CLASSES = 10


def load_metanetwork(model_path: str, input_dim: int = 4970, n_outputs: int = 10) -> nn.Module:
    """Load a trained meta-network from a .pt file."""
    model = FCN(
        input_dim=input_dim,
        n_layers=int(default_config["n_layers"]),
        n_hidden=int(default_config["n_hiddens"]),
        n_outputs=n_outputs,
        dropout_p=float(default_config["dropout_rate"]),
        activation=nn.ReLU,
        last_activation="sigmoid",
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate_single_model(
    model_idx: int,
    target_class: int,
    weights: np.ndarray,
    original_accuracy: np.ndarray,
    activation: str,
    metanetwork: nn.Module,
    dataset: str,
    condition: str,
    seed: int,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_steps: int = 10000,
    lr: float = 0.1,
) -> FaithfulnessResult:
    """
    Run unlearning on a single model and compute faithfulness metrics.

    Args:
        model_idx: Index of the model in the dataset
        target_class: Class to unlearn
        weights: CNN weights (4970,)
        original_accuracy: Original per-class accuracies (10,)
        activation: Activation function name for reconstruction
        metanetwork: Trained meta-network
        dataset: Dataset name
        condition: Training condition
        seed: Random seed
        x_test: Test images
        y_test: Test labels
        max_steps: Maximum unlearning steps
        lr: Learning rate

    Returns:
        FaithfulnessResult with all metrics
    """
    # Create callback for tracking
    callback = FaithfulnessCallback(original_accuracy)

    # Run unlearning
    state = unlearn(
        weights,
        metanetwork,
        target_class,
        max_steps=max_steps,
        lr=lr,
        loss_fn=simple_loss,
        stopping_criterium=step_stop_factory(max_steps),  # Run for all steps
        l2_penalty=0.0,
        step_callback=callback,
    )

    # Reconstruct and evaluate the unlearned CNN
    edited_weights = state.weights.squeeze(0).detach().numpy()
    model = reconstruct_network(edited_weights, activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    _, accuracy_after = evaluate_classifier(model, x_test, y_test)

    # Compute L2 distance
    l2_distance = float(np.linalg.norm(edited_weights - weights))

    # Create result
    result = create_faithfulness_result(
        model_idx=model_idx,
        target_class=target_class,
        dataset=dataset,
        condition=condition,
        seed=seed,
        original_accuracy=original_accuracy,
        accuracy_after=np.array(accuracy_after),
        init_pred=state.init_pred.numpy(),
        final_pred=state.pred.numpy(),
        callback=callback,
        total_steps=state.step + 1,  # step is 0-indexed
        distance_travelled=state.distance_travelled,
        l2_distance=l2_distance,
    )

    return result


def run_evaluation(
    meta_network_path: str,
    dataset: str,
    condition: str,
    seed: int,
    n_models: int = 100,
    output_dir: str = "results",
    max_steps: int = 10000,
    lr: float = 0.1,
    target_classes: list[int] | None = None,
    start_idx: int = 0,
) -> pd.DataFrame:
    """
    Run faithfulness evaluation for a meta-network.

    Args:
        meta_network_path: Path to the meta-network .pt file
        dataset: Dataset name
        condition: Training condition ('final-only' or 'multi-stage')
        seed: Random seed used for meta-network
        n_models: Number of models to evaluate
        output_dir: Directory to save results
        max_steps: Maximum unlearning steps
        lr: Learning rate for unlearning
        target_classes: List of target classes to evaluate (default: all 10)
        start_idx: Starting model index

    Returns:
        DataFrame with all results
    """
    # Load meta-network
    print(f"Loading meta-network from: {meta_network_path}")
    metanetwork = load_metanetwork(meta_network_path)

    # Load validation CNNs (using final checkpoints as the models to unlearn)
    print(f"Loading validation CNNs for {dataset}...")
    _, _, val_data = load_dataset(
        dataset,
        metrics_file="metrics_merged_final.csv",
        load_class_acc=True,
        stage="final",
    )
    weights_val, metrics_val, config_val = val_data
    accuracies_val = metrics_val[:, -10:]  # Last 10 columns are per-class accuracies

    # Load test data for CNN evaluation
    print(f"Loading test data for {dataset}...")
    x_test, y_test = load_testset_data(dataset)

    # Determine models and classes to evaluate
    target_classes = target_classes or list(range(N_CLASSES))
    end_idx = min(start_idx + n_models, len(weights_val))
    model_indices = range(start_idx, end_idx)

    print(f"Evaluating {len(model_indices)} models x {len(target_classes)} classes = {len(model_indices) * len(target_classes)} runs")

    # Run evaluation
    results = []
    total_runs = len(model_indices) * len(target_classes)

    with tqdm(total=total_runs, desc="Evaluating") as pbar:
        for model_idx in model_indices:
            weights = weights_val[model_idx]
            original_accuracy = accuracies_val[model_idx]
            activation = config_val.iloc[model_idx]["config.activation"]

            for target_class in target_classes:
                result = evaluate_single_model(
                    model_idx=model_idx,
                    target_class=target_class,
                    weights=weights,
                    original_accuracy=original_accuracy,
                    activation=activation,
                    metanetwork=metanetwork,
                    dataset=dataset,
                    condition=condition,
                    seed=seed,
                    x_test=x_test,
                    y_test=y_test,
                    max_steps=max_steps,
                    lr=lr,
                )
                results.append(asdict(result))
                pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    condition_slug = "final_only" if condition == "final-only" else "multi_stage"
    filename = f"{dataset}_{condition_slug}_seed{seed}.csv"
    filepath = output_path / filename

    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run faithfulness evaluation for a meta-network."
    )
    parser.add_argument(
        "--meta-network-path",
        type=str,
        required=True,
        help="Path to the meta-network .pt file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS,
        help="Dataset name.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=CONDITIONS,
        help="Training condition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used for meta-network.",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=100,
        help="Number of models to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum unlearning steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for unlearning.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting model index.",
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="+",
        default=None,
        help="Target classes to evaluate (default: all 10).",
    )

    args = parser.parse_args()

    # Change to script directory for relative paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    run_evaluation(
        meta_network_path=args.meta_network_path,
        dataset=args.dataset,
        condition=args.condition,
        seed=args.seed,
        n_models=args.n_models,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        lr=args.lr,
        target_classes=args.target_classes,
        start_idx=args.start_idx,
    )


if __name__ == "__main__":
    main()
