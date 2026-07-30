"""Validate a metanetwork's per-class accuracy prediction performance.

Loads a metanetwork (pickled nn.Module or .pt state dict), runs it on the
multi-stage validation set, and reports MSE / MAE / R² — the same metrics
used in notebooks/meta_network_performance.ipynb section 4.1.

Works with both the original FCN metanetworks (metanetworks/*.pkl) and the
SANE-wrapped metanetwork (SANE/model_export/meta_network.pkl).

Usage:
    # SANE metanetwork on MNIST (default)
    PYTHONPATH=/Users/ilic/Documents/WSL/SANE:/Users/ilic/Documents/WSL/WeightSpaceClassifier \
        jointenv/bin/python validate_metanetwork.py

    # Original FCN metanetwork, subsample 500 models for speed
    jointenv/bin/python validate_metanetwork.py \
        --meta-network-path metanetworks/meta_network_mnist_0.pkl \
        --max-samples 500

    # Different dataset
    PYTHONPATH=/Users/ilic/Documents/WSL/SANE:/Users/ilic/Documents/WSL/WeightSpaceClassifier \
        jointenv/bin/python validate_metanetwork.py -d fashion_mnist
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cnn_surgery.lenses.regressor_lens import FCN, default_config
from cnn_surgery.utils.load_dataset import load_dataset, load_multi_stage_dataset


def load_meta_network(path: str, device: str) -> nn.Module:
    """Load a metanetwork from a .pkl (pickle) or .pt (state dict) file."""
    if path.endswith(".pt"):
        net = FCN(
            input_dim=4970,
            n_layers=int(default_config["n_layers"]),
            n_hidden=int(default_config["n_hiddens"]),
            n_outputs=10,
            dropout_p=float(default_config["dropout_rate"]),
            activation=nn.ReLU,
            last_activation="sigmoid",
        )
        net.load_state_dict(torch.load(path, map_location=device))
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            net = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    net = net.to(device)
    net.eval()
    return net


def predict_all(net, weights: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """Run the metanetwork on all weight vectors, return predictions (N, 10)."""
    preds = []
    n = len(weights)
    for start in tqdm(range(0, n, batch_size), desc="Predicting"):
        batch = weights[start : start + batch_size]
        x = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            out = net(x)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute MSE, MAE, R² matching regressor_lens.py / the notebook.

    MSE = mean over samples of (mean over classes of (pred - target)^2)
    MAE = mean over samples of (mean over classes of |pred - target|)
    R²  = 1 - MSE / variance(target)
    """
    per_sample_mse = np.mean((pred - target) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(pred - target), axis=1)

    mse = float(np.mean(per_sample_mse))
    mae = float(np.mean(per_sample_mae))
    var = float(np.mean((target - np.mean(target)) ** 2))
    r2 = 1.0 - mse / var if var > 0 else float("nan")

    per_class_mse = np.mean((pred - target) ** 2, axis=0)
    per_class_mae = np.mean(np.abs(pred - target), axis=0)
    per_class_var = np.var(target, axis=0)
    per_class_r2 = 1.0 - per_class_mse / per_class_var

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "per_class_mse": per_class_mse,
        "per_class_mae": per_class_mae,
        "per_class_r2": per_class_r2,
    }


def print_report(metrics: dict, dataset: str, n_samples: int, split: str):
    print(f"\n{'=' * 50}")
    print(f"  Dataset: {dataset}  |  Split: {split}  |  N: {n_samples}")
    print(f"{'=' * 50}")
    print(f"  MSE  = {metrics['mse']:.6f}")
    print(f"  MAE  = {metrics['mae']:.6f}")
    print(f"  R²   = {metrics['r2']:.6f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':>6}  {'MSE':>10}  {'MAE':>10}  {'R²':>10}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for cls in range(10):
        print(
            f"  {cls:>6}  "
            f"{metrics['per_class_mse'][cls]:>10.6f}  "
            f"{metrics['per_class_mae'][cls]:>10.6f}  "
            f"{metrics['per_class_r2'][cls]:>10.6f}"
        )
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Validate metanetwork accuracy prediction.")
    parser.add_argument(
        "--meta-network-path", "-m",
        type=str,
        default="/Users/ilic/Documents/WSL/SANE/model_export/meta_network.pkl",
        help="Path to the metanetwork file (.pkl or .pt).",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10", "svhn_cropped"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Subsample N models from val set (None = use all).",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for forward passes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cpu', 'mps', 'cuda'). Auto-detected if None.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip evaluating on the train split (faster).",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Use only final-stage checkpoints (not the multi-stage early+middle+final concat).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Loading dataset: {args.dataset}")
    if args.final_only:
        print("  (final-stage checkpoints only)")
        train_data, test_data, val_data = load_dataset(
            dataset=args.dataset,
            metrics_file="metrics_merged_final.csv",
            load_class_acc=True,
            stage="final",
        )
        weights_train, metrics_train, _ = train_data
        weights_val, metrics_val, _ = val_data
        accuracies_train = metrics_train[:, -10:]
        accuracies_val = metrics_val[:, -10:]
    else:
        data = load_multi_stage_dataset(dataset=args.dataset)
        weights_val, accuracies_val, _ = data["val"]
        weights_train, accuracies_train, _ = data["train"]

    if args.max_samples is not None and args.max_samples < len(weights_val):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(weights_val), size=args.max_samples, replace=False)
        weights_val = weights_val[idx]
        accuracies_val = accuracies_val[idx]

    print(f"Val set: {len(weights_val)} models")
    if not args.no_train:
        print(f"Train set: {len(weights_train)} models (will also evaluate)")

    print(f"Loading metanetwork: {args.meta_network_path}")
    net = load_meta_network(args.meta_network_path, device)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  type: {type(net).__name__}, params: {n_params:,}")
    print(f"  device: {device}")

    print("\n--- Validation set ---")
    pred_val = predict_all(net, weights_val, device, args.batch_size)
    metrics_val = compute_metrics(pred_val, accuracies_val)
    print_report(metrics_val, args.dataset, len(weights_val), "val")

    if not args.no_train:
        print("--- Train set ---")
        pred_train = predict_all(net, weights_train, device, args.batch_size)
        metrics_train = compute_metrics(pred_train, accuracies_train)
        print_report(metrics_train, args.dataset, len(weights_train), "train")

    print("=" * 50)
    print("SUMMARY (val set):")
    print(f"  MSE = {metrics_val['mse']:.6f}  |  MAE = {metrics_val['mae']:.6f}  |  R² = {metrics_val['r2']:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
