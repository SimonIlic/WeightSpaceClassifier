"""Batched unlearning evaluation across CNN Zoo models.

Slim, TF-free replacement for cnn_surgery.evaluate_models: models are unlearned
and evaluated in batches instead of one-by-one. Baselines are out of scope for
now; the CSV keeps the same column names as the original for the shared
(unlearning-core) columns.

Usage:
    python -m slimdown.run -d mnist -c 5 --max-steps 10000 --lr 0.1 \
        --loss-fn simple --stopping-criterium acc_pred --stop-threshold 0.1 \
        --meta-network-path metanetworks/converted/meta_network_mnist_5.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from slimdown.data import load_testset_data, load_zoo
from slimdown.evaluate import evaluate_batch
from slimdown.nets import load_meta_network
from slimdown.unlearn import (
    acc_pred_stop_factory,
    boost_loss_factory,
    cosine_similarity_stop_factory,
    improve_loss,
    simple_loss,
    step_stop_factory,
    unlearn_batch,
)


def build_loss_fn(name: str, boost_beta: float):
    if name == "simple":
        return simple_loss
    if name == "boost":
        return boost_loss_factory(boost_beta)
    if name == "improve":
        return improve_loss
    raise ValueError(f"Unsupported loss function: {name}")


def build_stopping_criterium(name: str, args):
    stop_threshold = args.stop_threshold
    if name == "acc_pred":
        return acc_pred_stop_factory(stop_threshold)
    elif name == "acc_pred_relative":
        return acc_pred_stop_factory(stop_threshold, relative=True)
    elif name == "acc_pred_improve":
        return acc_pred_stop_factory(stop_threshold, improve=True)
    elif name == "cosine_similarity":
        return cosine_similarity_stop_factory(derivative=False, eps=1 - stop_threshold)
    elif name == "cosine_similarity_diff":
        return cosine_similarity_stop_factory(derivative=True, eps=1 - stop_threshold)
    elif name == "step":
        if stop_threshold is not None:
            raise ValueError("stop_threshold is not used with 'step' stopping criterium, use max_steps instead.")
        return step_stop_factory(max_steps=int(args.max_steps))
    raise ValueError(f"Unsupported stopping criterium: {name}")


def get_device(requested: str | None = None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Batched unlearning evaluation (slimdown).")
    parser.add_argument("-n", "--n-models", type=int, default=None, help="Number of models to evaluate. If None, evaluate all models.")  # fmt: skip
    parser.add_argument("-c", "--target-class", type=int, required=True, help="Class index to unlearn.")  # fmt: skip
    parser.add_argument("-d", "--dataset", type=str, default="mnist", help="Dataset name.", choices=["mnist", "fashion_mnist", "cifar10", "svhn_cropped"])  # fmt: skip
    parser.add_argument("-o", "--output-file", type=str, default="evaluation_results.csv", help="CSV file where the evaluation rows are appended.")  # fmt: skip
    parser.add_argument("--max-steps", type=int, default=2000, help="Max unlearning steps.")  # fmt: skip
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for unlearning.")  # fmt: skip
    parser.add_argument("--stop-threshold", type=float, help="Threshold parameter passed to the stopping criterium.")  # fmt: skip
    parser.add_argument("--l2-penalty", type=float, default=0.0, help="L2 regularisation strength.")  # fmt: skip
    parser.add_argument("--loss-fn", choices=["simple", "boost", "improve"], default="simple", help="Loss function used during unlearning.")  # fmt: skip
    parser.add_argument("--boost-beta", type=float, default=0.1, help="Beta parameter for boost loss (only used when --loss-fn=boost).")  # fmt: skip
    parser.add_argument("--stopping-criterium", choices=["acc_pred", "acc_pred_relative", "acc_pred_improve", "cosine_similarity", "cosine_similarity_diff", "step"], default="acc_pred", help="Stopping criterium to terminate unlearning.")  # fmt: skip
    parser.add_argument("--meta-network-path", type=str, default=None, help="Path to a converted meta-network .pt file. Defaults to metanetworks/converted/meta_network_{dataset}_{class}.pt")  # fmt: skip
    parser.add_argument("--start-idx", type=int, default=0, help="Starting model index.")  # fmt: skip
    parser.add_argument("--weights-set", type=str, default="val", choices=["train", "val", "test"], help="Which zoo split to unlearn.")  # fmt: skip
    parser.add_argument("--batch-size", type=int, default=None, help="Models per batch. Default: 256 on CUDA, 64 otherwise.")  # fmt: skip
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"], help="Device. Default: auto (cuda > mps > cpu).")  # fmt: skip
    parser.add_argument("--zoo-dir", type=str, default=None, help="Path to the model_zoo directory. Default: search upward from slimdown/ and cwd.")  # fmt: skip
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    batch_size = args.batch_size if args.batch_size is not None else (256 if device == "cuda" else 64)
    loss_fn = build_loss_fn(args.loss_fn, args.boost_beta)
    stopping_criterium = build_stopping_criterium(args.stopping_criterium, args)

    meta_network_path = args.meta_network_path or f"metanetworks/converted/meta_network_{args.dataset}_{args.target_class}.pt"
    metanetwork = load_meta_network(meta_network_path, device=device)

    x_test, y_test = load_testset_data(args.dataset)

    train_data, test_data, val_data = load_zoo(args.dataset, zoo_dir=args.zoo_dir)
    if args.weights_set == "test":
        print("Using test set for unlearning! This is not recommended for hyperparameter tuning.")
    weights_all, metrics_all, config_all = {"train": train_data, "val": val_data, "test": test_data}[args.weights_set]

    accuracies_all = metrics_all[:, -10:]
    overall_accuracies_all = metrics_all[:, 0]  # test_accuracy column
    end_idx = len(weights_all) if args.n_models is None else min(args.start_idx + args.n_models, len(weights_all))

    print(f"device={device}, batch_size={batch_size}, models {args.start_idx}..{end_idx - 1}")

    for batch_start in tqdm(range(args.start_idx, end_idx, batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, end_idx)
        idx = np.arange(batch_start, batch_end)

        # UNLEARNING HAPPENS HERE (whole batch at once)
        state = unlearn_batch(
            weights_all[idx],
            metanetwork,
            args.target_class,
            max_steps=args.max_steps,
            lr=args.lr,
            l2_penalty=args.l2_penalty,
            loss_fn=loss_fn,
            stopping_criterium=stopping_criterium,
            device=device,
        )

        edited = state.weights.numpy()
        activations = config_all["config.activation"].values[idx]
        overall_after, per_class_after = evaluate_batch(edited, activations, x_test, y_test, device=device)

        l2_distance = np.linalg.norm(edited - weights_all[idx], axis=1)

        rows = pd.DataFrame(
            [
                {
                    "model_idx": int(model_idx),
                    "original_accuracy": [float(a) for a in accuracies_all[model_idx]],
                    "original_overall_accuracy": float(overall_accuracies_all[model_idx]),
                    "accuracy_after": [float(a) for a in per_class_after[i]],
                    "overall_accuracy": float(overall_after[i]),
                    "target_class": args.target_class,
                    "dataset": args.dataset,
                    "lr": args.lr,
                    "stop_threshold": args.stop_threshold,
                    "l2_penalty": args.l2_penalty,
                    "loss_fn": args.loss_fn,
                    "stopping_criterium": args.stopping_criterium,
                    "max_steps": args.max_steps,
                    # unlearning state
                    "steps": int(state.steps[i]),
                    "final_loss": float(state.loss[i]),
                    "distance_travelled": float(state.distance_travelled[i]),
                    "l2_distance": float(l2_distance[i]),
                    "init_pred": [float(p) for p in state.init_pred[i]],
                    "final_pred": [float(p) for p in state.pred[i]],
                    "meta_network": meta_network_path,
                }
                for i, model_idx in enumerate(idx)
            ]
        )
        rows.to_csv(args.output_file, mode="a", header=not os.path.exists(args.output_file), index=False)


if __name__ == "__main__":
    main()
