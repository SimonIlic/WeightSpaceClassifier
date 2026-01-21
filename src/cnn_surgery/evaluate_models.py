"""
Evaluates CNNs on their own test set
Evaluate unlearning across multiple models from the CNN Zoo.

This script applies the metanetwork-guided unlearning algorithm to models from
the Small CNN Zoo dataset and compares against baseline methods.

Workflow:
    1. Load CNN weights from the model zoo (train or validation split)
    2. Load the trained metanetwork for the target dataset
    3. For each model:
       a. Apply unlearning via gradient descent on metanetwork input
       b. Compute baselines (random vector, finetune ascent)
       c. Evaluate all methods on the test set
    4. Save per-model results to CSV (appends incrementally)

Usage:
    python evaluate_models.py -c 3 -d mnist --meta-network-path meta_network_mnist.pkl
    python evaluate_models.py -c 0 -d fashion_mnist -n 100 --lr 0.05
"""

import argparse
import os
import pickle
import gc

import keras
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from cnn_surgery.lenses.regressor_lens import FCN, default_config
from cnn_surgery.unlearning import (  # fmt: skip
    acc_pred_stop_factory,
    boost_loss_factory,
    cosine_similarity_stop_factory,
    improve_loss,
    simple_loss,
    step_stop_factory,
    unlearn,
)
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.load_dataset import load_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network, reshape_weights, SHAPES
from cnn_surgery.baselines import finetune_retain, random_vector, finetune_ascent
from cnn_surgery.utils.train_network import get_dataset as get_tf_dataset  # FUCKED
from cnn_surgery.utils.benchmark_suite import js_similarity_score


def build_loss_fn(name: str, boost_beta: float):
    """Return the loss function selected by the user."""
    if name == "simple":
        return simple_loss
    if name == "boost":
        return boost_loss_factory(boost_beta)
    if name == "improve":
        return improve_loss
    raise ValueError(f"Unsupported loss function: {name}")


def build_stopping_criterium(name: str, args):
    """Return stopping criterium configured from CLI parameters."""
    stop_threshold = args.stop_threshold
    if name == "acc_pred":
        return acc_pred_stop_factory(stop_threshold)
    elif name == "acc_pred_relative":
        return acc_pred_stop_factory(stop_threshold, relative=True)
    elif name == "cosine_similarity":
        return cosine_similarity_stop_factory(derivative=False, eps=1 - stop_threshold)
    elif name == "cosine_similarity_diff":
        return cosine_similarity_stop_factory(derivative=True, eps=1 - stop_threshold)
    elif name == "step":
        if stop_threshold is not None:
            raise ValueError("stop_threshold is not used with 'step' stopping criterium, use max_steps instead.")
        return step_stop_factory(max_steps=int(args.max_steps))
    raise ValueError(f"Unsupported stopping criterium: {name}")


def get_device():
    """Auto-detect the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_meta_network(meta_network_path: str, input_dim: int, n_outputs: int, device: str | None = None):
    """Load a meta-network from a state dict (.pt) or pickle (.pkl) file.

    Args:
        meta_network_path: Path to the meta-network file
        input_dim: Input dimension for the network
        n_outputs: Number of output classes
        device: Device to load the model to

    Returns:
        Tuple of (metanetwork)
    """

    if meta_network_path.endswith(".pt"):
        metanetwork = FCN(
            input_dim=input_dim,
            n_layers=int(default_config["n_layers"]),
            n_hidden=int(default_config["n_hiddens"]),
            n_outputs=n_outputs,
            dropout_p=float(default_config["dropout_rate"]),
            activation=nn.ReLU,
            last_activation="sigmoid",
        )
        metanetwork.load_state_dict(torch.load(meta_network_path, map_location=device))
    elif meta_network_path.endswith(".pkl"):
        with open(meta_network_path, "rb") as meta_file:
            metanetwork = pickle.load(meta_file)
    else:
        raise ValueError(f"Unsupported pickle type: {meta_network_path.split('.')[-1]}")

    metanetwork = metanetwork.to(device)
    metanetwork.eval()
    return metanetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unlearning across multiple models.")
    parser.add_argument("-n", "--n-models", type=int, default=None, help="Number of models to evaluate. If None, evaluate all models.")  # fmt: skip
    parser.add_argument("-c", "--target-class", type=int, help="Class index to unlearn.")  # fmt: skip
    parser.add_argument("-d","--dataset", type=str, default="mnist", help="Dataset name.", choices=["mnist", "fashion_mnist", "cifar10", "svhn_cropped"])  # fmt: skip
    parser.add_argument("-o", "--output-file", type=str, default="evaluation_results.csv", help="CSV file where the evaluation rows are appended.")  # fmt: skip
    parser.add_argument("--max-steps", type=int, default=2000, help="Max unlearning steps.")  # fmt: skip
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for unlearning.")  # fmt: skip
    parser.add_argument("--stop-threshold", type=float, help="Threshold parameter passed to the stopping criterium.")  # fmt: skip
    parser.add_argument("--l2-penalty", type=float, default=0.0, help="L2 regularisation strength.")  # fmt: skip
    parser.add_argument("--loss-fn", choices=["simple", "boost", "improve"], default="simple", help="Loss function used during unlearning.")  # fmt: skip
    parser.add_argument("--boost-beta", type=float, default=0.1, help="Beta parameter for boost loss (only used when --loss-fn=boost).")  # fmt: skip
    parser.add_argument("--stopping-criterium", choices=["acc_pred", "acc_pred_relative", "cosine_similarity", "cosine_similarity_diff", "step"], default="acc_pred", help="Stopping criterium to terminate unlearning.",)  # fmt: skip
    parser.add_argument("--meta-network-path", type=str, help="Path to the meta-network file.")  # fmt: skip
    parser.add_argument("--start-idx", type=int, default=0, help="Starting model index (for parallel evaluations).")  # fmt: skip
    parser.add_argument("--weights-set", type=str, default="val", choices=["train", "val", "test"], help="Which set of weights to use for unlearning (train or val).")  # fmt: skip
    return parser.parse_args()


def evaluate_network(weights: np.ndarray, activation, data, labels):
    """
    Evaluate a CNN from the flat weights array
    """
    model = reconstruct_network(weights, activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    total_accuracy, accuracy_after = evaluate_classifier(model, data, labels)
    del model
    return total_accuracy, accuracy_after


def evaluate_networks_batch(weight_list: list, activation: str, x_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate multiple weight sets with a single model build/compile.

    This is more efficient than calling evaluate_network multiple times because
    the model architecture is only built and compiled once.

    Args:
        weight_list: List of flat weight arrays (each shape 4970)
        activation: Activation function name
        x_test: Test images
        y_test: Test labels

    Returns:
        List of (overall_acc, per_class_acc) tuples
    """
    # Build and compile model once
    model = reconstruct_network(weight_list[0], activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    results = []
    for weights in weight_list:
        # Just set new weights without rebuilding
        reshaped = reshape_weights(weights, SHAPES)
        model.set_weights(reshaped)
        overall_acc, per_class_acc = evaluate_classifier(model, x_test, y_test)
        results.append((overall_acc, per_class_acc))

    return results


def main():
    args = parse_args()
    loss_fn = build_loss_fn(args.loss_fn, args.boost_beta)
    stopping_criterium = build_stopping_criterium(args.stopping_criterium, args)
    metrics_file = "metrics_merged_final.csv"
    if args.meta_network_path:
        meta_network_path = args.meta_network_path
    else:
        meta_network_path = f"meta_network_{args.dataset}.pkl"

    # test set for CNN evaluation after unlearning (image data, labels)
    x_test, y_test = load_testset_data(args.dataset)

    # models to unlearn (CNN weights, per-class accuracies, config)
    train_data, test_data, val_data = load_dataset(dataset=args.dataset, metrics_file=metrics_file, load_class_acc=True)
    if args.weights_set == "train":
        weights_val, metrics_val, config_val = train_data
    elif args.weights_set == "val":
        weights_val, metrics_val, config_val = val_data
    elif args.weights_set == "test":
        print(f"Using test set for unlearning! This is not recommended for hyperparameter tuning.")
        weights_val, metrics_val, config_val = test_data

    accuracies_val = metrics_val[:, -10:]
    overall_accuracies_val = metrics_val[:, 0]  # test_accuracy column
    n_models = args.n_models if args.n_models is not None else len(weights_val) - args.start_idx

    # device = get_device()
    device = "cpu"

    metanetwork = load_meta_network(
        meta_network_path,
        input_dim=weights_val.shape[1],
        n_outputs=accuracies_val.shape[1],
        device=device,
    )

    # get dataset for baseline finetune ascent
    ft_dataset = get_tf_dataset(args.dataset, batchsize=512)
    ft_data_tr, _, _ = ft_dataset

    # Pre-filter and cache datasets for baselines (avoids repeated unbatch/filter/batch per iteration)
    # .prefetch() overlaps data loading with training; modest benefit here since .cache() already stores data in memory
    forget_data = ft_data_tr.unbatch().filter(lambda x, y: y == args.target_class).batch(512).cache().prefetch(tf.data.AUTOTUNE)
    retain_data = ft_data_tr.unbatch().filter(lambda x, y: y != args.target_class).batch(512).cache().prefetch(tf.data.AUTOTUNE)

    for model_idx in tqdm(range(args.start_idx, args.start_idx + n_models)):
        if model_idx >= len(weights_val):
            print(f"Model index {model_idx} is out of range for weights_val. Stopping.")
            break

        network = weights_val[model_idx]
        accuracy = accuracies_val[model_idx]
        overall_accuracy_before = overall_accuracies_val[model_idx]
        config = config_val.iloc[model_idx]

        # UNLEARNING HAPPENS HERE
        state = unlearn(
            network,
            metanetwork,
            args.target_class,
            max_steps=args.max_steps,
            lr=args.lr,
            l2_penalty=args.l2_penalty,
            loss_fn=loss_fn,
            stopping_criterium=stopping_criterium,
            device=device,
        )

        edited_network = state.weights.squeeze(0).detach()
        # baselines
        rv_weights = random_vector(network, edited_network)
        fa_weights = finetune_ascent(network, config, forget_data, steps=state.step, verbose=False)
        fr_weights = finetune_retain(network, config, retain_data, steps=state.step, verbose=False)

        # Batch evaluate all weight sets with single model build/compile
        eval_results = evaluate_networks_batch(
            [edited_network.numpy(), rv_weights, fa_weights, fr_weights], config["config.activation"], x_test, y_test
        )
        (acc_after_edit, per_class_acc_after_edit) = eval_results[0]
        (acc_after_rv, per_class_acc_after_rv) = eval_results[1]
        (acc_after_fa, per_class_acc_after_fa) = eval_results[2]
        (acc_after_fr, per_class_acc_after_fr) = eval_results[3]

        # js_similarity_edit_rv = js_similarity_score(edited_network.numpy(), rv_weights, config["config.activation"], x_test)
        # js_similarity_edit_fa = js_similarity_score(edited_network.numpy(), fa_weights, config["config.activation"], x_test)
        js_similarity_edit_fr = js_similarity_score(edited_network.numpy(), fr_weights, config["config.activation"], x_test)

        row = pd.DataFrame(
            [
                {
                    "model_idx": model_idx,
                    "original_accuracy": list(accuracy),
                    "original_overall_accuracy": overall_accuracy_before,
                    "accuracy_after": per_class_acc_after_edit,
                    "overall_accuracy": acc_after_edit,
                    "target_class": args.target_class,
                    "dataset": args.dataset,
                    "lr": args.lr,
                    "stop_threshold": args.stop_threshold,
                    "l2_penalty": args.l2_penalty,
                    "loss_fn": args.loss_fn,
                    "stopping_criterium": args.stopping_criterium,
                    "max_steps": args.max_steps,
                    # unlearning state
                    "steps": state.step,
                    "final_loss": state.loss,
                    "distance_travelled": state.distance_travelled,
                    "l2_distance": float(torch.norm(state.weights - torch.tensor(network)).item()),
                    "init_pred": list(state.init_pred.numpy()),
                    "final_pred": list(state.pred.numpy()),
                    "meta_network": meta_network_path,
                    # baselines
                    "accuracy_after_rv": per_class_acc_after_rv,
                    "overall_accuracy_rv": acc_after_rv,
                    "accuracy_after_fa": per_class_acc_after_fa,
                    "overall_accuracy_fa": acc_after_fa,
                    "accuracy_after_fr": per_class_acc_after_fr,
                    "overall_accuracy_fr": acc_after_fr,
                    # "js_similarity_edit_rv": js_similarity_edit_rv,
                    # "js_similarity_edit_fa": js_similarity_edit_fa,
                    "js_similarity_edit_fr": js_similarity_edit_fr,
                }
            ]
        )
        # this is nice because even if the csv already exists, we can append new models to it
        row.to_csv(args.output_file, mode="a", header=not os.path.exists(args.output_file), index=False)


if __name__ == "__main__":
    main()
