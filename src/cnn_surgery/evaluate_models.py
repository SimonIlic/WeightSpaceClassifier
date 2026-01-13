"""
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

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

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
from cnn_surgery.utils.reconstruct_network import reconstruct_network
from cnn_surgery.baselines import random_vector, finetune_ascent
from cnn_surgery.utils.train_network import get_dataset as get_tf_dataset


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
    elif name == "cosine_similarity":
        return cosine_similarity_stop_factory(derivative=False, eps=1 - stop_threshold)
    elif name == "cosine_similarity_diff":
        return cosine_similarity_stop_factory(derivative=True, eps=1 - stop_threshold)
    elif name == "step":
        if stop_threshold is not None:
            raise ValueError(
                "stop_threshold is not used with 'step' stopping criterium, use max_steps instead."
            )
        return step_stop_factory(max_steps=int(args.max_steps))
    raise ValueError(f"Unsupported stopping criterium: {name}")


def load_meta_network(meta_network_path: str, input_dim: int, n_outputs: int):
    """Load a meta-network from a state dict (.pt) or pickle (.pkl) file."""
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
        metanetwork.load_state_dict(torch.load(meta_network_path, map_location="cpu"))
    elif meta_network_path.endswith(".pkl"):
        with open(meta_network_path, "rb") as meta_file:
            metanetwork = pickle.load(meta_file)
    else:
        raise ValueError(f"Unsupported pickle type: {meta_network_path.split('.')[-1]}")

    metanetwork.eval()
    return metanetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unlearning across multiple models.")
    parser.add_argument(
        "-n",
        "--n-models",
        type=int,
        default=None,
        help="Number of models to evaluate. If None, evaluate all models.",
    )
    parser.add_argument("-c", "--target-class", type=int, help="Class index to unlearn.")
    parser.add_argument("-d","--dataset", type=str, default="mnist", help="Dataset name.", choices=["mnist", "fashion_mnist", "cifar10", "svhn_cropped"])  # fmt: skip
    parser.add_argument("-o", "--output-file", type=str, default="evaluation_results.csv", help="CSV file where the evaluation rows are appended.")  # fmt: skip
    parser.add_argument("--max-steps", type=int, default=10000, help="Max unlearning steps.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for unlearning.")
    parser.add_argument(
        "--stop-threshold", type=float, help="Threshold parameter passed to the stopping criterium."
    )
    parser.add_argument("--l2-penalty", type=float, default=0.0, help="L2 regularisation strength.")
    parser.add_argument(
        "--loss-fn",
        choices=["simple", "boost", "improve"],
        default="simple",
        help="Loss function used during unlearning.",
    )
    parser.add_argument("--boost-beta", type=float, default=0.1, help="Beta parameter for boost loss (only used when --loss-fn=boost).")  # fmt: skip
    parser.add_argument("--stopping-criterium", choices=["acc_pred", "cosine_similarity", "cosine_similarity_diff", "step"], default="acc_pred", help="Stopping criterium to terminate unlearning.",)  # fmt: skip
    parser.add_argument("--meta-network-path", type=str, help="Path to the meta-network file.")  # fmt: skip
    parser.add_argument("--start-idx", type=int, default=0, help="Starting model index (for parallel evaluations).")  # fmt: skip
    parser.add_argument(
        "--weights-set",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which set of weights to use for unlearning (train or val).",
    )
    return parser.parse_args()


def evaluate_network(weights, activation, data, labels):
    model = reconstruct_network(weights, activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    total_accuracy, accuracy_after = evaluate_classifier(model, data, labels)
    return total_accuracy, accuracy_after


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
    train_data, _, val_data = load_dataset(
        dataset=args.dataset, metrics_file=metrics_file, load_class_acc=True
    )
    weights_val, metrics_val, config_val = train_data if args.weights_set == "train" else val_data
    accuracies_val = metrics_val[:, -10:]
    n_models = args.n_models if args.n_models is not None else len(weights_val) - args.start_idx

    metanetwork = load_meta_network(
        meta_network_path,
        input_dim=weights_val.shape[1],
        n_outputs=accuracies_val.shape[1],
    )

    # get dataset for baseline finetune ascent
    ft_dataset = get_tf_dataset(args.dataset, batchsize=512)
    ft_data_tr, data_te, dataset_info = ft_dataset
    ft_data_tr = ft_data_tr.unbatch().filter(lambda x, y: y == args.target_class).batch(512)

    for model_idx in tqdm(range(args.start_idx, args.start_idx + n_models)):
        network = weights_val[model_idx]
        accuracy = accuracies_val[model_idx]
        config = config_val.iloc[model_idx]

        state = unlearn(
            network,
            metanetwork,
            args.target_class,
            max_steps=args.max_steps,
            lr=args.lr,
            l2_penalty=args.l2_penalty,
            loss_fn=loss_fn,
            stopping_criterium=stopping_criterium,
        )
        edited_network = state.weights.squeeze(0).detach()
        # baselines
        rv_weights = random_vector(network, edited_network)
        fa_weights = finetune_ascent(network, config, ft_data_tr, steps=state.step)

        acc_after_edit, per_class_acc_after_edit = evaluate_network(
            edited_network.numpy(), config["config.activation"], x_test, y_test
        )
        acc_after_rv, per_class_acc_after_rv = evaluate_network(
            rv_weights, config["config.activation"], x_test, y_test
        )
        acc_after_fa, per_class_acc_after_fa = evaluate_network(
            fa_weights, config["config.activation"], x_test, y_test
        )

        row = pd.DataFrame(
            [
                {
                    "model_idx": model_idx,
                    "original_accuracy": list(accuracy),
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
                }
            ]
        )
        # this is nice because even if the csv already exists, we can append new models to it
        row.to_csv(
            args.output_file, mode="a", header=not os.path.exists(args.output_file), index=False
        )


if __name__ == "__main__":
    main()
