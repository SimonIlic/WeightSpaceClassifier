import argparse
import os
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from cnn_surgery.utils.load_dataset import load_multi_stage_dataset
from unlearning import (
    acc_pred_stop_factory,
    boost_loss_factory,
    improve_loss,
    simple_loss,
    unlearn,
)


def build_loss_fn(name: str, boost_beta: float):
    if name == "simple":
        return simple_loss
    if name == "boost":
        return boost_loss_factory(boost_beta)
    if name == "improve":
        return improve_loss
    raise ValueError(f"Unsupported loss function: {name}")


def load_meta_networks(dataset: str, ids: Sequence[int]):
    nets = {}
    for net_id in ids:
        path = Path("metanetworks") / f"meta_network_{dataset}_{net_id}.pkl"
        with open(path, "rb") as f:
            net = pickle.load(f)
        net.eval()
        for param in net.parameters():
            param.requires_grad = False
        nets[net_id] = net
    return nets


def evaluate_holdout(meta_nets: dict[int, torch.nn.Module], weights: torch.Tensor):
    with torch.no_grad():
        preds = [net(weights.unsqueeze(0)).squeeze(0) for net in meta_nets.values()]
    return torch.stack(preds)


def sample_indices(accuracies: np.ndarray, target_class: int, num_samples: int, seed: int, min_base_acc: float | None):
    rng = np.random.default_rng(seed)
    candidates = np.where(accuracies[:, target_class] >= (min_base_acc if min_base_acc is not None else -np.inf))[0]
    if len(candidates) == 0:
        raise ValueError(f"No candidates found for class {target_class} with min_base_acc={min_base_acc}")
    if len(candidates) < num_samples:
        return rng.choice(candidates, size=len(candidates), replace=False)
    return rng.choice(candidates, size=num_samples, replace=False)


def run_pick_best(
    weights: np.ndarray,
    base_acc: np.ndarray,
    target_class: int,
    update_nets: dict[int, torch.nn.Module],
    eval_nets: dict[int, torch.nn.Module],
    args,
):
    stopping = acc_pred_stop_factory(args.stop_threshold)
    loss_fn = build_loss_fn(args.loss_fn, args.boost_beta)

    best_score = float("inf")
    best_state = None
    best_eval_mean = None
    winning_net = None

    for net_id, meta_net in update_nets.items():
        state = unlearn(
            weights,
            meta_net,
            target_class,
            max_steps=args.max_steps,
            lr=args.lr,
            loss_fn=loss_fn,
            stopping_criterium=stopping,
            l2_penalty=args.l2_penalty,
        )

        holdout_preds = evaluate_holdout(eval_nets, state.weights)
        holdout_mean = holdout_preds.mean(dim=0).detach().cpu().numpy()
        target_score = holdout_mean[target_class]

        if target_score < best_score:
            best_score = target_score
            best_state = state
            best_eval_mean = holdout_mean
            winning_net = net_id

    if best_state is None or best_eval_mean is None or winning_net is None:
        raise RuntimeError("No best state selected; check meta-network inputs")

    base_non_target = np.delete(base_acc, target_class).mean()
    best_non_target = np.delete(best_eval_mean, target_class).mean()

    return {
        "base_target_acc": float(base_acc[target_class]),
        "best_target_acc": float(best_eval_mean[target_class]),
        "target_drop": float(base_acc[target_class] - best_eval_mean[target_class]),
        "nontarget_drop": float(base_non_target - best_non_target),
        "distance": float(best_state.distance_travelled),
        "steps": int(best_state.step),
        "winning_net_id": int(winning_net),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Pick-best unlearning across meta-network ensemble.")
    parser.add_argument("--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist", "cifar10", "svhn_cropped"], help="Dataset name.")
    parser.add_argument("--target_class", type=int, required=True, help="Target class to unlearn.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of models to sample for this class.")
    parser.add_argument("--update_nets", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Meta-net IDs used for weight updates.")
    parser.add_argument("--eval_nets", type=int, nargs="+", default=[5, 6, 7, 8, 9], help="Hold-out meta-net IDs used for evaluation.")
    parser.add_argument("--max_steps", type=int, default=60, help="Maximum unlearning steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--stop_threshold", type=float, default=0.2, help="Stopping threshold for acc_pred criterium.")
    parser.add_argument("--l2_penalty", type=float, default=1e-6, help="L2 regularization strength.")
    parser.add_argument("--loss_fn", choices=["simple", "boost", "improve"], default="simple", help="Loss function during unlearning.")
    parser.add_argument("--boost_beta", type=float, default=0.1, help="Beta for boost loss.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--min_base_target_acc", type=float, default=None, help="Filter models with base target accuracy below this value.")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to output CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    data = load_multi_stage_dataset(dataset=args.dataset)
    weights_val, accuracies_val, _ = data["val"]

    update_nets = load_meta_networks(args.dataset, args.update_nets)
    eval_nets = load_meta_networks(args.dataset, args.eval_nets)

    indices = sample_indices(
        accuracies_val,
        args.target_class,
        args.num_samples,
        seed=args.seed,
        min_base_acc=args.min_base_target_acc,
    )

    rows = []
    for sample_idx, model_idx in enumerate(indices):
        row = run_pick_best(
            weights_val[model_idx],
            accuracies_val[model_idx],
            args.target_class,
            update_nets,
            eval_nets,
            args,
        )
        row.update(
            {
                "sample_idx": sample_idx,
                "model_idx": int(model_idx),
                "dataset": args.dataset,
                "target_class": args.target_class,
                "update_nets": " ".join(map(str, args.update_nets)),
                "eval_nets": " ".join(map(str, args.eval_nets)),
                "lr": args.lr,
                "max_steps": args.max_steps,
                "stop_threshold": args.stop_threshold,
                "l2_penalty": args.l2_penalty,
                "loss_fn": args.loss_fn,
                "boost_beta": args.boost_beta,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    main()

