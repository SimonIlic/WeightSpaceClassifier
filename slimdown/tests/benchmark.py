"""Benchmark: sequential (original) vs batched (slimdown) unlearning + evaluation.

Run from the main repo root:
    PYTHONPATH=src:<worktree> uv run --with torchvision python <worktree>/slimdown/tests/benchmark.py [--n-models 64] [--devices cpu mps]

The original numbers need cnn_surgery importable; pass --skip-original to
benchmark slimdown only.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

TARGET_CLASS = 5
MAX_STEPS = 300
LR = 0.1
THRESHOLD = 0.1


def bench_slimdown(weights, activations, x_test, y_test, device):
    from slimdown.evaluate import evaluate_batch
    from slimdown.nets import load_meta_network
    from slimdown.unlearn import acc_pred_stop_factory, simple_loss, unlearn_batch

    meta = load_meta_network(f"metanetworks/converted/meta_network_mnist_{TARGET_CLASS}.pt", device=device)

    t0 = time.perf_counter()
    state = unlearn_batch(
        weights,
        meta,
        TARGET_CLASS,
        max_steps=MAX_STEPS,
        lr=LR,
        l2_penalty=1e-6,
        loss_fn=simple_loss,
        stopping_criterium=acc_pred_stop_factory(THRESHOLD),
        device=device,
    )
    t_unlearn = time.perf_counter() - t0

    t0 = time.perf_counter()
    evaluate_batch(state.weights.numpy(), activations, x_test, y_test, device=device)
    t_eval = time.perf_counter() - t0
    return t_unlearn, t_eval, state


def bench_original(weights, activations, x_test_tf, y_test_tf):
    from cnn_surgery.evaluate_models import evaluate_network, load_meta_network
    from cnn_surgery.unlearning import acc_pred_stop_factory, simple_loss, unlearn

    meta = load_meta_network(f"metanetworks/meta_network_mnist_{TARGET_CLASS}.pkl", input_dim=4970, n_outputs=10, device="cpu")

    t0 = time.perf_counter()
    states = []
    for i in range(weights.shape[0]):
        states.append(
            unlearn(
                weights[i],
                meta,
                TARGET_CLASS,
                max_steps=MAX_STEPS,
                lr=LR,
                l2_penalty=1e-6,
                loss_fn=simple_loss,
                stopping_criterium=acc_pred_stop_factory(THRESHOLD),
                device="cpu",
            )
        )
    t_unlearn = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i, s in enumerate(states):
        evaluate_network(s.weights.squeeze(0).numpy(), activations[i], x_test_tf, y_test_tf)
    t_eval = time.perf_counter() - t0
    return t_unlearn, t_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=64)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--skip-original", action="store_true")
    args = parser.parse_args()

    from slimdown.data import load_testset_data, load_zoo

    _, _, val = load_zoo("mnist")
    weights = np.array(val[0][: args.n_models])
    activations = val[2]["config.activation"].values[: args.n_models]
    x_test, y_test = load_testset_data("mnist")
    n = args.n_models

    print(f"n_models={n}, max_steps={MAX_STEPS}, acc_pred<{THRESHOLD}\n")

    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print("mps not available, skipping")
            continue
        t_u, t_e, _ = bench_slimdown(weights, activations, x_test, y_test, device)
        print(
            f"slimdown [{device:4s}]  unlearn: {t_u:7.2f}s ({t_u / n * 1000:7.1f} ms/model)   eval: {t_e:6.2f}s ({t_e / n * 1000:6.1f} ms/model)"
        )

    if not args.skip_original:
        from cnn_surgery.utils.evaluate_per_class_accuracy import load_testset_data as tf_load

        x_tf, y_tf = tf_load("mnist")
        t_u, t_e = bench_original(weights, activations, x_tf, y_tf)
        print(
            f"original [cpu ]  unlearn: {t_u:7.2f}s ({t_u / n * 1000:7.1f} ms/model)   eval: {t_e:6.2f}s ({t_e / n * 1000:6.1f} ms/model)"
        )


if __name__ == "__main__":
    main()
