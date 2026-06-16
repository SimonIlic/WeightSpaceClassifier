"""Standalone verification: TF vs PyTorch numerical equivalence.

Run:  python pytorch_port/test_equivalence.py

Requires: tensorflow, cnn_surgery package, and optionally model_zoo data.
If model_zoo/mnist/weights.npy is not found, falls back to synthetic weights.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch

from pytorch_port.evaluate import evaluate_classifier, load_testset_data
from pytorch_port.reconstruct_network import TOTAL_PARAMS, reconstruct_network as pt_reconstruct

LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name} {detail}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {detail}")


def get_test_weights():
    """Try to load real model zoo weights; fall back to synthetic."""
    zoo_path = os.path.join(os.path.dirname(__file__), "..", "model_zoo", "mnist", "weights.npy")
    if os.path.exists(zoo_path):
        weights_all = np.load(zoo_path, mmap_mode="r")
        print(f"Loaded real weights from {zoo_path} (shape {weights_all.shape})")
        return weights_all[0], True
    else:
        print(f"Model zoo not found at {zoo_path}, using synthetic weights")
        np.random.seed(42)
        return np.random.randn(TOTAL_PARAMS).astype(np.float32), False


def test_weight_roundtrip(weights: np.ndarray, activation: str):
    """Verify that PyTorch model parameters match the original flat weights after conversion."""
    from cnn_surgery.utils.reconstruct_network import SHAPES
    from cnn_surgery.utils.process_models import flatten_weights_for_reconstruction
    from math import prod

    pt_model = pt_reconstruct(weights, activation)
    sd = pt_model.state_dict()

    pt_key_map = {
        "sequential/conv2d/bias:0": "conv1.bias",
        "sequential/conv2d/kernel:0": "conv1.weight",
        "sequential/conv2d_1/bias:0": "conv2.bias",
        "sequential/conv2d_1/kernel:0": "conv2.weight",
        "sequential/conv2d_2/bias:0": "conv3.bias",
        "sequential/conv2d_2/kernel:0": "conv3.weight",
        "sequential/dense/bias:0": "dense.bias",
        "sequential/dense/kernel:0": "dense.weight",
    }

    i = 0
    for tf_name, tf_shape in SHAPES.items():
        length = prod(tf_shape)
        flat_arr = weights[i : i + length].reshape(tf_shape)
        i += length

        pt_key = pt_key_map[tf_name]
        pt_val = sd[pt_key].numpy()

        if "kernel" in tf_name and "conv2d" in tf_name:
            expected = np.transpose(flat_arr, (3, 2, 0, 1))
        elif "kernel" in tf_name and "dense" in tf_name:
            expected = flat_arr.T
        else:
            expected = flat_arr

        match = np.allclose(pt_val, expected, atol=1e-7)
        check(f"{activation}/weight_roundtrip/{pt_key}", match)


def test_logits_comparison(weights: np.ndarray, activation: str, input_size: int):
    """Forward pass identical input through both models and compare logits."""
    from cnn_surgery.utils.reconstruct_network import reconstruct_network as tf_reconstruct

    # TF model
    tf_model = tf_reconstruct(weights, activation)
    tf_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # PT model
    pt_model = pt_reconstruct(weights, activation)

    # Same random input
    np.random.seed(123)
    x = np.random.randn(5, input_size, input_size, 1).astype(np.float32)

    tf_logits = tf_model.predict(x, verbose=0)
    with torch.no_grad():
        pt_logits = pt_model(torch.from_numpy(x.transpose(0, 3, 1, 2))).numpy()

    max_diff = float(np.max(np.abs(tf_logits - pt_logits)))
    mean_diff = float(np.mean(np.abs(tf_logits - pt_logits)))
    logits_close = bool(np.allclose(tf_logits, pt_logits, atol=LOGITS_ATOL, rtol=LOGITS_RTOL))
    argmax_match = bool(np.array_equal(np.argmax(tf_logits, axis=1), np.argmax(pt_logits, axis=1)))

    check(
        f"{activation}/logits_{input_size}x{input_size}",
        logits_close,
        f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})",
    )
    check(f"{activation}/argmax_{input_size}x{input_size}", argmax_match)


def test_full_evaluation(weights: np.ndarray, activation: str, has_real_data: bool):
    """End-to-end: reconstruct both models, evaluate on MNIST, compare per-class accuracy."""
    from cnn_surgery.utils.reconstruct_network import reconstruct_network as tf_reconstruct
    from cnn_surgery.utils.evaluate_per_class_accuracy import (
        evaluate_classifier as tf_evaluate,
        load_testset_data as tf_load_testset,
    )

    x_test, y_test = load_testset_data("mnist")

    # PT evaluation
    pt_model = pt_reconstruct(weights, activation)
    pt_overall, pt_per_class = evaluate_classifier(pt_model, x_test, y_test)

    # TF evaluation
    tf_model = tf_reconstruct(weights, activation)
    tf_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    tf_overall, tf_per_class = tf_evaluate(tf_model, x_test, y_test)

    check(
        f"{activation}/overall_accuracy",
        abs(pt_overall - tf_overall) < 0.001,
        f"(PT={pt_overall:.4f}, TF={tf_overall:.4f})",
    )

    for cls in range(10):
        diff = abs(pt_per_class[cls] - tf_per_class[cls])
        check(f"{activation}/class_{cls}_accuracy", diff < 0.001, f"(diff={diff:.4f})")


def main():
    weights, has_real_data = get_test_weights()

    activations = ["relu", "tanh", "sigmoid", "selu"]

    print("\n=== Test 1: Weight round-trip ===")
    for act in activations:
        test_weight_roundtrip(weights, act)

    print("\n=== Test 2: Logits comparison ===")
    for act in activations:
        test_logits_comparison(weights, act, 28)
        test_logits_comparison(weights, act, 32)

    print("\n=== Test 3: Full MNIST evaluation (relu only) ===")
    test_full_evaluation(weights, "relu", has_real_data)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        print("VERIFICATION FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
