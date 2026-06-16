"""Evaluate 100 final-checkpoint models from MNIST model zoo with PyTorch,
comparing per-class accuracies against recorded metrics.
Uses load_dataset to get final-checkpoint rows (same as the actual pipeline).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

from cnn_surgery.utils.load_dataset import load_dataset
from pytorch_port.evaluate import evaluate_classifier, load_testset_data
from pytorch_port.reconstruct_network import reconstruct_network

N_MODELS = 100
DATASET = "mnist"

train_data, test_data, val_data = load_dataset(DATASET, metrics_file="metrics_merged_final.csv", load_class_acc=True)
weights_all, metrics_all, configs_all = train_data
x_test, y_test = load_testset_data(DATASET)

print(f"Weights shape: {weights_all.shape}")
print(f"Configs shape: {configs_all.shape}")
print(f"Accuracies shape: {metrics_all.shape}")

n = min(N_MODELS, len(weights_all))
print(f"\nEvaluating {n} models...\n")

mismatches = 0
max_diffs = []
for i in range(n):
    activation = configs_all.iloc[i]["config.activation"]
    dropout = configs_all.iloc[i]["config.dropout"]
    l2_penalty = configs_all.iloc[i]["config.l2reg"]

    recorded_overall = metrics_all[i, 0]
    recorded_per_class = metrics_all[i, -10:]

    model = reconstruct_network(weights_all[i], activation=activation, dropout_rate=dropout)
    overall_acc, per_class_acc = evaluate_classifier(model, x_test, y_test)

    diffs = [abs(pt - rec) for pt, rec in zip(per_class_acc, recorded_per_class)]
    max_diff = max(diffs)
    max_diffs.append(max_diff)

    if max_diff >= 0.001:
        mismatches += 1
        print(f"Model {i} ({activation}): MISMATCH max_diff={max_diff:.4f}")
        for c in range(10):
            if diffs[c] >= 0.001:
                print(f"  class {c}: PT={per_class_acc[c]:.4f} rec={recorded_per_class[c]:.4f} diff={diffs[c]:.4f}")

overall_diff = abs(overall_acc - recorded_overall)

print(f"\n{'='*60}")
print(f"Evaluated {n} models")
print(f"Overall accuracy diff: {overall_diff:.6f}")
print(f"Max per-class diff across all models: {max(max_diffs):.6f}")
print(f"Mean max per-class diff: {np.mean(max_diffs):.6f}")
print(f"Models with any class diff >= 0.001: {mismatches}/{n}")
if mismatches == 0:
    print("ALL MODELS MATCH")