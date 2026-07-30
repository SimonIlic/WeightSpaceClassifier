#!/usr/bin/env bash
# Per-class unlearning evaluation for the mnist SANE meta-network.
#
# Must run in the jointenv venv with the SANE package importable, because the
# meta-network is a pickled SANEModelWrapper.
set -euo pipefail

REPO="/Users/ilic/Documents/WSL/WeightSpaceClassifier"
SANE_PKL="/Users/ilic/Documents/WSL/SANE/model_export/meta_network.pkl"
OUT="${1:-sane_mnist_class_evaluation.csv}"
N_MODELS="${2:-500}"
MAX_STEPS="${3:-2000}"

cd "$REPO"
export PYTHONPATH="/Users/ilic/Documents/WSL/SANE"

for cls in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== class ${cls} ($(date +%H:%M:%S)) ==="
    ./jointenv/bin/python -m slimdown.run \
        -c "$cls" \
        -d mnist \
        -n "$N_MODELS" \
        --loss-fn boost \
        --boost-beta 0.1 \
        --stopping-criterium acc_pred_relative \
        --stop-threshold 0.4 \
        --max-steps "$MAX_STEPS" \
        --lr 0.1 \
        --device mps \
        --batch-size 64 \
        --meta-network-path "$SANE_PKL" \
        -o "$OUT"
done

echo "=== done ($(date +%H:%M:%S)) -> ${OUT} ==="
