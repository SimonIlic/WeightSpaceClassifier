#!/bin/bash
# Evaluate faithfulness for CIFAR-10 dataset
#
# Usage:
#   ./scripts/eval_cifar10.sh           # Parallel by condition (2 jobs)
#   ./scripts/eval_cifar10.sh --serial  # Sequential

cd "$(dirname "$0")/.."

trap 'echo ""; echo "Interrupted. Killing all jobs..."; kill 0' INT TERM

N_MODELS=${N_MODELS:-50}
MAX_STEPS=${MAX_STEPS:-10000}
LR=${LR:-0.1}

DATASET="cifar10"
CONDITIONS="final-only multi-stage"
SEEDS="42 123 456 789 1011"

echo "============================================"
echo "Checkpoint Faithfulness: CIFAR-10 Evaluation"
echo "============================================"
echo "N_MODELS=$N_MODELS, MAX_STEPS=$MAX_STEPS, LR=$LR"
echo ""

run_single_eval() {
    local condition=$1
    local seed=$2

    if [[ "$condition" == "final-only" ]]; then
        condition_dir="final_only"
    else
        condition_dir="multi_stage"
    fi

    model_path="metanetworks/${condition_dir}/${DATASET}_seed${seed}.pt"

    if [[ ! -f "$model_path" ]]; then
        echo "Skipping: $model_path not found"
        return
    fi

    result_path="results/${DATASET}_${condition_dir}_seed${seed}.csv"
    if [[ -f "$result_path" ]]; then
        echo "Skipping: $result_path already exists"
        return
    fi

    echo "Running: $DATASET | $condition | seed=$seed"
    python run_evaluation.py \
        --meta-network-path "$model_path" \
        --dataset "$DATASET" \
        --condition "$condition" \
        --seed "$seed" \
        --n-models "$N_MODELS" \
        --max-steps "$MAX_STEPS" \
        --lr "$LR"
}

if [[ "$1" == "--serial" ]]; then
    echo "Mode: Sequential"
    echo ""
    for condition in $CONDITIONS; do
        for seed in $SEEDS; do
            run_single_eval "$condition" "$seed"
        done
    done
else
    echo "Mode: Parallel by condition (2 jobs)"
    echo ""
    for condition in $CONDITIONS; do
        (
            for seed in $SEEDS; do
                run_single_eval "$condition" "$seed"
            done
        ) &
    done
    wait
fi

echo ""
echo "============================================"
echo "CIFAR-10 evaluation complete!"
echo "============================================"
