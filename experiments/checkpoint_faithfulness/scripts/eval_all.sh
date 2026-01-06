#!/bin/bash
# Run all faithfulness evaluations in parallel
#
# Usage:
#   ./scripts/eval_all.sh              # 8 parallel jobs (by dataset+condition) ~10 hours
#   ./scripts/eval_all.sh --by-dataset # 4 parallel jobs (by dataset) ~20 hours
#   ./scripts/eval_all.sh --full       # 40 parallel jobs (all at once) ~2 hours
#   ./scripts/eval_all.sh --serial     # Sequential ~80 hours

cd "$(dirname "$0")/.."

# Trap Ctrl+C and kill all child processes
trap 'echo ""; echo "Interrupted. Killing all jobs..."; kill 0' INT TERM

N_MODELS=${N_MODELS:-50}
MAX_STEPS=${MAX_STEPS:-10000}
LR=${LR:-0.1}

echo "=========================================="
echo "Checkpoint Faithfulness: Evaluation Phase"
echo "=========================================="
echo "N_MODELS=$N_MODELS, MAX_STEPS=$MAX_STEPS, LR=$LR"
echo ""

DATASETS="mnist fashion_mnist cifar10 svhn_cropped"
CONDITIONS="final-only multi-stage"
SEEDS="42 123 456 789 1011"

run_single_eval() {
    local dataset=$1
    local condition=$2
    local seed=$3

    if [[ "$condition" == "final-only" ]]; then
        condition_dir="final_only"
    else
        condition_dir="multi_stage"
    fi

    model_path="metanetworks/${condition_dir}/${dataset}_seed${seed}.pt"

    if [[ ! -f "$model_path" ]]; then
        echo "Skipping: $model_path not found"
        return
    fi

    result_path="results/${dataset}_${condition_dir}_seed${seed}.csv"
    if [[ -f "$result_path" ]]; then
        echo "Skipping: $result_path already exists"
        return
    fi

    echo "Running: $dataset | $condition | seed=$seed"
    python run_evaluation.py \
        --meta-network-path "$model_path" \
        --dataset "$dataset" \
        --condition "$condition" \
        --seed "$seed" \
        --n-models "$N_MODELS" \
        --max-steps "$MAX_STEPS" \
        --lr "$LR"
}

if [[ "$1" == "--serial" ]]; then
    echo "Mode: Sequential (one at a time) ~80 hours"
    echo ""
    for dataset in $DATASETS; do
        for condition in $CONDITIONS; do
            for seed in $SEEDS; do
                run_single_eval "$dataset" "$condition" "$seed"
            done
        done
    done

elif [[ "$1" == "--by-dataset" ]]; then
    echo "Mode: Parallel by dataset (4 jobs) ~20 hours"
    echo ""
    for dataset in $DATASETS; do
        (
            for condition in $CONDITIONS; do
                for seed in $SEEDS; do
                    run_single_eval "$dataset" "$condition" "$seed"
                done
            done
        ) &
    done
    wait

elif [[ "$1" == "--full" ]]; then
    echo "Mode: Full parallel (40 jobs) ~2 hours"
    echo "WARNING: This uses significant CPU/memory!"
    echo ""
    for dataset in $DATASETS; do
        for condition in $CONDITIONS; do
            for seed in $SEEDS; do
                run_single_eval "$dataset" "$condition" "$seed" &
            done
        done
    done
    wait

else
    # Default: parallel by dataset+condition (8 jobs)
    echo "Mode: Parallel by dataset+condition (8 jobs) ~10 hours"
    echo ""
    for dataset in $DATASETS; do
        for condition in $CONDITIONS; do
            (
                for seed in $SEEDS; do
                    run_single_eval "$dataset" "$condition" "$seed"
                done
            ) &
        done
    done
    wait
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: results/"
echo "=========================================="
