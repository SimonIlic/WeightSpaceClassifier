#!/bin/bash
set -euo pipefail

DATASET="fashion_mnist"
NUM_SAMPLES=100
UPDATE_NET="0"
EVAL_NET="0"           # self-scored (no hold-out)
LR=0.01
MAX_STEPS=2000
STOP_THRESHOLD=0.3
L2=0
SEED=0
MIN_BASE=0.6
OUTDIR="experiments/ensemble_unlearning/data/single_self"
OUT_PREFIX="${DATASET}_single_self_class"

mkdir -p "$OUTDIR"

for CLASS in {0..9}; do
  python -m scripts.run_pickbest_unlearning \
    --dataset "$DATASET" \
    --target_class "$CLASS" \
    --num_samples "$NUM_SAMPLES" \
    --update_nets $UPDATE_NET \
    --eval_nets $EVAL_NET \
    --lr "$LR" \
    --max_steps "$MAX_STEPS" \
    --stop_threshold "$STOP_THRESHOLD" \
    --l2_penalty "$L2" \
    --loss_fn simple \
    --min_base_target_acc "$MIN_BASE" \
    --out_csv "$OUTDIR/${OUT_PREFIX}${CLASS}.csv" \
    --seed $((SEED + CLASS))
done
