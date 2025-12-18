#!/bin/bash
set -euo pipefail

DATASET="fashion_mnist"
NUM_SAMPLES=100
UPDATE_NETS="0 1 2 3 4"
EVAL_NETS="5 6 7 8 9"
LR=0.01
MAX_STEPS=2000
STOP_THRESHOLD=0.3
SEED=0
OUTDIR="experiments/ensemble_unlearning/data/ensemble_holdout"

mkdir -p "$OUTDIR"

for CLASS in {0..9}; do
  python -m scripts.run_pickbest_unlearning \
    --dataset "$DATASET" \
    --target_class "$CLASS" \
    --num_samples "$NUM_SAMPLES" \
    --update_nets $UPDATE_NETS \
    --eval_nets $EVAL_NETS \
    --lr "$LR" \
    --max_steps "$MAX_STEPS" \
    --stop_threshold "$STOP_THRESHOLD" \
    --loss_fn simple \
    --out_csv "$OUTDIR/${DATASET}_class${CLASS}.csv" \
    --seed $((SEED + CLASS)) \
    --min_base_target_acc 0.6
done
