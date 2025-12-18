#!/bin/bash
set -euo pipefail

DATASET="fashion_mnist"
NUM_SAMPLES=100
UPDATE_NETS="0 1 2 3 4"   # also used for scoring (no hold-out)
EVAL_NETS="0 1 2 3 4"
LR=0.01
MAX_STEPS=2000
STOP_THRESHOLD=0.3
L2=0
SEED=0
MIN_BASE=0.6
OUTDIR="experiments/ensemble_unlearning/data/ensemble_noholdout"
OUT_PREFIX="${DATASET}_ensemble_nohold_class"

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
    --l2_penalty "$L2" \
    --loss_fn simple \
    --min_base_target_acc "$MIN_BASE" \
    --out_csv "$OUTDIR/${OUT_PREFIX}${CLASS}.csv" \
    --seed $((SEED + CLASS))
done
