#!/bin/bash
# Train all meta-networks for the checkpoint faithfulness experiment
# Runs 4 datasets in parallel for ~4x speedup
#
# Usage:
#   ./scripts/train_all.sh          # Train all (parallel by dataset)
#   ./scripts/train_all.sh --serial # Train sequentially (lower memory)

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Checkpoint Faithfulness: Training Phase"
echo "=========================================="
echo "Training 40 models (4 datasets x 2 conditions x 5 seeds)"
echo ""

if [[ "$1" == "--serial" ]]; then
    echo "Mode: Sequential (one dataset at a time)"
    echo ""
    python train_metanetworks.py --all --verbose
else
    echo "Mode: Parallel (4 datasets simultaneously)"
    echo "Note: Use --serial if you run into memory issues"
    echo ""

    # Train each dataset in parallel
    python train_metanetworks.py --dataset mnist &
    PID_MNIST=$!
    python train_metanetworks.py --dataset fashion_mnist &
    PID_FMNIST=$!
    python train_metanetworks.py --dataset cifar10 &
    PID_CIFAR=$!
    python train_metanetworks.py --dataset svhn_cropped &
    PID_SVHN=$!

    echo "Started parallel jobs:"
    echo "  MNIST:         PID $PID_MNIST"
    echo "  Fashion-MNIST: PID $PID_FMNIST"
    echo "  CIFAR-10:      PID $PID_CIFAR"
    echo "  SVHN:          PID $PID_SVHN"
    echo ""
    echo "Waiting for all jobs to complete..."

    # Wait for all jobs
    wait $PID_MNIST $PID_FMNIST $PID_CIFAR $PID_SVHN
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "Models saved to: metanetworks/"
echo "=========================================="
