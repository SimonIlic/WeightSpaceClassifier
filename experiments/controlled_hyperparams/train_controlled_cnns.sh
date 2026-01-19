#!/bin/bash
# Train 1000 CNNs per dataset with optimal hyperparameters and different random seeds (1-1000)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$PROJECT_ROOT/configs/optimal_cnn_hyperparams.json"
OUTPUT_BASE="$SCRIPT_DIR/output"

DATASETS=("mnist" "fashion_mnist" "cifar10" "svhn_cropped")
NUM_SEEDS=1000

# Parse command line arguments
START_SEED=1
END_SEED=$NUM_SEEDS
SELECTED_DATASET=""
AGGREGATE=false
DELETE_CHECKPOINTS=false
TENSORBOARD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            SELECTED_DATASET="$2"
            shift 2
            ;;
        --start-seed)
            START_SEED="$2"
            shift 2
            ;;
        --end-seed)
            END_SEED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --aggregate)
            AGGREGATE=true
            shift
            ;;
        --delete-checkpoints)
            DELETE_CHECKPOINTS=true
            shift
            ;;
        --tensorboard)
            TENSORBOARD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET       Train only on specified dataset (mnist, fashion_mnist, cifar10, svhn_cropped)"
            echo "  --start-seed N          Start from seed N (default: 1)"
            echo "  --end-seed N            End at seed N (default: 1000)"
            echo "  --output-dir DIR        Output directory (default: ./output)"
            echo "  --aggregate             Run aggregation after training to create weights.npy and metrics.csv"
            echo "  --delete-checkpoints    Delete .keras files after successful aggregation (requires --aggregate)"
            echo "  --tensorboard           Enable TensorBoard logging for loss curves"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --dataset mnist --start-seed 1 --end-seed 100"
            echo "  $0 --dataset mnist --aggregate --delete-checkpoints"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Filter datasets if one is specified
if [[ -n "$SELECTED_DATASET" ]]; then
    DATASETS=("$SELECTED_DATASET")
fi

echo "Training CNNs with optimal hyperparameters"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_BASE"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: $START_SEED to $END_SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# actual training
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "=========================================="

    dataset_output="$OUTPUT_BASE/$dataset"
    mkdir -p "$dataset_output"

    for seed in $(seq $START_SEED $END_SEED); do
        workdir="$dataset_output/seed_$seed"

        # Skip if already completed
        if [[ -f "$workdir/results.json" ]]; then
            echo "[$dataset] Seed $seed: already completed, skipping"
            continue
        fi

        echo "[$dataset] Seed $seed: training..."

        TB_FLAG=""
        if [[ "$TENSORBOARD" == "true" ]]; then
            TB_FLAG="--tensorboard"
        fi

        python -m cnn_surgery.utils.train_network \
            --dataset "$dataset" \
            --config "$CONFIG_PATH" \
            --random_seed "$seed" \
            --workdir "$workdir" \
            --nosave_intermediate_checkpoints \
            --verbose 0 \
            $TB_FLAG

        echo "[$dataset] Seed $seed: done"
    done

    echo "[$dataset] Completed all seeds"
    echo ""
done

echo "All training complete!"

# Run aggregation if requested
if [[ "$AGGREGATE" == "true" ]]; then
    echo ""
    echo "=========================================="
    echo "Running aggregation..."
    echo "=========================================="

    AGGREGATE_ARGS="--output-dir $OUTPUT_BASE"
    if [[ "$DELETE_CHECKPOINTS" == "true" ]]; then
        AGGREGATE_ARGS="$AGGREGATE_ARGS --delete-checkpoints"
    fi

    for dataset in "${DATASETS[@]}"; do
        echo "Aggregating $dataset..."
        python "$SCRIPT_DIR/aggregate_results.py" --dataset "$dataset" $AGGREGATE_ARGS
    done

    echo "Aggregation complete!"
fi
