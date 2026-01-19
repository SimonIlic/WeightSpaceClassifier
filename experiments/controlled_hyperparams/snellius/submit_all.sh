#!/bin/bash
# Submit training and aggregation jobs for all (or selected) datasets
#
# Usage:
#   ./submit_all.sh                     # Submit for all datasets
#   ./submit_all.sh --dataset mnist     # Submit for one dataset only
#   ./submit_all.sh --no-aggregate      # Skip aggregation jobs
#   ./submit_all.sh --delete-checkpoints # Delete .keras files after aggregation
#   ./submit_all.sh --copy-to-home      # Copy final results to home directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
DATASETS=("mnist" "fashion_mnist" "cifar10" "svhn_cropped")
RUN_AGGREGATE=true
DELETE_CHECKPOINTS=0
COPY_TO_HOME=0
ARRAY_SPEC="1-1000%50"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASETS=("$2")
            shift 2
            ;;
        --array)
            ARRAY_SPEC="$2"
            shift 2
            ;;
        --no-aggregate)
            RUN_AGGREGATE=false
            shift
            ;;
        --delete-checkpoints)
            DELETE_CHECKPOINTS=1
            shift
            ;;
        --copy-to-home)
            COPY_TO_HOME=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET       Submit for single dataset only"
            echo "  --array SPEC            SLURM array specification (default: 1-1000%50)"
            echo "  --no-aggregate          Skip aggregation jobs"
            echo "  --delete-checkpoints    Delete .keras files after aggregation"
            echo "  --copy-to-home          Copy results to home directory after aggregation"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                              # All datasets, seeds 1-1000"
            echo "  $0 --dataset mnist              # MNIST only"
            echo "  $0 --array 1-100%20             # Seeds 1-100, max 20 concurrent"
            echo "  $0 --delete-checkpoints         # Clean up after aggregation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Snellius CNN Training Submission"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Array spec: $ARRAY_SPEC"
echo "Run aggregation: $RUN_AGGREGATE"
echo "Delete checkpoints: $DELETE_CHECKPOINTS"
echo "Copy to home: $COPY_TO_HOME"
echo ""

# Track job IDs for summary
declare -A TRAIN_JOBS
declare -A AGGREGATE_JOBS

for dataset in "${DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Submitting jobs for: $dataset"
    echo "----------------------------------------"

    # Submit training job array
    TRAIN_JOB_ID=$(sbatch \
        --parsable \
        --array="$ARRAY_SPEC" \
        --export=DATASET="$dataset" \
        train_job.sh)

    TRAIN_JOBS[$dataset]=$TRAIN_JOB_ID
    echo "  Training job: $TRAIN_JOB_ID (array: $ARRAY_SPEC)"

    # Submit aggregation job with dependency (if requested)
    if [[ "$RUN_AGGREGATE" == "true" ]]; then
        AGGREGATE_JOB_ID=$(sbatch \
            --parsable \
            --dependency=afterok:${TRAIN_JOB_ID} \
            --export=DATASET="$dataset",DELETE_CHECKPOINTS="$DELETE_CHECKPOINTS",COPY_TO_HOME="$COPY_TO_HOME" \
            aggregate_job.sh)

        AGGREGATE_JOBS[$dataset]=$AGGREGATE_JOB_ID
        echo "  Aggregation job: $AGGREGATE_JOB_ID (depends on $TRAIN_JOB_ID)"
    fi

    echo ""
done

# Print summary
echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo ""
echo "Training jobs:"
for dataset in "${DATASETS[@]}"; do
    echo "  $dataset: ${TRAIN_JOBS[$dataset]}"
done
echo ""

if [[ "$RUN_AGGREGATE" == "true" ]]; then
    echo "Aggregation jobs:"
    for dataset in "${DATASETS[@]}"; do
        echo "  $dataset: ${AGGREGATE_JOBS[$dataset]}"
    done
    echo ""
fi

echo "Monitor with: squeue -u \$USER"
echo "Job details:  sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed"
