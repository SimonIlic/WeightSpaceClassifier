DATASET=fashion_mnist

for i in {0..4}; do
    python ~/Desktop/snijzaal/WeightSpaceClassifier/evaluate_models.py \
        --n-models 2 \
        --target-class 4 \
        --dataset $DATASET \
        --output-file ~/Desktop/snijzaal/WeightSpaceClassifier/experiments/specific_models/${DATASET}/eval_results_meta_${i}_class_4.csv \
        --max-steps 1000 \
        --lr 0.3 \
        --stop-threshold 0.5 \
        --stopping-criterium acc_pred \
        --loss-fn boost \
        --meta-network-path ~/Desktop/snijzaal/WeightSpaceClassifier/models/good_bad_experiment_2/${DATASET}_metanetwork_$i.pt \
        --start-idx 1121
    done