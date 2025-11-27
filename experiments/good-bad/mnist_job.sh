for i in {0..4}; do
    for j in {0..9}; do
        python ~/Desktop/snijzaal/WeightSpaceClassifier/evaluate_models.py \
            --n-models 100 \
            --target-class $j \
            --dataset mnist \
            --output-file ~/Desktop/snijzaal/WeightSpaceClassifier/experiments/good-bad/mnist_eval_results_meta_${i}_class_${j}_idx100-200.csv \
            --max-steps 1000 \
            --lr 0.3 \
            --stop-threshold 0.5 \
            --stopping-criterium acc_pred \
            --loss-fn boost \
            --meta-network-path ~/Desktop/snijzaal/WeightSpaceClassifier/models/good_bad_experiment_2/mnist_metanetwork_$i.pt
    done
done