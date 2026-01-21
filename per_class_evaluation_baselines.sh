dataset='fashion_mnist'
group_size=500
for cls in 0 1 2 3 4 5 6 7 8 9; do 
    for idx in $(seq 0 $group_size 7500); do
        echo "Evaluating class $cls, until index $idx"
        python src/cnn_surgery/evaluate_models.py \
            -c=$cls \
            -d=$dataset \
            --start-idx=$idx \
            -n=$group_size \
            --weights-set='test' \
            --stopping-criterium='acc_pred_relative' \
            --max-steps=2000 \
            --loss-fn='boost' \
            --stop-threshold=0.4 \
            --meta-network-path="metanetworks/meta_network_${dataset}_0.pkl" \
            -o="${dataset}_class_evaluation_baselines_relative_3.csv"
    done
done