dataset='fashion_mnist'
group_size=500
for cls in 0 1; do 
    for idx in $(seq 0 $group_size 1000); do
        echo "Evaluating class $cls, until index $((idx + group_size))"
        KERAS_BACKEND=torch python src/cnn_surgery/evaluate_models.py \
            -c=$cls \
            -d=$dataset \
            --start-idx=$idx \
            -n=$group_size \
            --stopping-criterium='acc_pred_improve' \
            --max-steps=1500 \
            --loss-fn='improve' \
            --stop-threshold=0.9 \
            --meta-network-path="metanetworks/meta_network_${dataset}_0.pkl" \
            -o="${dataset}_class_evaluation_baselines_20260121.csv"
    done
done