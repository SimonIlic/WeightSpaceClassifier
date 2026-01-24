dataset='svhn_cropped'
group_size=500

# resume the rest of class 1
cls=1
for idx in $(seq 2500 $group_size 7500); do
    echo "Evaluating class $cls, until index $((idx + group_size))"
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
        -o="${dataset}_class_evaluation_baselines_20260121.csv"
done

# evaluate the remainder of the classes
for cls in 2 3 4 5 6 7 8 9; do 
    for idx in $(seq 0 $group_size 7500); do
        echo "Evaluating class $cls, until index $((idx + group_size))"
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
            -o="${dataset}_class_evaluation_baselines_20260121.csv"
    done
done