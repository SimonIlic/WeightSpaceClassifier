dataset='svhn_cropped'
group_size=500
idx=0
for cls in 2 3 4 5 6 7 8 9 ; do 
    echo "Evaluating class $cls, until index $((idx + group_size))"
    python src/cnn_surgery/evaluate_models.py \
        -c=$cls \
        -d=$dataset \
        --start-idx=$idx \
        -n=$group_size \
        --stopping-criterium='acc_pred_improve' \
        --max-steps=1500 \
        --loss-fn='improve' \
        --stop-threshold=0.9 \
        --meta-network-path="metanetworks/meta_network_${dataset}_0.pkl" \
        -o="${dataset}_improve_loss.csv" \
        --weights-set='test'
    done
done