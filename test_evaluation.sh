dataset='fashion_mnist'
                                                                                                                                           
python src/cnn_surgery/evaluate_models.py \
    -c=4 \
    -d=$dataset \
    --stopping-criterium='acc_pred' \
    --n-models=100 \
    --max-steps=2000 \
    --loss-fn='simple' \
    --stop-threshold=0.3 \
    --meta-network-path="metanetworks/meta_network_${dataset}_0.pkl" \
    -o="${dataset}_class_evaluation.csv"
