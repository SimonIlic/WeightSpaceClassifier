dataset='fashion_mnist'
for cls in 0 1 2 3 4 5 6 7 8 9; do                                                                                                                                              
python src/cnn_surgery/evaluate_models.py \
    -c=$cls \
    -d=$dataset \
    --stopping-criterium='acc_pred' \
    --max-steps=2000 \
    --loss-fn='simple' \
    --stop-threshold=0.3 \
    --meta-network-path="metanetworks/meta_network_${dataset}_0.pkl" \
    -o="${dataset}_class_evaluation.csv"
done