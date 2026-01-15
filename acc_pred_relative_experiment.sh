dataset='mnist'
for threshold in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
python src/cnn_surgery/evaluate_models.py \
    -c=4 \
    -d=$dataset \
    -n=200 \
    --stopping-criterium='acc_pred_relative' \
    --max-steps=2000 \
    --loss-fn='boost' \
    --stop-threshold=$threshold \
    --meta-network-path="metanetworks/meta_network_${dataset}_0.pkl" \
    -o="${dataset}_relative_acc_evaluation.csv"
done