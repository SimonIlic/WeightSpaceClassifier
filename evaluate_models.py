import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from cnn_surgery.utils.load_dataset import load_multi_stage_dataset, load_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.metrics import clipped_negative_mean_difference, min_difference, max_difference, target_difference, divergence_corrected_difference

from unlearning import unlearn, simple_loss
import os
# experiment parameters
N_MODELS = 1000  # Number of models to evaluate
TARGET_CLASS = 5
DATASET = 'mnist'

#unlearning parameters
MAX_STEPS = 10000
LR = 0.1
EPS = 0.9
LOSS_FN = simple_loss
L2_PENALTY = 1e-6

# CNN evaluation data
x_test, y_test = load_testset_data(DATASET)

_, _, val_data = load_dataset(dataset=DATASET, metrics_file='metrics_merged_final.csv', load_class_acc=True)
weights_val, metrics_val, config_val = val_data

test_accuracies = np.array([m[0] for m in metrics_val])
accuracies_val = metrics_val[:, -10:]

meta_network = pickle.load(open(f'meta_network_{DATASET}.pkl', 'rb'))
meta_network.eval()

for model_idx in tqdm(range(N_MODELS)):
    network = weights_val[model_idx]
    accuracy = accuracies_val[model_idx]
    config = config_val.iloc[model_idx]

    edited_network, unlearn_metrics = unlearn(network, meta_network, TARGET_CLASS,
                             max_steps=MAX_STEPS, lr=LR, eps=EPS, l2_penalty=L2_PENALTY, loss_fn=LOSS_FN)
    edited_network = edited_network.squeeze(0).detach()
    model = reconstruct_network(edited_network.numpy(), config['config.activation'])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    acc_after = evaluate_classifier(model, x_test, y_test)
    total_accuracy, accuracy_after = acc_after
    
    out_file = 'evaluation_results.csv'
    row = pd.DataFrame([{
        'model_idx': model_idx,
        'original_accuracy': list(accuracy),
        'accuracy_after': accuracy_after,
        'total_accuracy': total_accuracy,
        'target_class': TARGET_CLASS,
        'lr': LR,
        'eps': EPS,
        'max_steps': MAX_STEPS,
        'l2_penalty': L2_PENALTY,
        'loss_fn': LOSS_FN.__name__,
    }])
    row.to_csv(out_file, mode='a', header=not os.path.exists(out_file), index=False)
