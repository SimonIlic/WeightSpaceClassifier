import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from cnn_surgery.utils.load_dataset import load_multi_stage_dataset, load_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.metrics import clipped_negative_mean_difference, min_difference, max_difference, target_difference

from unlearning import unlearn
import os
# experiment parameters
N_MODELS = 1000  # Number of models to evaluate
TARGET_CLASS = 5
DATASET = 'mnist'

#unlearning parameters
MAX_STEPS = 10000
LR = 0.1
EPS = 0.9

# CNN evaluation data
x_test, y_test = load_testset_data(DATASET)

_, _, val_data = load_dataset(dataset=DATASET, metrics_file='metrics_merged_final.csv', load_class_acc=True)
weights_val, metrics_val, config_val = val_data

test_accuracies = np.array([m[0] for m in metrics_val])
accuracies_val = metrics_val[:, -10:]
print(test_accuracies[test_accuracies.argsort()[-N_MODELS:][::-1]])

meta_network = pickle.load(open(f'meta_network_{DATASET}.pkl', 'rb'))
meta_network.eval()

metrics = []
for MODEL_IDX in tqdm(range(N_MODELS)):
    network = weights_val[MODEL_IDX]
    accuracy = accuracies_val[MODEL_IDX]
    config = config_val.iloc[MODEL_IDX]

    if 0.0 in accuracy:
        print("Skipping model with 0 accuracy in some class.")
        continue

    edited_network = unlearn(network, meta_network, TARGET_CLASS,
                             max_steps=MAX_STEPS, lr=LR, eps=EPS, l2_penalty=1e-6).squeeze(0).detach()
    model = reconstruct_network(edited_network.numpy(), config['config.activation'])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    acc_after = evaluate_classifier(model, x_test, y_test)
    total_accuracy, accuracy_after = acc_after
    metrics.append(clipped_negative_mean_difference(accuracy, accuracy_after, TARGET_CLASS, proportional=False))
    
    out_file = 'evaluation_results.csv'
    row = pd.DataFrame([{
        'model_idx': MODEL_IDX,
        'original_accuracy': list(accuracy),
        'accuracy_after': accuracy_after,
        'total_accuracy': total_accuracy,
        'clipped_negative_mean_difference': metrics[-1],
        'max_difference': max_difference(accuracy, accuracy_after, TARGET_CLASS),
        'unlearned_target_difference': target_difference(accuracy, accuracy_after, TARGET_CLASS),
        'target_class': TARGET_CLASS,
        'lr': LR,
        'eps': EPS,
        'max_steps': MAX_STEPS
    }])
    row.to_csv(out_file, mode='a', header=not os.path.exists(out_file), index=False)

print(f"Average Clipped Negative Mean Difference over {N_MODELS} models: {np.mean(metrics):.4f}")
