import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from cnn_surgery.utils.load_dataset import load_multi_stage_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from metrics import clipped_negative_mean_difference, min_difference, max_difference

from unlearning import unlearn
import os
# experiment parameters
N_MODELS = 100  # Number of models to evaluate
TARGET_CLASS = 4
DATASET = 'fashion_mnist'

#unlearning parameters
MAX_STEPS = 10000
LR = 0.01
EPS = 0.8

# CNN evaluation data
x_test, y_test = load_testset_data(DATASET)

data = load_multi_stage_dataset(dataset=DATASET)
weights_train, accuracies_train, config_train = data['train']
weights_val, accuracies_val, config_val = data['val']

meta_network = pickle.load(open('meta_network.pkl', 'rb'))
meta_network.eval()

metrics = []
for MODEL_IDX in tqdm(range(-N_MODELS, 0)):
    network = weights_val[MODEL_IDX]
    accuracy = accuracies_val[MODEL_IDX]
    config = config_val.iloc[MODEL_IDX]

    edited_network = unlearn(network, meta_network, TARGET_CLASS,
                             max_steps=MAX_STEPS, lr=LR, eps=EPS, l2_penalty=0.0).squeeze(0).detach()
    model = reconstruct_network(edited_network.numpy(), config['config.activation'])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    acc_after = evaluate_classifier(model, x_test, y_test)
    total_accuracy, accuracy_after = acc_after
    metrics.append(clipped_negative_mean_difference(accuracy, accuracy_after, TARGET_CLASS))
    
    out_file = 'evaluation_results.csv'
    row = pd.DataFrame([{
        'model_idx': len(weights_val) - MODEL_IDX,
        'original_accuracy': accuracy,
        'accuracy_after': accuracy_after,
        'total_accuracy': total_accuracy,
        'clipped_negative_mean_difference': metrics[-1],
        'target_class': TARGET_CLASS,
        'lr': LR,
        'eps': EPS,
        'max_steps': MAX_STEPS
    }])
    row.to_csv(out_file, mode='a', header=not os.path.exists(out_file), index=False)

print(f"Average Clipped Negative Mean Difference over {N_MODELS} models: {np.mean(metrics):.4f}")
