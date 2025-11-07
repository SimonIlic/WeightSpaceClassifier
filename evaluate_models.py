import numpy as np
import pickle
from tqdm import tqdm
from cnn_surgery.utils.load_dataset import load_multi_stage_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from metrics import clipped_negative_mean_difference, min_difference, max_difference

from unlearning import unlearn

N_MODELS = 100  # Number of models to evaluate
TARGET_CLASS = 4

# CNN evaluation data
x_test, y_test = load_testset_data('mnist')

data = load_multi_stage_dataset(dataset='fashion_mnist')
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
                             max_steps=10**5, lr=0.01, eps=0.01).squeeze(0).detach()
    model = reconstruct_network(edited_network.numpy(), config['config.activation'])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    acc_after = evaluate_classifier(model, x_test, y_test)
    total_accuracy, accuracy_after = acc_after
    metrics.append(clipped_negative_mean_difference(accuracy, accuracy_after, TARGET_CLASS))

print(f"Average Clipped Negative Mean Difference over {N_MODELS} models: {np.mean(metrics):.4f}")