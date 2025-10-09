"""
TO DO:
 - [ ] Unit tests:
    - [ ] targeted_loss()
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from cnn_surgery.lenses.regressor_lens import get_regressor_lens, mse_mae
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.load_dataset import load_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network

DATASET = "mnist"
METRICS_FILENAME = "metrics_merged.csv"
MODEL_IDX = 502  # model to unlearn. Will be from test set
STEP_SIZE = 0.3
TARGET_CLASS = 5
STEPS = 100  # unlearning steps


def test_network_accuracy(weights: np.ndarray | torch.Tensor, activation_fn):
    """
    Reconstructs a network from weights and activation function, evaluates it on the test set
    BEWARE: when using it in unlearning, the input could be a tensor, so be sure to convert it to numpy array first:
        weights = weights.detach().numpy()

    Returns:
        mean accuracy: float
        per class accuracies: list of floats
    """

    if isinstance(weights, torch.Tensor):
        weights = weights.detach().numpy()

    CNNModel = reconstruct_network(weights, activation_fn)
    # this returns a Keras model (Unterthiner code), we have to compile it first
    CNNModel.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    x_test, y_test = load_testset_data(DATASET)
    overall_acc, class_accs = evaluate_classifier(CNNModel, x_test, y_test)
    return overall_acc, class_accs


# ----- loss functions -----
simple_loss = lambda pred, target_idx: pred[target_idx]  # minimize the accuracy of the target class


def targeted_loss(pred, true, target_idx):  # needs a better name
    target_term = pred[target_idx]  # we want to minimize this

    # set target index zero for this term by multiplying with a mask
    mask = torch.ones_like(pred, requires_grad=False)
    mask[target_idx] = 0
    maintain_rest_term = (((true - pred) * mask) ** 2).mean()  # we want to maintain the accuracy of the other classes

    return target_term + maintain_rest_term


def unlearn(input_weights, steps, lens, step_size, target_class, og_config):
    """
    Unlearns the specified target class from a model by optimizing the input weights.
    Args:
        input_weights: np.ndarray, weights of the model to be unlearned
        steps: int - number of optimization steps
        lens: torch.nn.Module - pretrained regressor lens model
        step_size: float - step size for the optimization
        target_class: int - class index to be unlearned
        og_config - original configuration of the model to be unlearned (for reconstructing the model)

    Returns:
        doctored_input_weights: torch.Tensor - the modified weights after unlearning
        diffs_list: list of float - list of mean prediction errors at each step
    """

    print("Starting unlearning procedure")
    diffs_list = []

    # convert input weights to tensor for optimization
    doctored_input_weights = torch.tensor(input_weights, requires_grad=True, dtype=torch.float32)

    lens.eval()

    # optimize only the INPUT weights
    for param in lens.parameters():
        param.requires_grad = False

    for i in tqdm(range(steps), desc="Unlearning in progress"):
        pred: torch.Tensor = lens(doctored_input_weights.unsqueeze(0)).squeeze(0)  # forward pass

        # this is expensive: a new model needs to be reconstructed and evaluated at every step
        true = torch.tensor(
            test_network_accuracy(doctored_input_weights.detach().numpy(), og_config["config.activation"])[1], dtype=torch.float32
        )

        loss = targeted_loss(pred, true, target_class)
        loss.backward()  # compute gradients
        gradients = doctored_input_weights.grad

        with torch.no_grad():
            doctored_input_weights -= step_size * gradients  # gradient step
            doctored_input_weights.grad.zero_()  # zero gradients

        mean_diff = abs((pred.detach().numpy() - np.array(true)).mean())
        diffs_list.append(mean_diff)

    return doctored_input_weights, diffs_list


# ----- plotting -----
def plot_class_accuracies(preds, actual_accs, before_accs):
    """
    args:
        preds: np.array, predicted accuracies by the lens (DETACHED)
    """

    classes = np.arange(len(preds))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(classes - bar_width, before_accs, width=bar_width, label="Before unlearning", alpha=0.7)
    plt.bar(classes, actual_accs, width=bar_width, label="After unlearning", alpha=0.7)
    plt.bar(classes + bar_width, preds, width=bar_width, label="Predicted by RegressorLens", alpha=0.7)

    plt.xlabel("Class index")
    plt.ylabel("Accuracy")
    plt.title("Class-wise accuracies: Before Unlearning vs Predicted & Actual, after Unlearning")
    plt.xticks(classes)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_diffs(diffs):
    plt.figure(figsize=(10, 6))
    plt.plot(diffs)
    plt.xlabel("Unlearning step")
    plt.ylabel("Mean prediction error")
    plt.title("Mean prediction error over unlearning steps")
    plt.ylim(0, max(diffs) * 1.2)
    plt.show()


def main():
    # ----- Load dataset -----
    train, test, val = load_dataset(DATASET, metrics_file=METRICS_FILENAME, load_class_acc=True)
    weights_train, outputs_train, configs_train = train
    weights_test, outputs_test, configs_test = test
    weights_val, outputs_val, configs_val = val

    train_class_accuracies = outputs_train[:, -10:]
    test_class_accuracies = outputs_test[:, -10:]
    val_class_accuracies = outputs_val[:, -10:]

    # ----- Load and train regressor lens -----
    # Currently the get_regressor_lens function trains a new lens from scratch
    # It would be good just use a good and pretrained one once the tuning has been done
    # In that case we could rewrite get_regressor_lens to accept a parameter that decides whether to load a model or initialize and train a new one
    RegressorLens = get_regressor_lens(weights_train, train_class_accuracies, weights_val, val_class_accuracies, device="cpu")

    # ----- Unlearning -----
    print(type(weights_test[MODEL_IDX]))

    doctored_weights, diffs = unlearn(
        input_weights=weights_test[MODEL_IDX],
        steps=STEPS,
        lens=RegressorLens,
        step_size=STEP_SIZE,
        target_class=TARGET_CLASS,
        og_config=configs_test.iloc[MODEL_IDX],
    )  # type: ignore

    # ----- Evaluation -----
    preds = RegressorLens(doctored_weights.unsqueeze(0)).squeeze().detach().numpy()
    actual_accs = test_network_accuracy(doctored_weights.detach().numpy(), configs_test.iloc[MODEL_IDX]["config.activation"])[1]
    before_accs = test_class_accuracies[MODEL_IDX]

    # print(f"Actual accuracies after unlearning: {actual_accs}")
    # print(f"Accuracies before unlearning: {before_accs}")
    # print(f"Predicted accuracies by RegressorLens after unlearning: {preds}")

    plot_class_accuracies(preds, actual_accs, before_accs)
    plot_diffs(diffs)


if __name__ == "__main__":
    main()
