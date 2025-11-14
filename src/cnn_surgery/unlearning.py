"""
@SIMON: DOESN'T WORK WELL YET. LATEST WORK IS IN notebooks/unlearning.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from jaxtyping import Array, Float
from tqdm import tqdm

from cnn_surgery.lenses.regressor_lens import get_regressor_lens, mse_mae
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier, load_testset_data
from cnn_surgery.utils.load_dataset import load_dataset
from cnn_surgery.utils.reconstruct_network import reconstruct_network

DATASET = "mnist"
METRICS_FILENAME = "metrics_merged.csv"
MODEL_IDX = 16385  # model to unlearn. Will be from test set
STEP_SIZE = 1
TARGET_CLASS = 4
STEPS = 100  # unlearning steps


def test_network_accuracy(weights: np.ndarray | torch.Tensor, activation_fn, dataset=DATASET):
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
    x_test, y_test = load_testset_data(dataset)
    overall_acc, class_accs = evaluate_classifier(CNNModel, x_test, y_test)
    return overall_acc, class_accs


# ----- loss functions -----
simple_loss = lambda pred, target_idx: pred[target_idx]  # minimize the accuracy of the target class


def faithfulness_loss(pred, true, target_idx, beta=1.0):  # needs a better name
    target_term = pred[target_idx]  # we want to minimize this

    # mse = ((true - pred) ** 2).mean()  # we want the regressorlens to remain faithful to the actual performance
    mae = (true - pred).abs().mean()

    # return target_term + mse
    return target_term + beta * mae  # minimize predicted target performance


def boost_loss(pred, target_idx, beta=1.0):
    target_term = pred[target_idx]  # we want to minimize this

    # set target index zero for this term by multiplying with a mask
    mask = torch.ones_like(pred, requires_grad=False)
    mask[target_idx] = 0
    maintain_rest_term = (pred * mask).mean()  # we want to increase or at least maintain the accuracy of the other classes

    return target_term - beta * maintain_rest_term


def unlearning_metric(before_accs: Float[Array, "n"], after_accs: Float[Array, "n"], target_idx: int):
    target_term: float = min((after_accs[target_idx] - before_accs[target_idx]), 0)  # type: ignore

    rest_before_accs = np.delete(before_accs, target_idx)
    rest_after_accs = np.delete(after_accs, target_idx)
    x = np.clip(rest_after_accs - rest_before_accs, a_min=None, a_max=0)
    maintain_rest_term = -1 * np.mean(x)
    return target_term + maintain_rest_term


def unlearn(
    input_weights: np.ndarray | torch.Tensor,
    steps: int,
    lens: nn.Module,
    step_size: float,
    target_class: int,
    og_config: pd.Series,
    beta: float = 1.0,
    save_plots_path: str | None = None,
):
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

    # ------- prerequisite initialisations --------
    print("Starting unlearning procedure")
    diffs_list = []
    loss_list = []
    target_term_list = []
    mse_term_list = []
    mae_term_list = []
    cosine_sim_list = []
    metric_list = []
    maintain_rest_list = []
    before_accs = test_network_accuracy(input_weights, og_config["config.activation"])[1]

    # --------- preparing for optimization --------
    doctored_input_weights = torch.tensor(input_weights, requires_grad=True, dtype=torch.float32)
    lens.eval()
    for param in lens.parameters():
        param.requires_grad = False  # optimize nothing but the INPUT weights

    for i in tqdm(range(steps), desc="Unlearning in progress"):
        # this is expensive: a new model needs to be reconstructed and evaluated at every step
        true = torch.tensor(
            test_network_accuracy(doctored_input_weights.detach().numpy(), og_config["config.activation"])[1],
            dtype=torch.float32,
        )
        pred: torch.Tensor = lens(doctored_input_weights.unsqueeze(0)).squeeze(0)  # forward pass

        loss = boost_loss(pred, target_class, beta=beta)
        loss.backward()  # compute gradients
        gradient = doctored_input_weights.grad.clone()  # type: ignore
        if i == 0:
            first_gradient = gradient.clone()
        # gradients = gradient / (torch.norm(gradient) + 1e-8)  # normalize gradients        gradients = gradient / (torch.norm(gradient) + 1e-8)  # normalize gradients

        ### ACTUAL OPTIMIZATION STEP ###
        with torch.no_grad():
            doctored_input_weights -= step_size * gradient  # gradient step
            doctored_input_weights.grad.zero_()  # zero gradients # type: ignore

        # --------- for analysis  ---------
        target_term: float = pred[target_class].item()
        # mse_term: float = ((true - pred) ** 2).mean().item()
        mae_term: float = (true - pred).abs().mean().item()
        mean_diff = abs((pred.detach().numpy() - np.array(true)).mean())
        metric = unlearning_metric(np.array(before_accs), true.detach().numpy(), target_class)  # type: ignore
        mask = torch.ones_like(pred, requires_grad=False)
        mask[target_class] = 0
        maintain_rest_term = beta * -1 * (pred * mask).sum().item()

        loss_list.append(loss.item())
        target_term_list.append(target_term)
        # mse_term_list.append(mse_term)
        mae_term_list.append(mae_term)
        diffs_list.append(mean_diff)
        cosine_sim_list.append(np.dot(gradient, first_gradient) / (np.linalg.norm(gradient) * np.linalg.norm(first_gradient)))
        metric_list.append(metric)
        maintain_rest_list.append(maintain_rest_term)

    plot(
        target_class=target_class,
        loss_list=loss_list,
        target_term_list=target_term_list,
        maintain_rest_list=maintain_rest_list,
        diffs_list=diffs_list,
        cosine_sim_list=cosine_sim_list,
        metric_list=metric_list,
        before_accs=before_accs,
        actual_accs=true.detach().numpy(),
        preds=pred.detach().numpy(),
        save_path=save_plots_path,
    )

    return (
        doctored_input_weights,
        diffs_list,
        loss_list,
    )


def plot(
    target_class: int,
    loss_list,
    target_term_list,
    maintain_rest_list,
    diffs_list,
    cosine_sim_list,
    metric_list,
    before_accs,
    actual_accs,
    preds,
    save_path=None,
):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f"UNLEARNING CLASS {target_class}")

    # ------- (0,0) loss plotting -------
    ax = axes[0, 0]
    ax.plot(loss_list, label="Total loss")
    ax.plot(target_term_list, label="Target class penalty term")
    ax.plot(maintain_rest_list, label="Maintain rest term")
    # ax.plot(mse_term_list, label="MSE (RegressorLens faithfulness) term")
    # ax.plot(mae_term_list, label="MAE (RegressorLens faithfulness) term")
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over unlearning steps")
    # ax.set_ylim(min(loss_list) * 1.2, max(loss_list) * 1.2)
    ax.legend()

    # ------- (0,1) accuracy -------
    ax = axes[0, 1]
    classes = np.arange(len(preds))
    bar_width = 0.25
    ax.bar(classes - bar_width, before_accs, width=bar_width, label="Before unlearning", alpha=0.7)
    ax.bar(classes, actual_accs, width=bar_width, label="After unlearning", alpha=0.7)
    ax.bar(classes + bar_width, preds, width=bar_width, label="Predicted by RegressorLens", alpha=0.7)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Accuracy")
    ax.set_title("Class-wise accuracies: Before vs Predicted & Actual")
    ax.set_xticks(classes)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    # ------- (1,0) mean prediction error -------
    ax = axes[1, 0]
    ax.plot(diffs_list)
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Mean prediction error")
    ax.set_title("Mean prediction error over unlearning steps")
    ax.set_ylim(0, max(diffs_list) * 1.2 if len(diffs_list) else 1)
    ax.grid(linestyle="--", alpha=0.3)

    # ------- (1,1) gradient cosine similarity -------
    ax = axes[1, 1]
    ax.plot(cosine_sim_list)
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity with first gradient")
    ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.3)

    # ------- (2,0) unlearning metric -------
    ax = axes[2, 0]
    ax.plot(metric_list)
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Unlearning metric")
    ax.set_title("Unlearning metric over unlearning steps")
    ax.set_ylim(min(metric_list) * 1.2, max(metric_list) * 1.2)
    ax.grid(linestyle="--", alpha=0.3)

    # hide the (2,1) subplot (bottom right)
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.show()


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
    # ----- Load dataset and train model -----
    early = load_dataset("mnist", metrics_file="metrics_merged_mnist_early.csv", load_class_acc=True, stage="early")
    middle = load_dataset("mnist", metrics_file="metrics_merged_mnist_middle.csv", load_class_acc=True, stage="middle")
    final = load_dataset("mnist", metrics_file="metrics_merged.csv", load_class_acc=True, stage="final")

    train_early, test_early, val_early = early
    train_middle, test_middle, val_middle = middle
    train_final, test_final, val_final = final

    weights_train = np.concatenate([train_early[0], train_middle[0], train_final[0]])
    weights_val = np.concatenate([val_early[0], val_middle[0], val_final[0]])
    weights_test = np.concatenate([test_early[0], test_middle[0], test_final[0]])

    accuracies_train = np.concatenate([train_early[1][:, -10:], train_middle[1][:, -10:], train_final[1][:, -10:]])
    accuracies_val = np.concatenate([val_early[1][:, -10:], val_middle[1][:, -10:], val_final[1][:, -10:]])
    accuracies_test = np.concatenate([test_early[1][:, -10:], test_middle[1][:, -10:], test_final[1][:, -10:]])

    configs_train = pd.concat([train_early[2], train_middle[2], train_final[2]], ignore_index=True)
    configs_val = pd.concat([val_early[2], val_middle[2], val_final[2]], ignore_index=True)
    configs_test = pd.concat([test_early[2], test_middle[2], test_final[2]], ignore_index=True)

    # all model indices are the same across training stages
    assert all(
        train_early[2].index.map(lambda x: x.split("/")[-2]).values
        == train_middle[2].index.map(lambda x: x.split("/")[-2]).values
    )
    assert all(
        train_early[2].index.map(lambda x: x.split("/")[-2]).values == train_final[2].index.map(lambda x: x.split("/")[-2]).values
    )

    RegressorLens = get_regressor_lens(weights_train, accuracies_train, weights_val, accuracies_val, device="cpu")

    # ----- Unlearning -----
    doctored_weights, diffs, losses = unlearn(
        input_weights=weights_test[MODEL_IDX],
        steps=STEPS,
        lens=RegressorLens,  # type: ignore
        step_size=STEP_SIZE,
        target_class=TARGET_CLASS,
        og_config=configs_test.iloc[MODEL_IDX],
    )

    # # ----- Evaluation -----
    # assert isinstance(RegressorLens, nn.Module), (
    #     f"RegressorLens should be a torch.nn.Module but is {type(RegressorLens)}, perhaps you called get_regressor_lens() with return_metrics=True? In that case it returns a tuple (model, metrics)."
    # )
    # preds = RegressorLens(doctored_weights.unsqueeze(0)).squeeze().detach().numpy()
    # actual_accs = test_network_accuracy(doctored_weights.detach().numpy(), configs_test.iloc[MODEL_IDX]["config.activation"])[1]
    # before_accs = accuracies_test[MODEL_IDX]
    # print(f"Actual accuracies after unlearning: {actual_accs}")
    # print(f"Accuracies before unlearning: {before_accs}")
    # print(f"Predicted accuracies by RegressorLens after unlearning: {preds}")


if __name__ == "__main__":
    main()
