# unlearning algorithm as proposed in "Amazing Paper Title" by Moos & Simon 2026"
import torch as th
import numpy as np
from dataclasses import dataclass


def torch_cosine_similarity(a: th.Tensor, b: th.Tensor) -> float:
    """Compute cosine similarity between two tensors using PyTorch (faster than sklearn)."""
    return th.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


@dataclass
class UnlearnState:
    """Defines a state during unlearning process. Contains anything useful for stopping criterium or logging and metrics."""

    step: int
    target_class: int
    weights: th.Tensor
    pred: th.Tensor
    loss: float
    grads: list
    distance_travelled: float
    init_pred: th.Tensor


def boost_loss_factory(beta=0.1):
    """Return a boost_loss(pred, target_class) function with given beta."""

    def boost_loss(pred, target_class):
        """Encourages unlearning of a specific class, whilst boosting accuracy on other classes."""
        weighting = -th.ones_like(pred) * beta  # all classes go up
        weighting[target_class] = 1  # target class goes down
        return (weighting * pred).sum()

    return boost_loss


# default boost_loss kept for backward compatibility
boost_loss = boost_loss_factory()


def simple_loss(pred, target_class):
    """Simple loss to reduce accuracy on target class."""
    return pred[target_class]


def improve_loss(pred, target_class):
    """Simple loss to improve accuracy on target class."""
    return -pred[target_class]


def l2_regularisation(weights):
    """L2 regularization on model weights."""
    return th.sum(weights**2)


def acc_pred_stop_factory(threshold=0.1, relative=False):
    """Return a stopping function that checks predicted accuracy for target class below threshold."""
    if relative:

        def acc_pred_stop(state: UnlearnState):
            return state.pred[state.target_class] < state.init_pred[state.target_class] * threshold
    else:

        def acc_pred_stop(state: UnlearnState):
            return state.pred[state.target_class] < threshold

    return acc_pred_stop


# backward-compatible default
acc_pred_stop = acc_pred_stop_factory()


def cosine_similarity_stop_factory(derivative=False, eps=1e-2):
    def cosine_similarity_stop(state: UnlearnState):
        grads = state.grads
        if len(grads) < 2:
            return False
        # Use PyTorch cosine similarity (faster than sklearn)
        return torch_cosine_similarity(grads[-1], grads[-2 if derivative else 0]) < 1 - eps

    return cosine_similarity_stop


def step_stop_factory(max_steps=100):
    def step_stop(state: UnlearnState):
        return state.step >= max_steps

    return step_stop


def unlearn(
    model_weights,
    meta_network: th.nn.Module,
    target_class,
    max_steps=100,
    lr=0.01,
    loss_fn=boost_loss,
    stopping_criterium=acc_pred_stop,
    l2_penalty=1e-6,
    step_callback=None,
    device=None,
    store_grads=None,
):
    """
    Unlearn a target class from the model using a meta-network to guide weight updates.

    Parameters:
    - model_weights: Initial weights of the model to be edited.
    - meta_network: A network that predicts per-class model performance from model parameters.
    - target_class: The class index to be unlearned.
    - max_steps: Maximum number of unlearning steps.
    - lr: Learning rate (step-size) for weight updates.
    - l2_penalty: L2 regularization penalty.
    - step_callback: Optional callable(step, pred, weights) called at each step for tracking.
    - device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
    - store_grads: Whether to store gradient history. If None, auto-detect based on stopping_criterium.

    Returns:
    - Updated model weights after unlearning. (as tensor), metrics
    """
    # Auto-detect device
    if device is None:
        if th.backends.mps.is_available():
            device = "mps"
        elif th.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Auto-detect whether we need gradient history (for cosine_similarity stopping)
    if store_grads is None:
        # Check if stopping criterium needs gradients by inspecting its name
        store_grads = "cosine_similarity" in str(stopping_criterium)

    # Move tensors and model to device
    weights = th.tensor(model_weights, requires_grad=True, device=device)
    meta_network = meta_network.to(device)

    grads = []
    distance_travelled = 0.0
    initial_prediction = meta_network(weights.unsqueeze(0)).squeeze(0).detach().clone()

    for step in range(max_steps):
        # forward pass through meta-network
        acc_pred = meta_network(weights.unsqueeze(0)).squeeze(0)

        # call step callback for tracking (e.g., faithfulness metrics)
        if step_callback is not None:
            step_callback(step, acc_pred.detach().clone(), weights.detach().clone())

        # compute loss to unlearn target class
        loss = loss_fn(acc_pred, target_class) + l2_penalty * l2_regularisation(weights)
        loss.backward()
        # update gradient history (only if needed for cosine_similarity stopping)
        if store_grads:
            grads.append(weights.grad.detach().clone())  # type: ignore

        state = UnlearnState(
            step=step,
            weights=weights.detach().clone(),
            pred=acc_pred.detach().clone(),
            loss=loss.item(),
            grads=grads,
            target_class=target_class,
            distance_travelled=distance_travelled,
            init_pred=initial_prediction,
        )
        # stop if stopping criterium is met
        if stopping_criterium(state):
            break
        # update weights
        with th.no_grad():
            weights -= lr * weights.grad  # type: ignore
            distance_travelled += th.norm(lr * weights.grad).item()  # type: ignore
        weights.grad.zero_()  # type: ignore
        meta_network.zero_grad()

    # Move final results back to CPU for numpy compatibility
    state = UnlearnState(
        step=step,
        target_class=target_class,
        weights=weights.detach().cpu().clone(),
        pred=acc_pred.detach().cpu().clone(),
        loss=loss.item(),
        grads=grads,  # Keep on device (not typically used after unlearn)
        distance_travelled=distance_travelled,
        init_pred=initial_prediction.cpu(),
    )

    return state


if __name__ == "__main__":
    import pickle

    network = pickle.load(open("network_weights.pkl", "rb"))
    meta_network = pickle.load(open("meta_network.pkl", "rb"))
    edited_network = unlearn(network, meta_network, target_class=3)
