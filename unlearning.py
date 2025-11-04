# unlearning algorithm as proposed in "Amazing Paper Title" by Moos & Simon 2026"
import torch as th
from torch.nn.functional import cosine_similarity


def boost_loss(pred, target_class, beta=0.1):
    """ Encourages unlearning of a specific class, whilst boosting accuracy on other classes."""
    weighting = -th.ones_like(pred) * beta  # all classes go up
    weighting[target_class] = 1  # target class goes down
    return (weighting * pred).mean()


def unlearn(model_weights, meta_network: th.nn.Module, target_class,
            max_steps=100, lr=0.01, eps=1e-2, loss_fn=boost_loss):
    """
    Unlearn a target class from the model using a meta-network to guide weight updates.

    Parameters:
    - model_weights: Initial weights of the model to be edited.
    - meta_network: A network that predicts per-class model performance from model parameters.
    - target_class: The class index to be unlearned.
    - max_steps: Maximum number of unlearning steps.
    - lr: Learning rate (step-size) for weight updates.
    - eps: Convergence threshold.

    Returns:
    - Updated model weights after unlearning.
    """
    weights = th.tensor(model_weights, requires_grad=True).unsqueeze(0)
    weights.retain_grad()
    grads = []

    for step in range(max_steps):
        # forward pass through meta-network
        acc_pred = meta_network(weights).squeeze(0)
        # compute loss to unlearn target class
        loss = loss_fn(acc_pred, target_class)

        loss.backward()
        grads.append(weights.grad.detach().clone())
        # update weights
        with th.no_grad():
            weights -= lr * weights.grad  # type: ignore
            weights.grad.zero_()  # type: ignore
            meta_network.zero_grad()

        # check convergence
        if step > 1 and cosine_similarity(grads[-1], grads[0]) < 1 - eps:
            print(f"Converged after {step} steps.")
            break

    return weights
