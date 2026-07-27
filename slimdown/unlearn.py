"""Batched unlearning: gradient descent in weight space, vectorized across models.

Per-model math is identical to cnn_surgery.unlearning.unlearn: the meta-network
is a row-wise MLP, so summing per-model losses before backward() yields each
row its own independent gradient. Per-model stopping is handled by removing
finished rows from the batch (compaction), so late steps only compute for
models that are still active.

Semantics preserved from the original sequential loop:
- The stop check fires after backward but before the weight update; a stopped
  model keeps its pre-update weights and records that step's pred/loss.
- Models that never stop get max_steps updates; their recorded pred/loss come
  from the final forward pass (pre-last-update), matching the original.
- distance_travelled accumulates ||lr * grad|| only for steps where the update
  was applied.
- cosine_similarity stops compare the current gradient against the first
  (derivative=False) or previous (derivative=True) gradient, first possible at
  step 1.
"""

from dataclasses import dataclass

import torch as th


# ---------------------------------------------------------------------
# Loss functions: (pred (b, 10), target_class) -> per-model loss (b,)
# ---------------------------------------------------------------------
def simple_loss(pred, target_class):
    """Reduce accuracy on the target class."""
    return pred[:, target_class]


def improve_loss(pred, target_class):
    """Increase accuracy on the target class (inverse unlearning)."""
    return -pred[:, target_class]


def boost_loss_factory(beta=0.1):
    """Unlearn the target class while boosting all other classes."""

    def boost_loss(pred, target_class):
        # weighting: -beta everywhere, +1 on the target class (same arithmetic
        # order as the original to stay bit-faithful in float32)
        weighting = -th.ones_like(pred[0]) * beta
        weighting[target_class] = 1
        return (weighting * pred).sum(dim=1)

    return boost_loss


# ---------------------------------------------------------------------
# Stopping criteria: StepState -> bool mask (b,) of rows that stop now
# ---------------------------------------------------------------------
@dataclass
class StepState:
    """Per-step view of the active rows, passed to stopping criteria."""

    step: int
    target_class: int
    pred: th.Tensor  # (b, 10)
    init_pred: th.Tensor  # (b, 10)
    grad: th.Tensor | None  # (b, D) current gradient
    first_grad: th.Tensor | None  # (b, D) gradient at step 0
    prev_grad: th.Tensor | None  # (b, D) gradient at previous step


def acc_pred_stop_factory(threshold=0.1, relative=False, improve=False):
    """Stop when the predicted target-class accuracy crosses the threshold."""

    def acc_pred_stop(state: StepState):
        if relative:
            below = state.pred[:, state.target_class] < state.init_pred[:, state.target_class] * threshold
        else:
            below = state.pred[:, state.target_class] < threshold
        return ~below if improve else below

    return acc_pred_stop


def cosine_similarity_stop_factory(derivative=False, eps=1e-2):
    """Stop when the gradient direction departs from the reference gradient."""

    def cosine_similarity_stop(state: StepState):
        ref = state.prev_grad if derivative else state.first_grad
        if state.step < 1 or ref is None:
            return th.zeros(state.pred.shape[0], dtype=th.bool, device=state.pred.device)
        cos = th.nn.functional.cosine_similarity(state.grad, ref, dim=1)
        return cos < 1 - eps

    return cosine_similarity_stop


def step_stop_factory(max_steps=100):
    def step_stop(state: StepState):
        return th.full((state.pred.shape[0],), state.step >= max_steps, dtype=th.bool, device=state.pred.device)

    return step_stop


def needs_grad_history(stopping_criterium) -> bool:
    return "cosine_similarity" in str(stopping_criterium)


# ---------------------------------------------------------------------
# Batched unlearning
# ---------------------------------------------------------------------
@dataclass
class BatchUnlearnState:
    """Final per-model results, all on CPU. Row i corresponds to input model i."""

    steps: th.Tensor  # (B,) long, last step index reached
    weights: th.Tensor  # (B, D) final weights
    pred: th.Tensor  # (B, 10) meta-network prediction at the final step
    init_pred: th.Tensor  # (B, 10) prediction before any update
    loss: th.Tensor  # (B,) loss at the final step
    distance_travelled: th.Tensor  # (B,) accumulated ||lr * grad|| over applied updates


def unlearn_batch(
    model_weights,
    meta_network: th.nn.Module,
    target_class: int,
    max_steps=100,
    lr=0.01,
    loss_fn=simple_loss,
    stopping_criterium=None,
    l2_penalty=1e-6,
    device: str = "cpu",
) -> BatchUnlearnState:
    """Unlearn a target class from a batch of models simultaneously.

    Args:
        model_weights: (B, D) array/tensor of flat model weights.
        meta_network: Row-wise network mapping (b, D) -> (b, n_classes) accuracies.
        target_class: Class index to unlearn.
        max_steps: Maximum number of gradient steps per model.
        lr: Step size.
        loss_fn: Vectorized loss, (pred (b, 10), target_class) -> (b,).
        stopping_criterium: Vectorized criterion, StepState -> bool mask (b,).
            Defaults to acc_pred_stop_factory().
        l2_penalty: L2 regularization on the weights.
        device: torch device for the optimization.
    """
    if stopping_criterium is None:
        stopping_criterium = acc_pred_stop_factory()

    w = th.as_tensor(model_weights, dtype=th.float32).to(device).clone()
    B, D = w.shape
    n_out = None
    track_grads = needs_grad_history(stopping_criterium)

    # Full-batch output buffers (CPU)
    out_steps = th.zeros(B, dtype=th.long)
    out_weights = w.detach().cpu().clone()
    out_loss = th.zeros(B)
    out_pred = None
    out_dist = th.zeros(B)

    with th.no_grad():
        init_pred_full = meta_network(w).detach()
    n_out = init_pred_full.shape[1]
    out_pred = th.zeros(B, n_out)

    # Active-row tensors; `orig` maps active row -> original batch index
    orig = th.arange(B, device=device)
    init_pred = init_pred_full
    dist = th.zeros(B, device=device)
    first_grad = None
    prev_grad = None

    w.requires_grad_(True)
    for step in range(max_steps):
        pred = meta_network(w)
        per_loss = loss_fn(pred, target_class) + l2_penalty * (w**2).sum(dim=1)
        per_loss.sum().backward()
        g = w.grad.detach()

        if track_grads and step == 0:
            first_grad = g.clone()

        stop = stopping_criterium(
            StepState(
                step=step,
                target_class=target_class,
                pred=pred.detach(),
                init_pred=init_pred,
                grad=g,
                first_grad=first_grad,
                prev_grad=prev_grad,
            )
        )

        last_step = step == max_steps - 1
        record = stop | th.full_like(stop, last_step)
        if record.any():
            idx = orig[record].cpu()
            out_steps[idx] = step
            out_loss[idx] = per_loss.detach()[record].cpu()
            out_pred[idx] = pred.detach()[record].cpu()
            # stopped rows keep pre-update weights; surviving last-step rows are
            # overwritten below after their final update
            out_weights[idx] = w.detach()[record].cpu()

        keep = ~stop
        if not keep.any():
            break

        # Compact: drop stopped rows, apply the update to the survivors
        w_new = (w.detach() - lr * g)[keep].clone()
        upd_norm = (lr * g[keep]).norm(dim=1)
        dist = dist[keep] + upd_norm
        orig = orig[keep]
        init_pred = init_pred[keep]
        if track_grads:
            prev_grad = g[keep].clone()
            first_grad = first_grad[keep]

        idx = orig.cpu()
        out_dist[idx] = dist.cpu()
        if last_step:
            out_weights[idx] = w_new.cpu()
            break

        w = w_new.requires_grad_(True)

    return BatchUnlearnState(
        steps=out_steps,
        weights=out_weights,
        pred=out_pred,
        init_pred=init_pred_full.cpu(),
        loss=out_loss,
        distance_travelled=out_dist,
    )
