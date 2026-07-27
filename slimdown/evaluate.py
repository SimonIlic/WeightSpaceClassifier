"""Batched CNN evaluation: per-class accuracy for many models in one pass.

B models are evaluated on the test set together via im2col + batched matmul
(smallcnn_forward_batched). Rows are grouped by activation function (the zoo
only contains relu/tanh) because the activation is a static choice per model.
"""

import numpy as np
import torch

from slimdown.nets import flat_to_params, smallcnn_forward_batched

# Memory budget for intermediate activations: models * images * ~6k floats each
# (dominated by the conv2 im2col). 2e8 elements ~= 800 MB of float32.
_ELEM_BUDGET = int(2e8)
_ACTIVATION_SIZE = 6000  # rough per-image, per-model intermediate footprint


def _predict_group(weights: torch.Tensor, activation: str, x_test: torch.Tensor) -> torch.Tensor:
    """Predicted class labels for a group of models sharing one activation.

    Args:
        weights: (b, 4970) flat weights on the eval device.
        activation: Activation name shared by the group.
        x_test: (N, 1, H, W) images on the eval device.

    Returns:
        (b, N) long tensor of predicted labels.
    """
    params = flat_to_params(weights)

    b = weights.shape[0]
    chunk = max(16, _ELEM_BUDGET // (_ACTIVATION_SIZE * b))
    preds = []
    with torch.no_grad():
        for start in range(0, x_test.shape[0], chunk):
            logits = smallcnn_forward_batched(params, x_test[start : start + chunk], activation)  # (b, n, 10)
            preds.append(logits.argmax(dim=-1))
    return torch.cat(preds, dim=1)


def evaluate_batch(
    weights: torch.Tensor | np.ndarray,
    activations,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate B models on the test set.

    Args:
        weights: (B, 4970) flat weights.
        activations: Length-B sequence of activation names (one per model).
        x_test: (N, 1, H, W) test images, normalized to [-1, 1].
        y_test: (N,) integer labels.
        num_classes: Forced class count; inferred from y_test if None.
        device: Device for inference.

    Returns:
        overall_acc: (B,) float64 array.
        per_class_acc: (B, num_classes) float64 array (NaN for absent classes).
    """
    weights = torch.as_tensor(weights, dtype=torch.float32)
    B = weights.shape[0]
    activations = np.asarray(activations)
    if num_classes is None:
        num_classes = int(y_test.max().item() + 1)

    x_test = x_test.to(device)
    y = y_test.to(device)

    preds = torch.empty(B, y.shape[0], dtype=torch.long, device=device)
    for act in np.unique(activations):
        rows = np.flatnonzero(activations == act)
        group_preds = _predict_group(weights[rows].to(device), str(act), x_test)
        preds[torch.as_tensor(rows, device=device)] = group_preds

    correct = (preds == y.unsqueeze(0)).float()  # (B, N)
    one_hot = torch.nn.functional.one_hot(y, num_classes).float()  # (N, C)
    class_counts = one_hot.sum(dim=0)  # (C,)
    correct_per_class = correct @ one_hot  # (B, C)

    overall_acc = correct.mean(dim=1).cpu().numpy().astype(np.float64)
    per_class = (correct_per_class / class_counts).cpu().numpy().astype(np.float64)
    per_class[:, (class_counts == 0).cpu().numpy()] = np.nan
    return overall_acc, per_class
