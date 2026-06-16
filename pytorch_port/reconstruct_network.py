import warnings
from math import prod

import numpy as np
import torch

from pytorch_port.model import SmallCNN

SHAPES = {
    "sequential/conv2d/bias:0": (16,),
    "sequential/conv2d/kernel:0": (3, 3, 1, 16),
    "sequential/conv2d_1/bias:0": (16,),
    "sequential/conv2d_1/kernel:0": (3, 3, 16, 16),
    "sequential/conv2d_2/bias:0": (16,),
    "sequential/conv2d_2/kernel:0": (3, 3, 16, 16),
    "sequential/dense/bias:0": (10,),
    "sequential/dense/kernel:0": (16, 10),
}

TOTAL_PARAMS = sum(prod(s) for s in SHAPES.values())


def _flat_weights_to_state_dict(weights: np.ndarray) -> dict[str, torch.Tensor]:
    """Parse a flat weight array (TF storage order) into a PyTorch state dict.

    Flat array layout: [bias0, kernel0, bias1, kernel1, ...]
    Conversions:
        Conv2D kernel: (kH, kW, C_in, C_out) -> (C_out, C_in, kH, kW)
        Conv2D bias:   (C_out,) -> (C_out,)
        Dense kernel:  (in, out) -> (out, in)
        Dense bias:    (out,) -> (out,)
    """
    if weights.shape[0] != TOTAL_PARAMS:
        raise ValueError(f"Expected {TOTAL_PARAMS} weights, got {weights.shape[0]}")

    # Parse flat array into (name, array) pairs in storage order
    parsed = []
    i = 0
    for name, shape in SHAPES.items():
        length = prod(shape)
        parsed.append((name, weights[i : i + length].reshape(shape)))
        i += length

    # Map TF names -> PyTorch state dict keys + apply format conversions
    tf_to_pt = {
        "sequential/conv2d/bias:0": ("conv1.bias", False),
        "sequential/conv2d/kernel:0": ("conv1.weight", True),
        "sequential/conv2d_1/bias:0": ("conv2.bias", False),
        "sequential/conv2d_1/kernel:0": ("conv2.weight", True),
        "sequential/conv2d_2/bias:0": ("conv3.bias", False),
        "sequential/conv2d_2/kernel:0": ("conv3.weight", True),
        "sequential/dense/bias:0": ("dense.bias", False),
        "sequential/dense/kernel:0": ("dense.weight", False),
    }

    state_dict = {}
    for name, arr in parsed:
        pt_key, is_conv = tf_to_pt[name]
        if is_conv:
            arr = np.transpose(arr, (3, 2, 0, 1))
        elif pt_key == "dense.weight":
            arr = arr.T
        state_dict[pt_key] = torch.from_numpy(arr.copy()).float()

    return state_dict


def reconstruct_network(
    weights: np.ndarray,
    activation: str,
    l2_penalty: float = 0.0,
    dropout_rate: float = 0.0,
    device: torch.device | str | None = None,
) -> SmallCNN:
    """Reconstruct a SmallCNN from the flat weight array.

    The weights array uses the same format as the TF version (SHAPES order).
    Returns a SmallCNN in eval mode with weights loaded.

    Args:
        weights: Flat weight array of length 4970.
        activation: One of 'relu', 'tanh', 'sigmoid', 'selu'.
        l2_penalty: Accepted for API compatibility only. No effect on forward pass.
        dropout_rate: Dropout rate (0.0 = no dropout).
        device: Device to place model on. None = CPU.

    Returns:
        SmallCNN in eval mode with weights loaded.
    """
    if l2_penalty > 0.0:
        warnings.warn(
            "l2_penalty is accepted for API compatibility but has no effect "
            "on the PyTorch model's forward pass. It only matters during training.",
            UserWarning,
            stacklevel=2,
        )

    model = SmallCNN(activation=activation, dropout_rate=dropout_rate)
    state_dict = _flat_weights_to_state_dict(weights)
    model.load_state_dict(state_dict)

    if device is not None:
        model = model.to(device)

    model.eval()
    return model
