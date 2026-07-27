"""Network definitions and flat-weight conversions.

Contains:
- The SmallCNN functional forward used for batched (vmapped) evaluation.
- Flat-weight parsing from the CNN Zoo storage format (verified against TF in
  pytorch_port/test_equivalence.py: [bias, kernel, ...] order, conv kernels
  stored as (kH, kW, C_in, C_out), dense kernels as (in, out)).
- The FCN meta-network (regressor lens) and a state-dict loader.
"""

from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = {
    "conv1.bias": (16,),
    "conv1.kernel": (3, 3, 1, 16),
    "conv2.bias": (16,),
    "conv2.kernel": (3, 3, 16, 16),
    "conv3.bias": (16,),
    "conv3.kernel": (3, 3, 16, 16),
    "dense.bias": (10,),
    "dense.kernel": (16, 10),
}

TOTAL_PARAMS = sum(prod(s) for s in SHAPES.values())  # 4970

ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "selu": F.selu,
}


def flat_to_params(weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Parse flat weights (TF storage order) into PyTorch-layout parameter tensors.

    Works batched: `weights` may be (D,) or (B, D); every returned tensor keeps
    the leading batch dimensions.

    Conversions (same as the verified pytorch_port):
        conv kernel (..., kH, kW, C_in, C_out) -> (..., C_out, C_in, kH, kW)
        dense kernel (..., in, out) -> (..., out, in)
    """
    if weights.shape[-1] != TOTAL_PARAMS:
        raise ValueError(f"Expected {TOTAL_PARAMS} weights in last dim, got {weights.shape[-1]}")

    batch_dims = weights.shape[:-1]
    params = {}
    i = 0
    for name, shape in SHAPES.items():
        length = prod(shape)
        chunk = weights[..., i : i + length].reshape(*batch_dims, *shape)
        i += length
        if name.startswith("conv") and name.endswith("kernel"):
            nd = len(batch_dims)
            chunk = chunk.permute(*range(nd), nd + 3, nd + 2, nd + 0, nd + 1)
        elif name == "dense.kernel":
            chunk = chunk.transpose(-1, -2)
        params[name.replace("kernel", "weight")] = chunk.contiguous()
    return params


def smallcnn_forward(params: dict[str, torch.Tensor], x: torch.Tensor, activation: str) -> torch.Tensor:
    """Functional forward pass of the Small CNN Zoo architecture (single model).

    Args:
        params: Unbatched parameter dict from flat_to_params.
        x: Images (N, 1, H, W), float32, normalized to [-1, 1].
        activation: One of 'relu', 'tanh', 'sigmoid', 'selu'.

    Returns:
        Logits (N, 10).
    """
    act = ACTIVATIONS[activation]
    x = act(F.conv2d(x, params["conv1.weight"], params["conv1.bias"], stride=2))
    x = act(F.conv2d(x, params["conv2.weight"], params["conv2.bias"], stride=2))
    x = act(F.conv2d(x, params["conv3.weight"], params["conv3.bias"], stride=2))
    x = x.mean(dim=(-2, -1))  # global average pooling
    return F.linear(x, params["dense.weight"], params["dense.bias"])


def _conv_out(size: int) -> int:
    """Output side length of a 3x3 stride-2 valid conv."""
    return (size - 3) // 2 + 1


def _shifted_conv(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """3x3 stride-2 valid conv for B models at once, as 9 shifted batched matmuls.

    Avoids materializing an im2col buffer (F.unfold dominates runtime on CPU).

    Args:
        h: (B, N, C, H, W) per-model feature maps.
        weight: (B, O, C, 3, 3) per-model kernels.
        bias: (B, O) per-model biases.

    Returns:
        (B, N, O, H', W') output feature maps.
    """
    oh, ow = _conv_out(h.shape[-2]), _conv_out(h.shape[-1])
    out = None
    for di in range(3):
        for dj in range(3):
            hs = h[..., di : di + 2 * oh - 1 : 2, dj : dj + 2 * ow - 1 : 2]
            contrib = torch.einsum("boc,bnchw->bnohw", weight[..., di, dj], hs)
            out = contrib if out is None else out + contrib
    return out + bias.reshape(bias.shape[0], 1, -1, 1, 1)


def smallcnn_forward_batched(params: dict[str, torch.Tensor], x: torch.Tensor, activation: str) -> torch.Tensor:
    """Forward pass of B models at once via batched matmuls.

    Much faster than vmap-over-conv2d (which lowers to grouped convolution and
    hits a slow path on CPU). The first conv shares one im2col across all
    models; the deeper convs use shift-and-matmul.

    Args:
        params: Parameter dict from flat_to_params with a leading batch dim B.
        x: Images (N, 1, H, W) shared by all models.
        activation: Activation name shared by the group.

    Returns:
        Logits (B, N, 10).
    """
    act = ACTIVATIONS[activation]
    B = params["conv1.weight"].shape[0]
    N, _, H, W = x.shape

    # conv1: input shared across models -> one im2col for all (small: 1 channel)
    oh, ow = _conv_out(H), _conv_out(W)
    u = F.unfold(x, kernel_size=3, stride=2)  # (N, 9, oh*ow)
    w = params["conv1.weight"].reshape(B, 16, 9)
    h = torch.einsum("bok,nkl->bnol", w, u) + params["conv1.bias"].reshape(B, 1, 16, 1)
    h = act(h).reshape(B, N, 16, oh, ow)

    h = act(_shifted_conv(h, params["conv2.weight"], params["conv2.bias"]))
    h = act(_shifted_conv(h, params["conv3.weight"], params["conv3.bias"]))

    pooled = h.mean(dim=(-2, -1))  # (B, N, 16) global average pooling
    return torch.einsum("bok,bnk->bno", params["dense.weight"], pooled) + params["dense.bias"].reshape(B, 1, 10)


# ---------------------------------------------------------------------
# Meta-network (regressor lens), identical to cnn_surgery.lenses.regressor_lens.FCN
# ---------------------------------------------------------------------
DEFAULT_META_ARCH = dict(n_layers=5, n_hidden=256, n_outputs=10, dropout_p=0.03, last_activation="sigmoid")


class FCN(nn.Module):
    def __init__(self, input_dim, n_layers, n_hidden, n_outputs, dropout_p, activation=nn.ReLU, last_activation="sigmoid"):
        super().__init__()
        self.flatten = nn.Flatten()
        blocks, in_f = [], input_dim
        for _ in range(n_layers):
            lin = nn.Linear(in_f, n_hidden)
            blocks += [lin, activation()]
            if dropout_p > 0:
                blocks.append(nn.Dropout(dropout_p))
            in_f = n_hidden
        self.hidden = nn.Sequential(*blocks)
        self.out = nn.Linear(in_f, n_outputs)
        self.last_activation = last_activation

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.out(x)
        if self.last_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


def load_meta_network(path: str, device: str = "cpu") -> FCN:
    """Load a meta-network saved by slimdown/convert_metanetworks.py.

    The .pt file holds {"state_dict": ..., "arch": {...}}. Meta-network
    parameters are frozen (requires_grad=False): unlearning only needs
    gradients w.r.t. the input weights.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if not (isinstance(payload, dict) and "state_dict" in payload and "arch" in payload):
        raise ValueError(
            f"{path} is not a slimdown meta-network file. Convert legacy .pkl files first: "
            "uv run python slimdown/convert_metanetworks.py"
        )
    arch = payload["arch"]
    model = FCN(
        input_dim=arch["input_dim"],
        n_layers=arch["n_layers"],
        n_hidden=arch["n_hidden"],
        n_outputs=arch["n_outputs"],
        dropout_p=arch["dropout_p"],
        activation=nn.ReLU,
        last_activation=arch.get("last_activation", "sigmoid"),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
