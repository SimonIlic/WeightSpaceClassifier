import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "selu": nn.SELU,
}


class SmallCNN(nn.Module):
    def __init__(self, activation: str = "relu", dropout_rate: float = 0.0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")

        act_cls = ACTIVATIONS[activation]

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0)
        self.act1 = act_cls()
        self.drop1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.act2 = act_cls()
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.act3 = act_cls()
        self.drop3 = nn.Dropout(p=dropout_rate)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(16, 10)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            xavier_uniform_(m.weight)
            zeros_(m.bias)
        xavier_uniform_(self.dense.weight)
        zeros_(self.dense.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1] == 1, (
            f"Expected NCHW input with C=1, got shape {x.shape}"
        )

        x = self.drop1(self.act1(self.conv1(x)))
        x = self.drop2(self.act2(self.conv2(x)))
        x = self.drop3(self.act3(self.conv3(x)))

        x = self.pool(x)
        x = x.flatten(1)
        x = self.dense(x)
        return x
