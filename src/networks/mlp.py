import warnings
from typing import List

import torch.nn as nn
from torch.nn.functional import normalize


class Normalize(nn.Module):
    def forward(self, x):
        return normalize(x)


class MLP(nn.Module):
    def __init__(
            self,
            dims: List[int],
            activation: bool | nn.Module = True,
            normalization: bool | nn.Module = True,
            last_layer: nn.Module | None = None,
            normalize_first: int = -1,
            dropout: float = 0.0,
            bias: bool | None = None
    ):
        r"""Generic and customizable multi-layer perceptron

        Args:
            dims: list of channels of each layer.
                For a d-deep MLP, d+1 channels should be provided (input channels + one per layer)
            activation: activation function between each linear layer.
                True stands for inplace ReLU and False for no activation at all.
            normalization: normalization between each layer (batchnorm, layernorm...).
                True stands for BatchNorm1d and False for no normalization at all.
            last_layer: eventual final layer (sigmoid, softmax...). None by default.
            bias: whether linear layers should have bias. By default, we add bias when there is no normalization.
                The last layer always have a bias.
        """
        super().__init__()
        self.in_dim = dims[0]
        self.out_dim = dims[-1]

        if len(dims) <= 2:
            if activation or normalization:
                warnings.warn(f"Defined an MLP with only {len(dims)-1} layers. "
                              f"`activation_fn` and `norm_fn` will be ignored.")

        # define activation layer
        if activation is True:
            activation = nn.ReLU(inplace=True)

        # define batch-norm layer
        if normalization is True:
            normalization = nn.BatchNorm1d
        elif isinstance(normalization, nn.Module):
            normalization = normalization.__class__

        # useless to add bias just before a batch-norm layer but add the option for completeness
        if bias is None:
            bias = not normalization

        layers = []

        output_dim = dims.pop()

        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if i == normalize_first:
                layers.append(Normalize())
            else:
                if normalization:
                    layers.append(normalization(out_dim))
                if activation:
                    layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims.pop(), output_dim, bias=True))

        if last_layer is not None:
            layers.append(last_layer)

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)
