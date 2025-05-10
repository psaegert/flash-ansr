from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from flash_ansr.models.factory import ModelFactory


class ConfigurableSequential(nn.Sequential):
    '''
    A configurable version of nn.Sequential that can be created from a configuration dictionary.
    '''
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConfigurableSequential":
        '''
        Create a ConfigurableSequential from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            The configuration dictionary. Must contain a "layers" key with a list of layer configurations in the format {"type": str, "kwargs": dict[str, Any]}.

        Returns
        -------
        ConfigurableSequential
            The ConfigurableSequential instance.
        '''
        return cls([ModelFactory.get_model(layer["type"], **layer["kwargs"]) for layer in config["layers"]])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the layers.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        '''
        return self.layers(X)


class ReLU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the output of the combined linear layer
        x12: torch.Tensor = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2  # SwiGLU activation
        return self.w3(hidden)
