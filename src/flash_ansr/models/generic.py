from typing import Any

import torch
from torch import nn

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
