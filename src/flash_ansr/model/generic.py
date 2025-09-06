from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from flash_ansr.model.factory import ModelFactory


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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The output is cast to the input's dtype, preventing issues with mixed precision.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SetNorm(nn.Module):
    """
    Set Normalization layer.

    Normalizes features across the set and feature dimensions for each batch element.
    Given an input X of shape (B, M, D), it computes statistics mu and sigma
    over the M and D dimensions, resulting in statistics of shape (B, 1, 1).
    It then applies learnable affine parameters gamma and beta of shape (1, 1, D).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, M, D)
        # Calculate mean and std over the set and feature dimensions (1 and 2)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True)

        # Normalize and apply affine transformation
        x_norm = (x - mu) / (sigma + self.eps)
        return x_norm * self.gamma + self.beta


class RMSSetNorm(nn.Module):
    """
    RMS Normalization layer for sets.

    Normalizes features across the set and feature dimensions for each batch element.
    Given an input X of shape (B, M, D), it computes the RMS over the M and D dimensions,
    resulting in statistics of shape (B, 1, 1).
    It then applies a learnable affine parameter gamma of shape (1, 1, D).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, M, D)
        rms = self._rms(x)
        x_norm = x / rms
        return x_norm * self.gamma


def get_norm_layer(norm_type: str, dim: int, **kwargs: Any) -> nn.Module:
    """Factory for normalization layers.

    Parameters
    ----------
    norm_type : str
        Type of normalization. Supported: "rms", "layer", "none", "set", "rms_set".
    dim : int
        Feature dimension.
    **kwargs : Any
        Extra keyword arguments passed to the underlying norm constructor.
    """
    norm_type_l = norm_type.lower()
    if norm_type_l in ("rms", "rmsnorm", "rms_norm"):
        return RMSNorm(dim, **kwargs)
    if norm_type_l in ("layer", "layernorm", "ln"):
        return nn.LayerNorm(dim, **kwargs)
    if norm_type_l in ("none", "identity", "id"):
        return nn.Identity()
    if norm_type_l in ("set", "setnorm"):
        return SetNorm(dim, **kwargs)
    if norm_type_l in ("rms_set", "rmssetnorm"):
        return RMSSetNorm(dim, **kwargs)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class FeedForward(nn.Module):
    """Standard Feed-Forward Network with GELU activation."""
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))
