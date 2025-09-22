from abc import abstractmethod
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


class SetNormBase(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class OriginalSetNorm(SetNormBase):
    """
    Mask-aware Set Normalization layer with improved numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Store original dtype and upcast to float32 for stable calculations
        input_dtype = x.dtype
        x = x.float()

        if attn_mask is None:
            mu = x.mean(dim=(1, 2), keepdim=True)
            # Use unbiased=False to match torch.var default for population variance
            sigma = x.std(dim=(1, 2), keepdim=True, unbiased=False)
        else:
            mask_expanded = attn_mask.unsqueeze(-1)
            # Mask the input to zero out padded values before summing
            masked_x = x * mask_expanded
            # Count non-padded elements, ensuring it's at least 1 to avoid division by zero
            n_elements = (attn_mask.sum(dim=1, keepdim=True) * x.shape[-1]).clamp(min=1).unsqueeze(-1)

            # Calculate masked mean
            mu = masked_x.sum(dim=(1, 2), keepdim=True) / n_elements

            # Calculate masked variance and std
            var = (masked_x - mu).pow(2)
            var = (var * mask_expanded).sum(dim=(1, 2), keepdim=True) / n_elements
            sigma = torch.sqrt(var)

        x_norm = (x - mu) / (sigma + self.eps)

        # Apply learnable parameters and cast back to the original dtype
        return (x_norm * self.gamma + self.beta).to(input_dtype)


class RMSSetNorm(SetNormBase):
    """
    Mask-aware RMS Normalization layer for sets with improved numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Store original dtype and upcast to float32 for stable calculations
        input_dtype = x.dtype
        x = x.float()

        if attn_mask is None:
            mean_sq = x.pow(2).mean(dim=(1, 2), keepdim=True)
        else:
            mask_expanded = attn_mask.unsqueeze(-1)
            # Calculate sum of squares only on non-padded elements
            sum_sq = (x.pow(2) * mask_expanded).sum(dim=(1, 2), keepdim=True)
            # Count non-padded elements, ensuring it's at least 1
            n_elements = (attn_mask.sum(dim=1, keepdim=True) * x.shape[-1]).clamp(min=1).unsqueeze(-1)
            # Calculate mean square over valid elements
            mean_sq = sum_sq / n_elements

        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x / rms

        # Apply learnable parameter and cast back to the original dtype
        return (x_norm * self.gamma).to(input_dtype)


class DynLogNorm(nn.Module):
    """
    Dynamic Logarithmic Normalization. Similar to Dynamic Tanh but uses logarithmic scaling without saturation.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.log1p(torch.sigmoid(self.alpha) * torch.abs(x)) * torch.sign(x) + self.beta


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
        return OriginalSetNorm(dim, **kwargs)
    if norm_type_l in ("rms_set", "rmssetnorm"):
        return RMSSetNorm(dim, **kwargs)
    if norm_type_l in ("dyn_log", "dynlognorm"):
        return DynLogNorm(dim, **kwargs)
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
