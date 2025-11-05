"""Common neural network components shared by encoders and decoders."""
from abc import abstractmethod
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x)
        return output * self.weight


class SetNormBase(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class OriginalSetNorm(SetNormBase):
    """Mask-aware set normalisation layer with improved numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = x.dtype

        if attn_mask is None:
            mu = x.mean(dim=(1, 2), keepdim=True)
            sigma = x.std(dim=(1, 2), keepdim=True, unbiased=False)
        else:
            mask_expanded = attn_mask.unsqueeze(-1)
            masked_x = x * mask_expanded
            n_elements = (attn_mask.sum(dim=1, keepdim=True) * x.shape[-1]).clamp(min=1).unsqueeze(-1)

            mu = masked_x.sum(dim=(1, 2), keepdim=True) / n_elements

            var = (masked_x - mu).pow(2)
            var = (var * mask_expanded).sum(dim=(1, 2), keepdim=True) / n_elements
            sigma = torch.sqrt(var)

        x_norm = (x - mu) / (sigma + self.eps)
        return (x_norm * self.gamma + self.beta).to(input_dtype)


class RMSSetNorm(SetNormBase):
    """Mask-aware RMS normalisation layer for sets."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = x.dtype

        if attn_mask is None:
            mean_sq = x.pow(2).mean(dim=(1, 2), keepdim=True)
        else:
            mask_expanded = attn_mask.unsqueeze(-1)
            sum_sq = (x.pow(2) * mask_expanded).sum(dim=(1, 2), keepdim=True)
            n_elements = (attn_mask.sum(dim=1, keepdim=True) * x.shape[-1]).clamp(min=1).unsqueeze(-1)
            mean_sq = sum_sq / n_elements

        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x / rms
        return (x_norm * self.gamma).to(input_dtype)


def get_norm_layer(norm_type: str, dim: int, **kwargs: Any) -> nn.Module:
    """Factory for normalisation layers shared by encoders and decoders."""
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
    raise ValueError(f"Unknown norm_type: {norm_type}")


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""

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
