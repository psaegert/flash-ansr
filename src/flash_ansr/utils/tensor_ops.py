"""Tensor helpers used across the project."""
import numpy as np
import torch
from torch import nn


def pad_input_set(X: np.ndarray | torch.Tensor, length: int) -> np.ndarray | torch.Tensor:
    """Right-pad ``X`` to ``length`` along the last dimension."""
    pad_length = length - X.shape[-1]
    if pad_length > 0:
        if isinstance(X, torch.Tensor):
            X = nn.functional.pad(X, (0, pad_length, 0, 0), value=0)
        elif isinstance(X, np.ndarray):
            X = np.pad(X, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)  # type: ignore[arg-type]
    return X
