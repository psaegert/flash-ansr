"""Optimizer factory utilities for training."""
from typing import Any

import torch
import torch_optimizer


def get_optimizer(name: str, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
    """Instantiate an optimiser by ``name`` from ``torch.optim`` or ``torch_optimizer``."""
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(*args, **kwargs)
    if hasattr(torch_optimizer, name):
        return getattr(torch_optimizer, name)(*args, **kwargs)
    raise NotImplementedError(f"Optimizer {name} not found in torch.optim or torch_optimizer")
