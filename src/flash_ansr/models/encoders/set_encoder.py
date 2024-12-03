from typing import Any
from abc import abstractmethod

import torch
from torch import nn

from flash_ansr.utils import load_config


class SetEncoder(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "SetEncoder":
        config_ = load_config(config)

        if "encoder" in config_.keys():
            config_ = config_["encoder"]

        return cls(**config_)
