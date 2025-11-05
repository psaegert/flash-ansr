"""Base classes and serialization helpers for set encoders."""
import os
import warnings
from abc import abstractmethod
from typing import Any, Literal

import torch
from torch import nn

from flash_ansr.utils.config_io import load_config, save_config
from flash_ansr.utils.paths import substitute_root_path


class SetEncoder(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "SetEncoder":
        config_ = load_config(config)

        if "encoder" in config_.keys():
            config_ = config_["encoder"]

        return cls(**config_)

    def save(
        self,
        directory: str,
        config: dict[str, Any] | str | None = None,
        reference: str = "relative",
        recursive: bool = True,
        errors: Literal["raise", "warn", "ignore"] = "warn",
    ) -> None:
        """Persist the encoder weights and its configuration."""
        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(directory, "state_dict.pt"))

        if config is None:
            if errors == "raise":
                raise ValueError(
                    "No config specified, saving the model without a config file. "
                    "Loading the model will require manual configuration."
                )
            if errors == "warn":
                warnings.warn(
                    "No config specified, saving the model without a config file. "
                    "Loading the model will require manual configuration."
                )
        else:
            save_config(
                load_config(config, resolve_paths=True),
                directory=directory,
                filename="set_encoder.yaml",
                reference=reference,
                recursive=recursive,
                resolve_paths=True,
            )

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "SetEncoder"]:
        """Restore an encoder and its configuration from ``directory``."""
        directory = substitute_root_path(directory)

        config_path = os.path.join(directory, "set_encoder.yaml")

        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return load_config(config_path), model

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
