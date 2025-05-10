from typing import Any
from abc import abstractmethod
from typing import Literal
import os
import warnings

import torch
from torch import nn

from flash_ansr.utils import load_config, save_config, substitute_root_path


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

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True, errors: Literal['raise', 'warn', 'ignore'] = 'warn') -> None:
        '''
        Save the model to a directory.

        Parameters
        ----------
        directory : str
            The directory to save the model to.
        config : dict[str, Any] or str, optional
            The configuration dictionary or file, by default None.
        reference : str, optional
            Determines the reference base path. One of
            - 'relative': relative to the specified directory
            - 'project': relative to the project root
            - 'absolute': absolute paths
        recursive : bool, optional
            Save any referenced configs too
        errors : {'raise', 'warn', 'ignore'}, optional
            How to handle errors, by default 'warn'.

        Raises
        ------
        ValueError
            If no config is specified and errors is 'raise'.
        '''
        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(directory, "state_dict.pt"))

        # Copy the config to the directory for best portability
        if config is None:
            if errors == 'raise':
                raise ValueError("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
            if errors == 'warn':
                warnings.warn("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
        else:
            save_config(
                load_config(config, resolve_paths=True),
                directory=directory,
                filename='set_encoder.yaml',
                reference=reference,
                recursive=recursive,
                resolve_paths=True)

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "SetEncoder"]:
        '''
        Load a model from a directory.

        Parameters
        ----------
        directory : str
            The directory to load the model from.

        Returns
        -------
        dict[str, Any]
            The configuration dictionary.
        FlashANSRTransformer
            The SetEncoder instance
        '''
        directory = substitute_root_path(directory)

        config_path = os.path.join(directory, 'set_encoder.yaml')

        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return load_config(config_path), model

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
