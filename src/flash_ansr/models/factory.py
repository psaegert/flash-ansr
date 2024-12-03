import importlib
from typing import Any

from torch import nn


class ModelFactory():
    '''
    Factory class to create models from a string name.
    Supports models from torch.nn and nsr.models
    '''
    @staticmethod
    def get_model(model: str, *args: Any, **kwargs: Any) -> nn.Module:
        '''
        Get a model by name.
        Supports models from torch.nn and nsr.models

        Parameters
        ----------
        model : str
            The name of the model to create.
        *args : Any
            Positional arguments to pass to the model constructor.
        **kwargs : Any
            Keyword arguments to pass to the model constructor.

        Returns
        -------
        nn.Module
            The model instance.
        '''
        # Try to import the layer from torch.nn
        if hasattr(nn, model):
            return getattr(nn, model)(*args, **kwargs)

        # Try to import the layer from nsr.models.layers
        nsr_models = importlib.import_module("flash_ansr.models")
        if hasattr(nsr_models, model):
            return getattr(nsr_models, model)(*args, **kwargs)

        raise NotImplementedError(f"Layer {model} not found in torch.nn or nsr.models")
