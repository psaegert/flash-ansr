import json
from functools import partial
from typing import Callable

import omegaconf

from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, BFGSParams


def load_nesymres(eq_setting_path: str, config_path: str, weights_path: str, beam_size: int | None = 32, n_restarts: int | None = 4, device: str = "cpu") -> tuple[Model, Callable]:
    '''
    Load the NeSymReS model and the fitting function.

    Parameters
    ----------
    eq_setting_path : str
        Path to the equation setting file.
    config_path : str
        Path to the configuration file.
    weights_path : str
        Path to the weights file.
    beam_size : int, optional
        Beam size for the beam search algorithm. Default is 32.
    n_restarts : int, optional
        Number of restarts for the BFGS algorithm. Default is 4.
    device : str, optional
        Device to load the model. Default is "cpu".

    Returns
    -------
    tuple[Model, Callable]
        The NeSymReS model and the fitting function.

    Notes
    -----
    This function is largely based on the code from the Neural Symbolic Regression That Scales repository:
    https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/jupyter/fit_func.ipynb
    '''
    # Load equation configuration and architecture configuration
    with open(eq_setting_path, 'r') as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load(config_path)

    # Set up BFGS load rom the hydra config yaml
    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=n_restarts or cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(
        word2id=eq_setting["word2id"],
        id2word={int(k): v for k, v in eq_setting["id2word"].items()},
        una_ops=eq_setting["una_ops"],
        bin_ops=eq_setting["bin_ops"],
        total_variables=list(eq_setting["total_variables"]),
        total_coefficients=list(eq_setting["total_coefficients"]),
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=beam_size or cfg.inference.beam_size  # This parameter is a tradeoff between accuracy and fitting time
    )

    # Load architecture, set into eval mode, and pass the config parameters
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture).eval().to(device)

    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    return model, fitfunc
