import os
import copy
from typing import Literal, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from sklearn.base import BaseEstimator

from flash_ansr.utils import substitute_root_path
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.models import FlashANSRTransformer
from flash_ansr.expressions import ExpressionSpace


class FlashANSR(BaseEstimator):
    '''
    Flash Amortized Neural Symbolic Regressor.

    Parameters
    ----------
    expression_space : ExpressionSpace
        The expression space used for manipulating expressions.
    flash_ansr_transformer : FlashANSRTransformer
        The core transformer model.
    generation_type : {'beam_search'}, optional
        The type of generation to use, by default 'beam_search'.
    beam_width : int, optional
        The number of beams to generate, by default 1.
    numeric_head : bool, optional
        Whether to use the numeric head, by default False.
    equivalence_pruning : bool, optional
        Whether to use equivalence pruning, by default True.
    n_restarts : int, optional
        The number of restarts for the refiner, by default 1.
    max_len : int, optional
        The maximum length of the generated expression, by default 32.
    refiner_method : str, optional
        The optimization method to use. One of
        - 'curve_fit_lm': Use the curve_fit method with the Levenberg-Marquardt algorithm
        - 'minimize_bfgs': Use the minimize method with the BFGS algorithm
    p0_noise : {'uniform', 'normal'}, optional
        The type of noise to add to the initial guess, by default 'normal'.
    p0_noise_kwargs : dict, optional
        The keyword arguments for the noise function, by default None.
    numpy_errors : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        The behavior for numpy errors, by default 'ignore'.
    parsimony : float, optional
        The parsimony coefficient, by default 0.01.
    verbose : bool, optional
        Whether to print verbose output, by default False.
    '''
    def __init__(
            self,
            expression_space: ExpressionSpace,
            flash_ansr_transformer: FlashANSRTransformer,
            generation_type: Literal['beam_search'] = 'beam_search',
            beam_width: int = 1,
            numeric_head: bool = False,
            equivalence_pruning: bool = True,
            n_restarts: int = 1,
            max_len: int = 32,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.01,
            verbose: bool = False):
        self.expression_space = expression_space
        self.flash_ansr_transformer = flash_ansr_transformer.eval()

        self.generation_type = generation_type
        self.beam_width = beam_width
        self.numeric_head = numeric_head
        self.equivalence_pruning = equivalence_pruning
        self.n_restarts = n_restarts
        self.max_len = max_len
        self.refiner_method = refiner_method
        self.p0_noise = p0_noise
        self.p0_noise_kwargs = p0_noise_kwargs
        self.numpy_errors = numpy_errors
        self.parsimony = parsimony

        self._results: list[tuple[Refiner, dict]] = []
        self.verbose = verbose

        self.variable_mapping: dict[str, str] = {}

    @classmethod
    def load(
            cls,
            directory: str,
            generation_type: Literal['beam_search'] = 'beam_search',
            beam_width: int = 1,
            numeric_head: bool = False,
            equivalence_pruning: bool = True,
            n_restarts: int = 1,
            max_len: int = 32,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.01,
            verbose: bool = False) -> "FlashANSR":
        directory = substitute_root_path(directory)

        expression_space_path = os.path.join(directory, 'expression_space.yaml')
        flash_ansr_transformer_path = os.path.join(directory, 'nsr.yaml')

        expression_space = ExpressionSpace.from_config(expression_space_path)

        model = FlashANSRTransformer.from_config(flash_ansr_transformer_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return cls(
            expression_space=expression_space,
            flash_ansr_transformer=model,
            generation_type=generation_type,
            beam_width=beam_width,
            numeric_head=numeric_head,
            equivalence_pruning=equivalence_pruning,
            n_restarts=n_restarts,
            max_len=max_len,
            refiner_method=refiner_method,
            p0_noise=p0_noise,
            p0_noise_kwargs=p0_noise_kwargs,
            numpy_errors=numpy_errors,
            parsimony=parsimony,
            verbose=verbose)

    def fit(self, X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series, variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto', converge_error: Literal['raise', 'ignore', 'print'] = 'ignore', verbose: bool = False) -> "FlashANSR":
        '''
        Perform symbolic regression on the input data.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor | pd.DataFrame
            The input data.
        y : np.ndarray | torch.Tensor | pd.DataFrame | pd.Series
            The target data.
        variable_names : list[str] | dict[str, str] | {'auto'}, optional
            The variable names, by default 'auto'.
            - If list[str], the i-th column of X will be named variable_names[i].
            - If dict[str, str]:
                - If X is array-like, the i-th column of X will be named after the variable_names keys
                - If X is a DataFrame, variable_names will be used to map the column names to the variable names.
            - If 'auto':
                - If X is a DataFrame, the column names will be used as variable names.
                - If X is an array or tensor, the variables will be named x0, x1, x2, ...
            - If None, the variables will be named x0, x1, x2, ...
        converge_error : {'raise', 'ignore', 'print'}, optional
            The behavior for convergence errors, by default 'ignore'.
        verbose : bool, optional
            Whether to display a progress bar, by default False.

        Returns
        -------
        FlashANSR
            The fitted model.
        '''
        # Default: No mapping
        self.variable_mapping = {}

        if isinstance(variable_names, list):
            # column i -> variable_names[i]
            self.variable_mapping = {f"x{i + 1}": name for i, name in enumerate(variable_names)}

        elif isinstance(variable_names, dict):
            if isinstance(X, pd.DataFrame):
                # column i -> variable_names[column i]
                self.variable_mapping = {f"x{i + 1}": variable_names[c] for i, c in enumerate(X.columns)}
            else:
                # custom mapping
                self.variable_mapping = variable_names

        elif variable_names == 'auto':
            if isinstance(X, pd.DataFrame):
                # column i -> column name
                self.variable_mapping = {f"x{i + 1}": name for i, name in enumerate(X.columns)}

        with torch.no_grad():
            # Convert the input data to a tensor
            if not isinstance(X, torch.Tensor):
                if isinstance(X, pd.DataFrame):
                    X = torch.tensor(X.values, dtype=torch.float32, device=self.flash_ansr_transformer.device)
                else:
                    X = torch.tensor(X, dtype=torch.float32, device=self.flash_ansr_transformer.device)
            else:
                X = X.to(self.flash_ansr_transformer.device)

            if not isinstance(y, torch.Tensor):
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = torch.tensor(y.values, dtype=torch.float32, device=self.flash_ansr_transformer.device)
                else:
                    y = torch.tensor(y, dtype=torch.float32, device=self.flash_ansr_transformer.device)
            else:
                y = y.to(self.flash_ansr_transformer.device)

            if y.dim() == 1:
                y = y.unsqueeze(-1)

            # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
            pad_length = self.flash_ansr_transformer.encoder_max_n_variables - X.shape[-1] - y.shape[-1]

            if pad_length > 0:
                X = nn.functional.pad(X, (0, pad_length, 0, 0), value=0)

            # Concatenate x and y along the feature dimension
            data_tensor = torch.cat([X, y], dim=-1)

            # Generate the beams
            if self.generation_type == 'beam_search':
                beams, log_probs = self.flash_ansr_transformer.beam_search(data_tensor, beam_width=self.beam_width, max_len=self.max_len, equivalence_pruning=self.equivalence_pruning, verbose=verbose)
            elif self.generation_type == 'softmax_sampling':
                raise NotImplementedError("Softmax sampling is not yet implemented")
            beams_decoded = [self.expression_space.tokenizer.decode(beam, special_tokens='<num>') for beam in beams]

            # Silence numpy errors
            numpy_errors_before = np.geterr()
            np.seterr(all=self.numpy_errors)

            self._results = []

            # Fit the refiner to each beam
            for beam, beam_decoded, log_prob in tqdm(zip(beams, beams_decoded, log_probs), desc="Fitting Constants", disable=not verbose, total=len(beams)):
                if self.expression_space.is_valid(beam_decoded):
                    numeric_prediction = None

                    if self.numeric_head:
                        with torch.no_grad():
                            _, num_output = self.flash_ansr_transformer.forward(beam.unsqueeze(0), data_tensor.unsqueeze(0), numeric_head=True)
                            numeric_prediction = num_output[0, :, 0][beam == self.expression_space.tokenizer["<num>"]]  # FIXME: Start at 1 or 0?

                    try:
                        refiner = Refiner(expression_space=self.expression_space).fit(
                            expression=beam_decoded,
                            X=X.cpu().numpy(),
                            y=y.cpu().numpy(),
                            n_restarts=self.n_restarts,
                            method=self.refiner_method,
                            p0=numeric_prediction,
                            p0_noise=self.p0_noise,
                            p0_noise_kwargs=self.p0_noise_kwargs,
                            converge_error=converge_error)

                        if refiner.constants_values is None:  # Fit failed
                            score = np.inf
                        else:
                            score = refiner._all_constants_values[0][-1] + self.parsimony * len(beam_decoded)

                        self._results.append((
                            refiner,
                            {
                                'numeric_prediction': numeric_prediction,
                                'beam': beam,
                                'log_prob': log_prob,
                                'expression': beam_decoded,
                                'lambda': refiner.expression_lambda,
                                'fits': copy.deepcopy(refiner._all_constants_values),
                                'score': score
                            }))

                    except ConvergenceError:
                        if self.verbose and converge_error == 'print':
                            print(f"Failed to converge for beam: {beam_decoded}")

            if not self._results:
                raise ConvergenceError("The optimization did not converge for any beam")

            # Sort the results by the best loss of each beam
            self._results = list(sorted(self._results, key=lambda x: x[1]['score']))

            np.seterr(**numpy_errors_before)

            return self

    def predict(self, X: np.ndarray | torch.Tensor | pd.DataFrame, nth_best_beam: int = 0, nth_best_constants: int = 0) -> np.ndarray:
        '''
        Predict the target data using the fitted model.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor | pd.DataFrame
            The input data.
        nth_best_beam : int, optional
            The nth best beam to use, by default 0.
        nth_best_constants : int, optional
            The nth best constants to use for the given beam, by default 0.

        Returns
        -------
        np.ndarray
            The predicted target data.
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
        pad_length = self.flash_ansr_transformer.encoder_max_n_variables - X.shape[-1] - 1

        if pad_length > 0:
            if isinstance(X, torch.Tensor):
                X = nn.functional.pad(X, (0, pad_length, 0, 0), value=0)
            elif isinstance(X, np.ndarray):
                X = np.pad(X, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)

        if len(self._results) == 0:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")

        return self._results[nth_best_beam][0].predict(X, nth_best_constants=nth_best_constants)

    def get_expression(self, nth_best_beam: int = 0, nth_best_constants: int = 0, return_prefix: bool = False, precision: int = 2, map_variables: bool = True, **kwargs: Any) -> list[str] | str:
        '''
        Get the nth best expression.

        Parameters
        ----------
        nth_best_beam : int, optional
            The nth best beam to use, by default 0.
        nth_best_constants : int, optional
            The nth best constants to use for the given beam, by default 0.
        return_prefix : bool, optional
            Whether to return the expression with the prefix, by default False.
        precision : int, optional
            The precision for rounding the constants, by default 2.
        map_variables : bool, optional
            Whether to map the variables to their specified names if possible, by default True.
        **kwargs : Any

        Returns
        -------
        list[str] | str
            The nth best expression.
        '''
        return self._results[nth_best_beam][0].transform(
            expression=self._results[nth_best_beam][1]['expression'],
            nth_best_constants=nth_best_constants,
            return_prefix=return_prefix,
            precision=precision,
            variable_mapping=self.variable_mapping if map_variables else None,
            **kwargs)

    def to(self, device: str) -> "FlashANSR":
        '''
        Move the model to a device.

        Parameters
        ----------
        device : str
            The device to move the model to.

        Returns
        -------
        FlashANSR
            The model on the new device.
        '''
        self.flash_ansr_transformer.to(device)
        return self
