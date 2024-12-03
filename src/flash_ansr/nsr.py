import copy
from typing import Literal

import numpy as np
import torch
from torch import nn

from sklearn.base import BaseEstimator

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
    nsr_transformer : FlashANSRTransformer
        The core transformer model.
    generation_type : {'beam_search'}, optional
        The type of generation to use, by default 'beam_search'.
    n_beams : int, optional
        The number of beams to generate, by default 1.
    numeric_head : bool, optional
        Whether to use the numeric head, by default False.
    equivalence_pruning : bool, optional
        Whether to use equivalence pruning, by default True.
    n_restarts : int, optional
        The number of restarts for the refiner, by default 1.
    max_len : int, optional
        The maximum length of the generated expression, by default 32.
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
            nsr_transformer: FlashANSRTransformer,
            generation_type: Literal['beam_search'] = 'beam_search',
            n_beams: int = 1,
            numeric_head: bool = False,
            equivalence_pruning: bool = True,
            n_restarts: int = 1,
            max_len: int = 32,
            p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            p0_noise_kwargs: dict | None = None,
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.01,
            verbose: bool = False):
        self.expression_space = expression_space
        self.nsr_transformer = nsr_transformer

        self.generation_type = generation_type
        self.n_beams = n_beams
        self.numeric_head = numeric_head
        self.equivalence_pruning = equivalence_pruning
        self.n_restarts = n_restarts
        self.max_len = max_len
        self.p0_noise = p0_noise
        self.p0_noise_kwargs = p0_noise_kwargs
        self.numpy_errors = numpy_errors
        self.parsimony = parsimony

        self._results: list[tuple[Refiner, dict]] = []
        self.verbose = verbose

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, converge_error: Literal['raise', 'ignore', 'print'] = 'ignore') -> "FlashANSR":
        '''
        Perform symbolic regression on the input data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            The input data.
        y : np.ndarray or torch.Tensor
            The target data.
        converge_error : {'raise', 'ignore', 'print'}, optional
            The behavior for convergence errors, by default 'ignore'.

        Returns
        -------
        NSR
            The fitted model.
        '''
        with torch.no_grad():
            # Convert the input data to a tensor
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)

            # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
            pad_length = self.nsr_transformer.encoder_max_n_variables - X.shape[-1] - y.shape[-1]
            if pad_length > 0:
                X = nn.functional.pad(X, (0, pad_length, 0, 0, 0, 0), value=0)

            # Concatenate x and y along the feature dimension
            data_tensor = torch.cat([X, y], dim=-1)

            # Generate the beams
            if self.generation_type == 'beam_search':
                beams, log_probs = self.nsr_transformer.beam_search(data_tensor, beam_size=self.n_beams, max_len=self.max_len, equivalence_pruning=self.equivalence_pruning)
            elif self.generation_type == 'softmax_sampling':
                raise NotImplementedError("Softmax sampling is not yet implemented")
                beams, log_probs = self.nsr_transformer.softmax_sampling(data_tensor, n_choices=self.n_beams, temperature=1, max_len=self.max_len)
            beams_decoded = [self.expression_space.tokenizer.decode(beam, special_tokens='<num>') for beam in beams]

            # Silence numpy errors
            numpy_errors_before = np.geterr()
            np.seterr(all=self.numpy_errors)

            self._results = []

            # Fit the refiner to each beam
            for beam, beam_decoded, log_prob in zip(beams, beams_decoded, log_probs):
                if self.expression_space.is_valid(beam_decoded):
                    numeric_prediction = None

                    if self.numeric_head:
                        with torch.no_grad():
                            _, num_output = self.nsr_transformer.forward(beam.unsqueeze(0), data_tensor.unsqueeze(0), numeric_head=True)
                            numeric_prediction = num_output[0, :, 0][beam == self.expression_space.tokenizer["<num>"]]

                    try:
                        refiner = Refiner(expression_space=self.expression_space).fit(
                            expression=beam_decoded,
                            X=X.cpu().numpy(),
                            y=y.cpu().numpy(),
                            n_restarts=self.n_restarts,
                            p0=numeric_prediction,
                            p0_noise=self.p0_noise,
                            p0_noise_kwargs=self.p0_noise_kwargs,
                            converge_error=converge_error)

                        self._results.append((
                            refiner,
                            {
                                'numeric_prediction': numeric_prediction,
                                'beam': beam,
                                'log_prob': log_prob,
                                'expression': beam_decoded,
                                'lambda': refiner.expression_lambda,
                                'fits': copy.deepcopy(refiner._all_constants_values),
                                'score': refiner._all_constants_values[0][-1] + self.parsimony * len(beam_decoded)
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

    def predict(self, X: np.ndarray, nth_best_beam: int = 0, nth_best_constants: int = 0) -> np.ndarray:
        '''
        Predict the target data using the fitted model.

        Parameters
        ----------
        X : np.ndarray
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
        return self._results[nth_best_beam][0].predict(X, nth_best_constants=nth_best_constants)
