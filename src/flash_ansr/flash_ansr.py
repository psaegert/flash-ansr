import os
import copy
from typing import Literal, Any, Iterable, TypedDict, Callable
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.base import BaseEstimator

from simplipy import SimpliPyEngine

from flash_ansr.utils import substitute_root_path, pad_input_set, GenerationConfig
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.model import FlashANSRModel, Tokenizer


class Result(TypedDict):
    refiner: Refiner
    numeric_prediction: torch.Tensor | None
    beam: list[int]
    log_prob: float
    expression: list[str]
    raw_beam: list[int]
    raw_beam_decoded: str
    complexity: int
    function: Callable
    fits: list[tuple[np.ndarray, np.ndarray, float]]
    score: float
    target_complexity: int | float | None
    fvu: float


class FlashANSR(BaseEstimator):
    """Flash Amortized Neural Symbolic Regressor.

    Parameters
    ----------
    simplipy_engine : SimpliPyEngine
        Engine responsible for manipulating and evaluating symbolic expressions.
    flash_ansr_transformer : FlashANSRModel
        Trained transformer backbone that proposes expression programs.
    tokenizer : Tokenizer
        Tokenizer mapping model outputs to expression tokens.
    generation_config : GenerationConfig, optional
        Configuration that controls candidate generation. If ``None`` a default
        ``GenerationConfig`` is created.
    numeric_head : bool, optional
        Whether to enable the numeric head to predict constants directly.
    n_restarts : int, optional
        Number of optimizer restarts used by the refiner when fitting constants.
    refiner_method : {'curve_fit_lm', 'minimize_bfgs'}
        Optimization routine employed by the refiner.
    refiner_p0_noise : {'uniform', 'normal'}, optional
        Distribution applied to perturb initial constant guesses. ``None`` disables
        perturbations.
    refiner_p0_noise_kwargs : dict or {'default'} or None, optional
        Keyword arguments forwarded to the noise sampler. ``'default'`` yields
        ``{'low': -5, 'high': 5}`` for the uniform distribution.
    numpy_errors : {'ignore', 'warn', 'raise', 'call', 'print', 'log'} or None, optional
        Desired NumPy error handling strategy applied during constant refinement.
    parsimony : float, optional
        Penalty coefficient that discourages overly complex expressions.
    """

    def __init__(
            self,
            simplipy_engine: SimpliPyEngine,
            flash_ansr_transformer: FlashANSRModel,
            tokenizer: Tokenizer,
            generation_config: GenerationConfig | None = None,
            numeric_head: bool = False,
            n_restarts: int = 8,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'uniform',
            refiner_p0_noise_kwargs: dict | None | Literal['default'] = 'default',
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.05):
        self.simplipy_engine = simplipy_engine
        self.flash_ansr_transformer = flash_ansr_transformer.eval()
        self.tokenizer = tokenizer

        if refiner_p0_noise_kwargs == 'default':
            refiner_p0_noise_kwargs = {'low': -5, 'high': 5}

        if generation_config is None:
            generation_config = GenerationConfig()

        self.generation_config = generation_config
        self.numeric_head = numeric_head
        self.n_restarts = n_restarts
        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        self.refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs) if refiner_p0_noise_kwargs is not None else None
        self.numpy_errors = numpy_errors
        self.parsimony = parsimony

        self._results: list[Result] = []
        self.results: pd.DataFrame = pd.DataFrame()

        self.variable_mapping: dict[str, str] = {}

    @classmethod
    def load(
            cls,
            directory: str,
            generation_config: GenerationConfig | None = None,
            numeric_head: bool = False,
            n_restarts: int = 1,
            refiner_method: Literal['curve_fit_lm', 'minimize_bfgs'] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'uniform',
            refiner_p0_noise_kwargs: dict | None | Literal['default'] = 'default',
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.05,
            device: str = 'cpu') -> "FlashANSR":
        """Instantiate a :class:`FlashANSR` model from a configuration directory.

        Parameters
        ----------
        directory : str
            Directory that contains ``model.yaml``, ``tokenizer.yaml`` and
            ``state_dict.pt`` artifacts.
        generation_config : GenerationConfig, optional
            Generation parameters to override defaults during candidate search.
        numeric_head : bool, optional
            Whether to enable the numeric head for constant prediction.
        n_restarts : int, optional
            Number of restarts passed to the refiner.
        refiner_method : {'curve_fit_lm', 'minimize_bfgs'}
            Optimization routine for constant fitting.
        refiner_p0_noise : {'uniform', 'normal'}, optional
            Distribution used to perturb initial constant guesses.
        refiner_p0_noise_kwargs : dict or {'default'} or None, optional
            Additional keyword arguments for the noise sampler. ``'default'``
            resolves to ``{'low': -5, 'high': 5}``.
        numpy_errors : {'ignore', 'warn', 'raise', 'call', 'print', 'log'} or None, optional
            NumPy floating-point error policy applied during refinement.
        parsimony : float, optional
            Parsimony coefficient used when compiling results.
        device : str, optional
            Torch device where the model weights will be loaded.

        Returns
        -------
        model : FlashANSR
            Fully initialized regressor ready for inference.
        """
        directory = substitute_root_path(directory)

        flash_ansr_transformer_path = os.path.join(directory, 'model.yaml')
        tokenizer_path = os.path.join(directory, 'tokenizer.yaml')

        model = FlashANSRModel.from_config(flash_ansr_transformer_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True, map_location=device))
        model.eval().to(device)

        tokenizer = Tokenizer.from_config(tokenizer_path)

        return cls(
            simplipy_engine=model.simplipy_engine,
            flash_ansr_transformer=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            numeric_head=numeric_head,
            n_restarts=n_restarts,
            refiner_method=refiner_method,
            refiner_p0_noise=refiner_p0_noise,
            refiner_p0_noise_kwargs=refiner_p0_noise_kwargs,
            numpy_errors=numpy_errors,
            parsimony=parsimony)

    @property
    def n_variables(self) -> int:
        """Number of variables the model was trained on."""
        return self.flash_ansr_transformer.encoder_max_n_variables - 1

    def _truncate_input(self, X: np.ndarray | torch.Tensor | pd.DataFrame) -> np.ndarray | torch.Tensor | pd.DataFrame:
        """Limit input features to the number of variables seen during training.

        Parameters
        ----------
        X : ndarray or Tensor or DataFrame
            Candidate input data whose trailing dimension enumerates variables.

        Returns
        -------
        truncated : ndarray or Tensor or DataFrame
            Input truncated to ``self.n_variables`` columns when necessary.

        Raises
        ------
        ValueError
            If the input cannot be sliced to the expected number of variables.
        """
        if X.shape[-1] <= self.n_variables:
            return X

        warnings.warn(f"Input data has more variables than the model was trained on. The model was trained on {self.n_variables=} variables, but the input data has {X.shape[-1]=} variables. X and y will be truncated to {self.n_variables} variables.")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, :self.n_variables]

        try:
            return X[..., :self.n_variables]
        except IndexError:
            try:
                return X[:, :self.n_variables]
            except IndexError as exc:
                raise ValueError('Cannot truncate the input data') from exc

    def generate(self, data: torch.Tensor, complexity: int | float | None = None, verbose: bool = False) -> tuple[list[list[int]], list[float], list[bool]]:
        """Generate candidate expression beams from the transformer.

        Parameters
        ----------
        data : torch.Tensor
            Batched input tensor where the final feature corresponds to targets.
        complexity : int or float or None, optional
            Target expression complexity supplied to the generator.
        verbose : bool, optional
            If ``True``, progress output is emitted where supported.

        Returns
        -------
        beams : list[list[int]]
            Raw token sequences proposed by the transformer.
        log_probs : list[float]
            Log probabilities associated with each beam.
        completed : list[bool]
            Flags indicating whether the beam terminated with an end token.

        Raises
        ------
        ValueError
            If an unsupported generation method is requested.
        """
        match self.generation_config.method:
            case 'beam_search':
                return self.flash_ansr_transformer.beam_search(
                    data=data,
                    complexity=complexity,
                    verbose=verbose,
                    **self.generation_config)
            case 'softmax_sampling':
                return self.flash_ansr_transformer.sample_top_kp(
                    data=data,
                    complexity=complexity,
                    verbose=verbose,
                    **self.generation_config)
            case _:
                raise ValueError(f"Invalid generation method: {self.generation_config.method}")

    def fit(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            complexity: int | float | Iterable | None = None,
            variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto',
            converge_error: Literal['raise', 'ignore', 'print'] = 'ignore',
            verbose: bool = False) -> None:
        """Perform symbolic regression on ``(X, y)`` and refine candidate expressions.

        Parameters
        ----------
        X : ndarray or Tensor or DataFrame
            Feature matrix where rows index observations and columns variables.
        y : ndarray or Tensor or DataFrame or Series
            Target values. Multi-output targets are unsupported.
        complexity : int or float or Iterable or None, optional
            Desired expression complexity. Iterables allow sweeping multiple
            complexity targets. ``None`` defers to the generator defaults.
        variable_names : list[str] or dict[str, str] or {'auto'} or None, optional
            Mapping from internal variable tokens to descriptive names.
        converge_error : {'raise', 'ignore', 'print'}, optional
            Handling strategy when the refiner fails to converge.
        verbose : bool, optional
            If ``True`` progress bars and diagnostic output are displayed.

        Raises
        ------
        ValueError
            If ``y`` has more than one output dimension or cannot be reshaped.
        """

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif y.shape[-1] != 1:
            raise ValueError("The target data must have a single output dimension")

        X = self._truncate_input(X)

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

        # Normalize ``complexity`` into an iterable so downstream logic can iterate uniformly.
        if complexity is None or not hasattr(complexity, '__iter__'):
            complexity_list: list[int | float | None] = [complexity]
        else:
            complexity_list = complexity  # type: ignore

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

            y_variance = y.var(dim=0).item()

            X = pad_input_set(X, self.n_variables)

            # Concatenate x and y along the feature dimension
            data_tensor = torch.cat([X, y], dim=-1)

            self._results = []

            # Temporarily adopt the configured floating-point error policy for refinement.
            numpy_errors_before = np.geterr()
            np.seterr(all=self.numpy_errors)

            for complexity in complexity_list:
                raw_beams, log_probs, _ = self.generate(data_tensor, complexity=complexity, verbose=verbose)

                beams = [self.flash_ansr_transformer.tokenizer.extract_expression_from_beam(raw_beam)[0] for raw_beam in raw_beams]

                raw_beams_decoded = [self.tokenizer.decode(raw_beam, special_tokens='<constant>') for raw_beam in raw_beams]
                beams_decoded = [self.tokenizer.decode(beam, special_tokens='<constant>') for beam in beams]

                # Fit the refiner to each beam
                for raw_beam, raw_beam_decoded, beam, beam_decoded, log_prob in tqdm(zip(raw_beams, raw_beams_decoded, beams, beams_decoded, log_probs), desc="Fitting Constants", disable=not verbose, total=len(beams)):
                    if self.simplipy_engine.is_valid(beam_decoded):
                        numeric_prediction = None

                        if self.numeric_head:
                            raise NotImplementedError("Numeric head is not yet implemented")

                        try:
                            refiner = Refiner(simplipy_engine=self.simplipy_engine, n_variables=self.n_variables).fit(
                                expression=beam_decoded,
                                X=X.cpu().numpy(),
                                y=y.cpu().numpy(),
                                n_restarts=self.n_restarts,
                                method=self.refiner_method,
                                p0=numeric_prediction,
                                p0_noise=self.refiner_p0_noise,
                                p0_noise_kwargs=self.refiner_p0_noise_kwargs,
                                converge_error=converge_error)

                            if not refiner.valid_fit:  # Fit failed
                                fvu = np.nan
                                score = np.nan
                            else:
                                if y.shape[0] == 1:
                                    # Cannot compute variance for a single sample. Use loss instead.
                                    fvu = refiner._all_constants_values[0][-1]
                                else:
                                    fvu = refiner._all_constants_values[0][-1] / np.clip(y_variance, np.finfo(np.float32).eps, None)
                                score = np.log10(fvu) + self.parsimony * len(beam_decoded)

                            self._results.append({
                                'log_prob': log_prob,
                                'fvu': fvu,
                                'score': score,
                                'expression': beam_decoded,
                                'complexity': len(beam_decoded),
                                'target_complexity': complexity,
                                'numeric_prediction': numeric_prediction,
                                'raw_beam': raw_beam,
                                'beam': beam,    # type: ignore
                                'raw_beam_decoded': raw_beam_decoded,
                                'function': refiner.expression_lambda,
                                'refiner': refiner,
                                'fits': copy.deepcopy(refiner._all_constants_values),
                            })

                        except ConvergenceError:
                            if converge_error == 'print':
                                print(f"Failed to converge for beam: {beam_decoded}")

            self.compile_results(self.parsimony)

            np.seterr(**numpy_errors_before)

    def compile_results(self, parsimony: float) -> None:
        """Aggregate refiner outputs into a tidy :class:`pandas.DataFrame`.

        Parameters
        ----------
        parsimony : float
            Parsimony coefficient used to recompute scores before ranking.

        Raises
        ------
        ConvergenceError
            If no beams converged during refinement.
        """
        if not self._results:
            raise ConvergenceError("The optimization did not converge for any beam")

        self.initial_parsimony = self.parsimony
        self.parsimony = parsimony

        # Compute the new score for each result
        for result in self._results:
            if 'score' in result:
                # Recompute the score with the new parsimony coefficient
                result['score'] = np.log10(result['fvu']) + self.parsimony * len(result['expression'])

        # Sort the results by the best loss of each beam
        self._results = list(sorted(self._results, key=lambda x: (
            x['score'] if not np.isnan(x['score']) else float('inf'),
            np.isnan(x['score'])
        )))

        # Create a dataframe
        self.results = pd.DataFrame(self._results)

        # Explode the fits for each beam
        self.results = self.results.explode('fits')
        self.results['beam_id'] = self.results.index
        self.results.reset_index(drop=True, inplace=True)

        # Split the fit tuples into columns
        fits_columns = pd.DataFrame(self.results['fits'].tolist(), columns=['fit_constants', 'fit_covariances', 'fit_loss'])
        self.results = pd.concat([self.results, fits_columns], axis=1)
        self.results.drop(columns=['fits'], inplace=True)

    def predict(self, X: np.ndarray | torch.Tensor | pd.DataFrame, nth_best_beam: int = 0, nth_best_constants: int = 0) -> np.ndarray:
        """Evaluate a fitted expression on new data.

        Parameters
        ----------
        X : ndarray or Tensor or DataFrame
            Feature matrix to evaluate.
        nth_best_beam : int, optional
            Beam index to select from the ranked results.
        nth_best_constants : int, optional
            Index of the constant fit to choose for the selected beam.

        Returns
        -------
        y_pred : ndarray
            Predicted targets with the same leading dimension as ``X``.

        Raises
        ------
        ValueError
            If the model has not been fitted before prediction.
        """
        X = self._truncate_input(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = pad_input_set(X, self.n_variables)

        if len(self._results) == 0:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")

        return self._results[nth_best_beam]['refiner'].predict(X, nth_best_constants=nth_best_constants)

    def get_expression(self, nth_best_beam: int = 0, nth_best_constants: int = 0, return_prefix: bool = False, precision: int = 2, map_variables: bool = True, **kwargs: Any) -> list[str] | str:
        """Retrieve a formatted expression from the compiled results.

        Parameters
        ----------
        nth_best_beam : int, optional
            Beam index to extract from ``self._results``.
        nth_best_constants : int, optional
            Constant fit index for the selected beam.
        return_prefix : bool, optional
            If ``True`` return the prefix notation instead of infix string.
        precision : int, optional
            Number of decimal places used when rendering constants.
        map_variables : bool, optional
            When ``True`` apply ``self.variable_mapping`` to humanise variables.
        **kwargs : Any
            Extra keyword arguments forwarded to :meth:`Refiner.transform`.

        Returns
        -------
        expression : list[str] or str
            Expression either as a token list or human-readable string.
        """
        return self._results[nth_best_beam]['refiner'].transform(
            expression=self._results[nth_best_beam]['expression'],
            nth_best_constants=nth_best_constants,
            return_prefix=return_prefix,
            precision=precision,
            variable_mapping=self.variable_mapping if map_variables else None,
            **kwargs)

    def to(self, device: str) -> "FlashANSR":
        """Move the transformer weights to ``device``.

        Parameters
        ----------
        device : str
            Target torch device (e.g. ``'cpu'`` or ``'cuda:0'``).

        Returns
        -------
        model : FlashANSR
            Self, enabling fluent chaining.
        """
        self.flash_ansr_transformer.to(device)
        return self

    def eval(self) -> "FlashANSR":
        """Put the transformer into evaluation mode.

        Returns
        -------
        model : FlashANSR
            Self, enabling fluent chaining.
        """
        self.flash_ansr_transformer.eval()
        return self
