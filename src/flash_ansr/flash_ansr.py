import os
import copy
import numbers
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import TracebackType
from typing import Literal, Any, Iterable, Iterator, TypedDict, Callable, Tuple, Sequence, TypeVar
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.base import BaseEstimator

from simplipy import SimpliPyEngine

from flash_ansr.generation import run_beam_search, run_softmax_sampling, run_mcts_generation
from flash_ansr.model import FlashANSRModel, Tokenizer
from flash_ansr.decoding.mcts import MCTSConfig
from flash_ansr.preprocessing import PromptPrefix, prepare_prompt_prefix
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.utils.generation import GenerationConfig, SoftmaxSamplingConfig
from flash_ansr.utils.paths import substitute_root_path
from flash_ansr.utils.tensor_ops import pad_input_set
from flash_ansr.results import (
    RESULTS_FORMAT_VERSION,
    deserialize_results_payload,
    load_results_payload,
    save_results_payload,
    serialize_results_payload,
)


class Result(TypedDict):
    refiner: Refiner
    beam: list[int]
    log_prob: float
    expression: list[str]
    raw_beam: list[int]
    raw_beam_decoded: str
    complexity: int
    function: Callable
    fits: list[tuple[np.ndarray, np.ndarray, float]]
    score: float
    requested_complexity: int | float | None
    fvu: float
    prompt_metadata: dict[str, list[list[str]]] | None


_GLOBAL_SIMPLIPY_ENGINE: SimpliPyEngine | None = None
_GLOBAL_REFINEMENT_DATA: dict[str, np.ndarray | None] = {'X': None, 'y': None}


_T = TypeVar('_T')


def _iterate_with_progress(iterable: Iterable[_T], total: int, verbose: bool, desc: str) -> Iterator[_T]:
    if not verbose:
        yield from iterable
        return

    yield from tqdm(iterable, total=total, desc=desc, smoothing=0.0)


class _RefinementContext:
    def __init__(self, engine: SimpliPyEngine, inputs: np.ndarray, targets: np.ndarray) -> None:
        self._engine = engine
        self._inputs = inputs
        self._targets = targets
        self._previous_engine: SimpliPyEngine | None = None
        self._previous_data: dict[str, np.ndarray | None] | None = None

    def __enter__(self) -> None:
        global _GLOBAL_SIMPLIPY_ENGINE, _GLOBAL_REFINEMENT_DATA
        self._previous_engine = _GLOBAL_SIMPLIPY_ENGINE
        self._previous_data = _GLOBAL_REFINEMENT_DATA.copy()
        _GLOBAL_SIMPLIPY_ENGINE = self._engine
        _GLOBAL_REFINEMENT_DATA = {'X': self._inputs, 'y': self._targets}

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        global _GLOBAL_SIMPLIPY_ENGINE, _GLOBAL_REFINEMENT_DATA
        _GLOBAL_SIMPLIPY_ENGINE = self._previous_engine
        if self._previous_data is not None:
            _GLOBAL_REFINEMENT_DATA = self._previous_data
        else:
            _GLOBAL_REFINEMENT_DATA = {'X': None, 'y': None}


def _resolve_refinement_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    X = payload.get('X')
    y = payload.get('y')
    if X is None or y is None:
        X = _GLOBAL_REFINEMENT_DATA.get('X')
        y = _GLOBAL_REFINEMENT_DATA.get('y')
    if X is None or y is None:
        raise RuntimeError("Refinement worker is missing shared input data.")
    return X, y


def _refine_candidate_worker(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    simplipy_engine = payload.get('simplipy_engine') or _GLOBAL_SIMPLIPY_ENGINE
    if simplipy_engine is None:
        raise RuntimeError("Refinement worker does not have access to a SimpliPyEngine instance.")

    numpy_errors = payload.get('numpy_errors')
    numpy_state = np.geterr()
    if numpy_errors is not None:
        np.seterr(all=numpy_errors)

    seed = payload.get('seed')
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    X, y = _resolve_refinement_arrays(payload)

    try:
        refiner = Refiner(simplipy_engine=simplipy_engine, n_variables=payload['n_variables']).fit(
            expression=payload['expression'],
            X=X,
            y=y,
            n_restarts=payload['n_restarts'],
            method=payload['method'],
            p0=None,
            p0_noise=payload['p0_noise'],
            p0_noise_kwargs=payload['p0_noise_kwargs'],
            converge_error=payload['converge_error'],
        )
    except ConvergenceError:
        if payload['converge_error'] == 'raise':
            raise
        warning = f"Failed to converge for beam: {payload['expression']}" if payload['converge_error'] == 'print' else None
    finally:
        np.seterr(**numpy_state)

    if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
        warning = f"Failed to converge for beam: {payload['expression']}" if payload['converge_error'] == 'print' else None
        if payload['converge_error'] == 'raise':
            raise ConvergenceError("The optimization did not converge")
        return None, warning

    loss = float(refiner._all_constants_values[0][-1])
    sample_count = int(y.shape[0])
    fvu = FlashANSR._compute_fvu(loss, sample_count, payload['y_variance'])
    score = FlashANSR._score_from_fvu(fvu, len(payload['expression']), payload['parsimony'])

    serialized_fits: list[tuple[np.ndarray, np.ndarray | None, float]] = []
    for constants, constants_cov, fit_loss in refiner._all_constants_values:
        cov_payload: np.ndarray | None
        if constants_cov is None or getattr(constants_cov, 'size', 0) == 0:
            cov_payload = None
        else:
            cov_payload = np.asarray(constants_cov)
        serialized_fits.append((np.asarray(constants), cov_payload, float(fit_loss)))

    metadata_snapshot = payload.get('metadata_snapshot')
    result = {
        'log_prob': payload['log_prob'],
        'fvu': fvu,
        'score': score,
        'expression': payload['expression'],
        'complexity': len(payload['expression']),
        'requested_complexity': payload['complexity'],
        'raw_beam': payload['raw_beam'],
        'beam': payload['beam'],
        'raw_beam_decoded': payload['raw_beam_decoded'],
        'fits': serialized_fits,
        'valid_fit': refiner.valid_fit,
        'prompt_metadata': copy.deepcopy(metadata_snapshot) if metadata_snapshot is not None else None,
    }

    return result, None


class FlashANSR(BaseEstimator):
    """Flash Amortized Neural Symbolic Regressor.

    Parameters
    ----------
    simplipy_engine : SimpliPyEngine
        Engine responsible for manipulating and evaluating symbolic expressions.
    flash_ansr_model : FlashANSRModel
        Trained transformer backbone that proposes expression programs.
    tokenizer : Tokenizer
        Tokenizer mapping model outputs to expression tokens.
    generation_config : GenerationConfig, optional
        Configuration that controls candidate generation. If ``None`` a default
        ``SoftmaxSamplingConfig`` is created.
    n_restarts : int, optional
        Number of optimizer restarts used by the refiner when fitting constants.
    refiner_method : {'curve_fit_lm', 'minimize_bfgs', 'minimize_lbfgsb', 'minimize_neldermead', 'minimize_powell', 'least_squares_trf', 'least_squares_dogbox'}
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
    refiner_workers : int or None, optional
        Number of worker processes to run during constant refinement. ``None``
        (the default) uses all available CPU cores, while explicit integers
        select a fixed pool size. Set ``0`` to disable multiprocessing.
    """

    FLOAT64_EPS: float = float(np.finfo(np.float64).eps)

    @classmethod
    def _normalize_variance(cls, variance: float) -> float:
        if not np.isfinite(variance):
            return cls.FLOAT64_EPS
        return max(float(variance), cls.FLOAT64_EPS)

    @classmethod
    def _compute_fvu(cls, loss: float, sample_count: int, variance: float) -> float:
        if sample_count <= 1:
            return float(loss)
        return float(loss) / cls._normalize_variance(variance)

    @classmethod
    def _score_from_fvu(cls, fvu: float, complexity: int, parsimony: float) -> float:
        if not np.isfinite(fvu) or fvu <= 0:
            safe_fvu = cls.FLOAT64_EPS
        else:
            safe_fvu = max(float(fvu), cls.FLOAT64_EPS)
        return float(np.log10(safe_fvu) + parsimony * complexity)

    def __init__(
            self,
            simplipy_engine: SimpliPyEngine,
            flash_ansr_model: FlashANSRModel,
            tokenizer: Tokenizer,
            generation_config: GenerationConfig | None = None,
            n_restarts: int = 8,
            refiner_method: Literal[
                'curve_fit_lm',
                'minimize_bfgs',
                'minimize_lbfgsb',
                'minimize_neldermead',
                'minimize_powell',
                'least_squares_trf',
                'least_squares_dogbox',
            ] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            refiner_p0_noise_kwargs: dict | None | Literal['default'] = 'default',
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.05,
            refiner_workers: int | None = None):
        self.simplipy_engine = simplipy_engine
        self.flash_ansr_model = flash_ansr_model.eval()
        self.tokenizer = tokenizer

        if refiner_p0_noise_kwargs == 'default':
            refiner_p0_noise_kwargs = {'low': -5, 'high': 5}

        if generation_config is None:
            generation_config = SoftmaxSamplingConfig()

        self.generation_config = generation_config
        self.n_restarts = n_restarts
        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        self.refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs) if refiner_p0_noise_kwargs is not None else None
        self.numpy_errors = numpy_errors
        self.parsimony = parsimony

        cpu_count = os.cpu_count() or 1

        if refiner_workers is None:
            resolved_workers = max(1, cpu_count)
        elif isinstance(refiner_workers, numbers.Integral):
            resolved_workers = max(0, int(refiner_workers))
        else:
            raise TypeError("refiner_workers must be an integer or None.")

        self.refiner_workers = resolved_workers

        self._results: list[Result] = []
        self.results: pd.DataFrame = pd.DataFrame()
        self._mcts_cache: dict[Tuple[int, ...], dict[str, Any]] = {}

        self._input_dim: int | None = None

        self.variable_mapping: dict[str, str] = {}
        self._prompt_prefix: PromptPrefix | None = None
        self._prompt_metadata: dict[str, list[list[str]]] | None = None

    @classmethod
    def load(
            cls,
            directory: str,
            generation_config: GenerationConfig | None = None,
            n_restarts: int = 1,
            refiner_method: Literal[
                'curve_fit_lm',
                'minimize_bfgs',
                'minimize_lbfgsb',
                'minimize_neldermead',
                'minimize_powell',
                'least_squares_trf',
                'least_squares_dogbox',
            ] = 'curve_fit_lm',
            refiner_p0_noise: Literal['uniform', 'normal'] | None = 'normal',
            refiner_p0_noise_kwargs: dict | None | Literal['default'] = 'default',
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            parsimony: float = 0.05,
            device: str = 'cpu',
            refiner_workers: int | None = None) -> "FlashANSR":
        """Instantiate a `FlashANSR` model from a configuration directory.

        Parameters
        ----------
        directory : str
            Directory that contains ``model.yaml``, ``tokenizer.yaml`` and
            ``state_dict.pt`` artifacts.
        generation_config : GenerationConfig, optional
            Generation parameters to override defaults during candidate search.
        n_restarts : int, optional
            Number of restarts passed to the refiner.
        refiner_method : {'curve_fit_lm', 'minimize_bfgs', 'minimize_lbfgsb', 'minimize_neldermead', 'minimize_powell', 'least_squares_trf', 'least_squares_dogbox'}
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
        refiner_workers : int or None, optional
            Desired worker-pool size for constant refinement. ``None`` uses the
            number of available CPU cores, integers select an explicit pool size,
            and ``0`` disables multiprocessing. Mirrors the constructor parameter.

        Returns
        -------
        model : FlashANSR
            Fully initialized regressor ready for inference.
        """
        directory = substitute_root_path(directory)

        flash_ansr_model_path = os.path.join(directory, 'model.yaml')
        tokenizer_path = os.path.join(directory, 'tokenizer.yaml')

        model = FlashANSRModel.from_config(flash_ansr_model_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True, map_location=device))
        model.eval().to(device)

        tokenizer = Tokenizer.from_config(tokenizer_path)

        return cls(
            simplipy_engine=model.simplipy_engine,
            flash_ansr_model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            n_restarts=n_restarts,
            refiner_method=refiner_method,
            refiner_p0_noise=refiner_p0_noise,
            refiner_p0_noise_kwargs=refiner_p0_noise_kwargs,
            numpy_errors=numpy_errors,
            parsimony=parsimony,
            refiner_workers=refiner_workers)

    @property
    def n_variables(self) -> int:
        """Number of variables the model was trained on."""
        return self.flash_ansr_model.encoder_max_n_variables - 1

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

    def _build_mcts_config(self) -> MCTSConfig:
        cfg = self.generation_config

        reward_transform = getattr(cfg, 'reward_transform', None)
        if reward_transform is not None and not callable(reward_transform):
            reward_transform = None

        return MCTSConfig(
            simulations=getattr(cfg, 'simulations', 256),
            uct_c=getattr(cfg, 'uct_c', 1.4),
            expansion_top_k=getattr(cfg, 'expansion_top_k', 32),
            max_depth=getattr(cfg, 'max_depth', getattr(cfg, 'max_len', 64)),
            rollout_max_len=getattr(cfg, 'rollout_max_len', None),
            rollout_policy=getattr(cfg, 'rollout_policy', 'sample'),
            temperature=getattr(cfg, 'temperature', 1.0),
            dirichlet_alpha=getattr(cfg, 'dirichlet_alpha', None),
            dirichlet_epsilon=getattr(cfg, 'dirichlet_epsilon', 0.25),
            invalid_penalty=getattr(cfg, 'invalid_penalty', 1e6),
            min_visits_before_expansion=getattr(cfg, 'min_visits_before_expansion', 1),
            reward_transform=reward_transform,
        )

    def generate(
        self,
        data: torch.Tensor,
        *,
        prompt_prefix: PromptPrefix | None = None,
        complexity: int | float | None = None,
        verbose: bool = False,
    ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        """Generate candidate expression beams from the transformer.

        Parameters
        ----------
        data : torch.Tensor
            Batched input tensor where the final feature corresponds to targets.
        prompt_prefix : PromptPrefix or None, optional
            Serialized prompt metadata to seed generation. When omitted, the
            method will synthesize a minimal prefix from ``complexity``
            (if provided) so that all prompt hints share the same entry point.
        complexity : int or float or None, optional
            Numeric prompt hint that is only used when ``prompt_prefix`` is not
            supplied. Callers should prefer constructing a full
            `PromptPrefix` via the preprocessing pipeline.
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
        rewards : list[float]
            Decoder-specific score used during search (``nan`` for methods that do
            not compute one).

        Raises
        ------

        ValueError
            If an unsupported generation method is requested.
        """
        self._mcts_cache = {}

        generation_kwargs = self.generation_config.to_kwargs()

        effective_prompt = prompt_prefix
        if effective_prompt is None and complexity is not None:
            preprocessor = getattr(self.flash_ansr_model, 'preprocessor', None)
            effective_prompt = prepare_prompt_prefix(
                preprocessor,
                complexity=complexity,
                allowed_terms=None,
                include_terms=None,
                exclude_terms=None,
            )

        match self.generation_config.method:
            case 'beam_search':
                beams, log_probs, completed, rewards = run_beam_search(
                    self.flash_ansr_model,
                    data=data,
                    verbose=verbose,
                    prompt_prefix=effective_prompt,
                    generation_kwargs=generation_kwargs,
                )
                return beams, log_probs, completed, rewards
            case 'softmax_sampling':
                beams, log_probs, completed, rewards = run_softmax_sampling(
                    self.flash_ansr_model,
                    data=data,
                    verbose=verbose,
                    prompt_prefix=effective_prompt,
                    generation_kwargs=generation_kwargs,
                )
                return beams, log_probs, completed, rewards
            case 'mcts':
                config = self._build_mcts_config()
                completion_sort = generation_kwargs.get('completion_sort', 'reward')
                beam_width = generation_kwargs.get('beam_width', 16)

                beams, log_probs, completed, rewards, refiner_cache = run_mcts_generation(
                    transformer=self.flash_ansr_model,
                    tokenizer=self.tokenizer,
                    simplipy_engine=self.simplipy_engine,
                    data=data,
                    config=config,
                    beam_width=beam_width,
                    completion_sort=completion_sort,
                    n_variables=self.n_variables,
                    n_restarts=self.n_restarts,
                    refiner_method=self.refiner_method,
                    refiner_p0_noise=self.refiner_p0_noise,
                    refiner_p0_noise_kwargs=self.refiner_p0_noise_kwargs,
                    parsimony=self.parsimony,
                    compute_fvu=self._compute_fvu,
                    score_from_fvu=self._score_from_fvu,
                    float64_eps=self.FLOAT64_EPS,
                    prompt_prefix=effective_prompt,
                    verbose=verbose,
                )

                self._mcts_cache = refiner_cache
                return beams, log_probs, completed, rewards
            case _:
                raise ValueError(f"Invalid generation method: {self.generation_config.method}")

    def _prepare_prompt_prefix(
            self,
            *,
            complexity: int | float | None,
            allowed_terms: Iterable[Sequence[Any]] | None,
            include_terms: Iterable[Sequence[Any]] | None,
            exclude_terms: Iterable[Sequence[Any]] | None) -> PromptPrefix | None:
        preprocessor = getattr(self.flash_ansr_model, 'preprocessor', None)
        prompt_prefix = prepare_prompt_prefix(
            preprocessor,
            complexity=complexity,
            allowed_terms=allowed_terms,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

        self._prompt_prefix = prompt_prefix
        return prompt_prefix

    def _create_result_entry(self, *, payload: dict[str, Any], input_dim: int) -> Result | None:
        fits_payload = payload.get('fits')
        if not fits_payload:
            return None

        if not payload.get('valid_fit', False):
            return None

        refiner = Refiner.from_serialized(
            simplipy_engine=self.simplipy_engine,
            n_variables=self.n_variables,
            expression=payload['expression'],
            n_inputs=input_dim,
            fits=fits_payload,
        )

        if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
            return None

        entry: Result = {
            'log_prob': payload['log_prob'],
            'fvu': payload['fvu'],
            'score': payload['score'],
            'expression': payload['expression'],
            'complexity': payload['complexity'],
            'requested_complexity': payload.get('requested_complexity'),
            'raw_beam': payload['raw_beam'],
            'beam': payload['beam'],
            'raw_beam_decoded': payload['raw_beam_decoded'],
            'function': refiner.expression_lambda,
            'refiner': refiner,
            'fits': copy.deepcopy(refiner._all_constants_values),
            'prompt_metadata': copy.deepcopy(payload.get('prompt_metadata')) if payload.get('prompt_metadata') is not None else None,
        }

        return entry

    def fit(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto',
            converge_error: Literal['raise', 'ignore', 'print'] = 'ignore',
            verbose: bool = False,
            *,
            complexity: int | float | None = None,
            allowed_terms: Iterable[Sequence[Any]] | None = None,
            include_terms: Iterable[Sequence[Any]] | None = None,
            exclude_terms: Iterable[Sequence[Any]] | None = None) -> None:
        """Perform symbolic regression on ``(X, y)`` and refine candidate expressions.

        Parameters
        ----------
        X : ndarray or Tensor or DataFrame
            Feature matrix where rows index observations and columns variables.
        y : ndarray or Tensor or DataFrame or Series
            Target values. Multi-output targets are unsupported.
        variable_names : list[str] or dict[str, str] or {'auto'} or None, optional
            Mapping from internal variable tokens to descriptive names.
        converge_error : {'raise', 'ignore', 'print'}, optional
            Handling strategy when the refiner fails to converge.
        verbose : bool, optional
            If ``True`` progress bars and diagnostic output are displayed.
        allowed_terms : Iterable[Sequence[str]] or None, optional
            Keyword-only list of term token sequences that may appear in the
            generated expression.
        include_terms : Iterable[Sequence[str]] or None, optional
            Keyword-only subset of allowed terms that the expression should
            prioritise using.
        exclude_terms : Iterable[Sequence[str]] or None, optional
            Keyword-only list of term token sequences that should be discouraged
            during generation.

        Raises
        ------
        ValueError
            If ``y`` has more than one output dimension or cannot be reshaped.
        """
        # TODO: Support lists
        # TODO: Support 0-d and 1-d tensors

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

        if complexity is not None and not isinstance(complexity, numbers.Real):
            raise TypeError("complexity must be a real scalar when provided")

        with torch.no_grad():
            # Convert the input data to a tensor
            if not isinstance(X, torch.Tensor):
                if isinstance(X, pd.DataFrame):
                    X = torch.tensor(X.values, dtype=torch.float32, device=self.flash_ansr_model.device)
                else:
                    X = torch.tensor(X, dtype=torch.float32, device=self.flash_ansr_model.device)
            else:
                X = X.to(self.flash_ansr_model.device)

            if not isinstance(y, torch.Tensor):
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = torch.tensor(y.values, dtype=torch.float32, device=self.flash_ansr_model.device)
                else:
                    y = torch.tensor(y, dtype=torch.float32, device=self.flash_ansr_model.device)
            else:
                y = y.to(self.flash_ansr_model.device)

            if y.dim() == 1:
                y = y.unsqueeze(-1)

            sample_count = y.shape[0]
            if sample_count <= 1:
                # Torch warns when computing an unbiased variance with a single sample.
                # Skip the reduction entirely so downstream scoring quietly falls back
                # to the residual loss via ``_compute_fvu``.
                y_variance = float('nan')
            else:
                y_variance = y.var(dim=0).item()

            X = pad_input_set(X, self.n_variables)

            # Concatenate x and y along the feature dimension
            data_tensor = torch.cat([X, y], dim=-1)

            self._results = []

            # Temporarily adopt the configured floating-point error policy for refinement.
            numpy_errors_before = np.geterr()
            np.seterr(all=self.numpy_errors)

            prompt_prefix = self._prepare_prompt_prefix(
                complexity=complexity,
                allowed_terms=allowed_terms,
                include_terms=include_terms,
                exclude_terms=exclude_terms,
            )

            metadata_snapshot: dict[str, list[list[str]]] | None
            if prompt_prefix is not None:
                metadata_snapshot = copy.deepcopy(prompt_prefix.metadata)
            else:
                metadata_snapshot = None

            self._prompt_metadata = copy.deepcopy(metadata_snapshot) if metadata_snapshot is not None else None

            raw_beams, log_probs, _completed_flags, _rewards = self.generate(
                data_tensor,
                prompt_prefix=prompt_prefix,
                complexity=complexity,
                verbose=verbose,
            )

            beams = [self.flash_ansr_model.tokenizer.extract_expression_from_beam(raw_beam)[0] for raw_beam in raw_beams]

            raw_beams_decoded = [self.tokenizer.decode(raw_beam, special_tokens='<constant>') for raw_beam in raw_beams]
            beams_decoded = [self.tokenizer.decode(beam, special_tokens='<constant>') for beam in beams]

            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()

            refinement_jobs: list[dict[str, Any]] = []
            beam_iterator = zip(raw_beams, raw_beams_decoded, beams, beams_decoded, log_probs)
            for raw_beam, raw_beam_decoded, beam, beam_decoded, log_prob in beam_iterator:
                if not self.simplipy_engine.is_valid(beam_decoded):
                    continue

                job: dict[str, Any] = {
                    'raw_beam': raw_beam,
                    'raw_beam_decoded': raw_beam_decoded,
                    'beam': beam,
                    'expression': beam_decoded,
                    'log_prob': log_prob,
                    'n_variables': self.n_variables,
                    'n_restarts': self.n_restarts,
                    'method': self.refiner_method,
                    'p0_noise': self.refiner_p0_noise,
                    'p0_noise_kwargs': copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                    'converge_error': converge_error,
                    'numpy_errors': self.numpy_errors,
                    'y_variance': y_variance,
                    'parsimony': self.parsimony,
                    'complexity': complexity,
                    'metadata_snapshot': metadata_snapshot,
                }
                refinement_jobs.append(job)

            if refinement_jobs:
                available_methods = mp.get_all_start_methods()
                max_workers = min(self.refiner_workers, len(refinement_jobs))
                use_parallel = max_workers > 1 and 'fork' in available_methods

                input_dim = X_np.shape[1]
                self._input_dim = input_dim

                if max_workers > 1 and not use_parallel:
                    warnings.warn("Parallel refinement requires the 'fork' start method; falling back to serial execution.")

                with _RefinementContext(self.simplipy_engine, X_np, y_np):
                    if use_parallel:
                        ctx = mp.get_context('fork')
                        seed_sequence = np.random.SeedSequence()
                        spawned = seed_sequence.spawn(len(refinement_jobs))
                        for job, seq in zip(refinement_jobs, spawned):
                            job['seed'] = int(seq.generate_state(1, dtype=np.uint32)[0])

                        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                            futures = [executor.submit(_refine_candidate_worker, job) for job in refinement_jobs]
                            for future in _iterate_with_progress(
                                as_completed(futures),
                                total=len(futures),
                                verbose=verbose,
                                desc="Fitting Constants",
                            ):
                                result, warning_msg = future.result()
                                if warning_msg and converge_error == 'print':
                                    print(warning_msg)
                                if result is not None:
                                    entry = self._create_result_entry(payload=result, input_dim=input_dim)
                                    if entry is not None:
                                        self._results.append(entry)
                    else:
                        for job in _iterate_with_progress(
                            refinement_jobs,
                            total=len(refinement_jobs),
                            verbose=verbose,
                            desc="Fitting Constants",
                        ):
                            serial_payload = job.copy()
                            serial_payload.update({'X': X_np, 'y': y_np, 'simplipy_engine': self.simplipy_engine})
                            result, warning_msg = _refine_candidate_worker(serial_payload)
                            if warning_msg and converge_error == 'print':
                                print(warning_msg)
                            if result is not None:
                                entry = self._create_result_entry(payload=result, input_dim=input_dim)
                                if entry is not None:
                                    self._results.append(entry)

            self.compile_results(self.parsimony)

            np.seterr(**numpy_errors_before)

    def compile_results(self, parsimony: float) -> None:
        """Aggregate refiner outputs into a tidy `pandas.DataFrame`.

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
                fvu = result.get('fvu', np.nan)
                if np.isfinite(fvu):
                    result['score'] = self._score_from_fvu(float(fvu), len(result['expression']), self.parsimony)
                else:
                    result['score'] = np.nan

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
        # TODO: Support lists
        # TODO: Support 0-d and 1-d tensors

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

    def save_results(self, path: str) -> None:
        """Persist fitted results (minus lambdas) for later reuse."""

        if not self._results:
            raise ValueError("No results available to save. Run `fit` first.")

        input_dim = self._input_dim if self._input_dim is not None else self.n_variables
        metadata = {
            "format_version": RESULTS_FORMAT_VERSION,
            "parsimony": self.parsimony,
            "n_variables": self.n_variables,
            "input_dim": input_dim,
            "variable_mapping": copy.deepcopy(self.variable_mapping),
        }

        payload = serialize_results_payload(self._results, metadata=metadata)
        save_results_payload(payload, path)

    def load_results(self, path: str, *, rebuild_refiners: bool = True) -> None:
        """Load previously saved results and rebuild refiners if requested."""

        payload = load_results_payload(path)
        metadata = payload.get("metadata", {})

        version = int(payload.get("version", 0))
        if version != RESULTS_FORMAT_VERSION:
            warnings.warn(
                f"Results payload version {version} does not match expected {RESULTS_FORMAT_VERSION}; attempting to proceed anyway."
            )

        parsimony = float(metadata.get("parsimony", self.parsimony))
        n_variables = int(metadata.get("n_variables", self.n_variables))
        input_dim = int(metadata.get("input_dim", n_variables))

        self._input_dim = input_dim
        self.variable_mapping = metadata.get("variable_mapping", self.variable_mapping)

        restored = deserialize_results_payload(
            payload,
            simplipy_engine=self.simplipy_engine,
            n_variables=n_variables,
            input_dim=input_dim,
            rebuild_refiners=rebuild_refiners,
        )

        self._results = restored
        self.compile_results(parsimony)

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
        self.flash_ansr_model.to(device)
        return self

    def eval(self) -> "FlashANSR":
        """Put the transformer into evaluation mode.

        Returns
        -------
        model : FlashANSR
            Self, enabling fluent chaining.
        """
        self.flash_ansr_model.eval()
        return self
