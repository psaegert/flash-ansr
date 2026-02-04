import os
import copy
import math
import numbers
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import TracebackType
from typing import Literal, Any, Iterable, Iterator, TypedDict, Callable, Tuple, Sequence, TypeVar, cast
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
    constant_count: int
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
    pruned_variant: bool


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

    expression_tokens = payload['expression']
    constant_count = int(payload.get('constant_count', 0))
    if constant_count <= 0:
        constant_count = sum(1 for tok in expression_tokens if FlashANSR._is_constant_token(tok))
        payload['constant_count'] = constant_count

    score = FlashANSR._score_from_fvu(
        fvu,
        len(expression_tokens),
        constant_count,
        payload.get('log_prob'),
        payload['length_penalty'],
        payload['constants_penalty'],
        payload['likelihood_penalty'],
    )

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
        'constant_count': payload['constant_count'],
        'complexity': len(payload['expression']),
        'requested_complexity': payload['complexity'],
        'raw_beam': payload['raw_beam'],
        'beam': payload['beam'],
        'raw_beam_decoded': payload['raw_beam_decoded'],
        'fits': serialized_fits,
        'valid_fit': refiner.valid_fit,
        'prompt_metadata': copy.deepcopy(metadata_snapshot) if metadata_snapshot is not None else None,
        'pruned_variant': bool(payload.get('pruned_variant', False)),
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
        ``{'loc': 0.0, 'scale': 5.0}`` for the normal distribution.
    numpy_errors : {'ignore', 'warn', 'raise', 'call', 'print', 'log'} or None, optional
        Desired NumPy error handling strategy applied during constant refinement.
    length_penalty : float, optional
        Penalty coefficient that discourages overly long expressions.
    constants_penalty : float, optional
        Penalty coefficient applied to the number of constants present in an expression.
    likelihood_penalty : float, optional
        Penalty coefficient applied to the negative log likelihood of the generated beam.
    refiner_workers : int or None, optional
        Number of worker processes to run during constant refinement. ``None``
        (the default) uses all available CPU cores, while explicit integers
        select a fixed pool size. Set ``0`` to disable multiprocessing.
    prune_constant_budget : int or float, optional
        Apply constant-pruning refinement to the best beams after the initial
        refinement (ranked by FVU). If ``> 1`` treated as an absolute count; if
        ``0 < value <= 1`` treated as a fraction of beams (ceil). Set to 0 to
        disable. Defaults to 16.
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
    def _score_from_fvu(
            cls,
            fvu: float,
            complexity: int,
            constant_count: int,
            log_prob: float | None,
            length_penalty: float,
            constants_penalty: float,
            likelihood_penalty: float) -> float:
        if not np.isfinite(fvu) or fvu <= 0:
            safe_fvu = cls.FLOAT64_EPS
        else:
            safe_fvu = max(float(fvu), cls.FLOAT64_EPS)

        likelihood_term = 0.0
        if log_prob is not None and np.isfinite(log_prob):
            likelihood_term = likelihood_penalty * (-float(log_prob))

        return float(np.log10(safe_fvu)
                     + length_penalty * complexity
                     + constants_penalty * max(int(constant_count), 0)
                     + likelihood_term)

    def _score_log_probs_batch(
            self,
            *,
            sequences: list[list[int]],
            prompt_prefix: PromptPrefix | None,
            memory: torch.Tensor,
            device: torch.device) -> list[float]:
        if not sequences:
            return []

        base_tokens, _base_num = self.flash_ansr_model._resolve_generation_prefix(prompt_prefix=prompt_prefix)
        try:
            eos_id = int(self.tokenizer['<eos>'])
        except KeyError:
            eos_id = None
        pad_id = int(self.tokenizer['<pad>']) if '<pad>' in self.tokenizer else 0

        full_sequences: list[list[int]] = []
        lengths: list[int] = []
        for seq in sequences:
            extended = list(base_tokens) + list(seq)
            if eos_id is not None:
                extended = extended + [eos_id]
            full_sequences.append(extended)
            lengths.append(len(extended))

        max_len = max(lengths)
        batch_size = len(full_sequences)

        input_tokens = torch.full((batch_size, max_len), pad_id, device=device, dtype=torch.long)
        for i, seq in enumerate(full_sequences):
            input_tokens[i, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)

        with torch.no_grad():
            logits = self.flash_ansr_model.forward(input_tokens=input_tokens, data=None, input_num=None, memory=memory)
            log_probs = torch.log_softmax(logits, dim=-1)

        gathered: list[float] = []
        for i, seq_len in enumerate(lengths):
            if seq_len <= 1:
                gathered.append(float('-inf'))
                continue

            targets = input_tokens[i, 1:seq_len]
            step_log_probs = log_probs[i, :seq_len - 1]
            token_log_probs = step_log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
            gathered.append(float(token_log_probs.sum().item()))

        return gathered

    def _resolve_prune_count(self, candidate_total: int) -> int:
        if candidate_total <= 0:
            return 0

        k_value = self.prune_constant_budget
        if k_value <= 0:
            return 0

        if 0 < k_value <= 1:
            return min(candidate_total, max(1, int(math.ceil(candidate_total * k_value))))

        return min(candidate_total, int(math.floor(k_value)))

    @staticmethod
    def _is_constant_token(token: str) -> bool:
        if token == '<constant>':
            return True
        if token.startswith('C_') and token[2:].isdigit():
            return True
        if token in {'0', '1', '(-1)', 'np.pi', 'np.e', 'float("inf")', 'float("-inf")', 'float("nan")'}:
            return True
        try:
            float(token)
            return True
        except ValueError:
            return False

    @classmethod
    def _count_constants(cls, expression: Sequence[str]) -> int:
        return sum(1 for token in expression if cls._is_constant_token(token))

    def _get_operator_arity(self, token: str) -> int | None:
        aliases = getattr(self.simplipy_engine, 'operator_aliases', {})
        operator_arity = getattr(self.simplipy_engine, 'operator_arity_compat', {})
        return operator_arity.get(aliases.get(token, token))

    def _prefix_to_tree(self, expression: list[str]) -> list[Any] | str:
        aliases = getattr(self.simplipy_engine, 'operator_aliases', {})
        operator_arity = getattr(self.simplipy_engine, 'operator_arity_compat', {})

        stack: list[list[Any] | str] = []
        for token in reversed(expression):
            arity = operator_arity.get(aliases.get(token, token))
            if arity is None:
                stack.append(token)
                continue

            if len(stack) < arity:
                raise ValueError(f"Cannot build tree for expression: {expression}")

            children = [stack.pop() for _ in range(arity)]
            children.reverse()
            stack.append([token, *children])

        if len(stack) != 1:
            raise ValueError(f"Expression did not reduce to a single tree: {expression}")

        return stack[0]

    def _tree_to_prefix(self, node: list[Any] | str) -> list[str]:
        if isinstance(node, list):
            tokens: list[str] = [node[0]]
            for child in node[1:]:
                tokens.extend(self._tree_to_prefix(child))
            return tokens
        return [node]

    def _collapse_binary_after_constant_prune(
            self,
            operator: str,
            kept_child: list[Any] | str,
            removed_left: bool,
            removed_right: bool) -> list[Any] | str | None:
        canonical = getattr(self.simplipy_engine, 'operator_aliases', {}).get(operator, operator)
        has_neg = self._get_operator_arity('neg') == 1
        has_inv = self._get_operator_arity('inv') == 1

        if removed_left and removed_right:
            return None

        if canonical in {'+', '*'}:
            return kept_child

        if canonical == '-':
            if removed_left and not removed_right:
                if has_neg:
                    return ['neg', kept_child]
                return ['*', '(-1)', kept_child]
            return kept_child

        if canonical == '/':
            if removed_left:
                if has_inv:
                    return ['inv', kept_child]
                return ['/', '1', kept_child]
            return kept_child

        return kept_child

    def _prune_constants_from_tree(
            self,
            node: list[Any] | str,
            removal_ids: set[int],
            counter: list[int]) -> list[Any] | str | None:
        if isinstance(node, list):
            operator = node[0]
            children = node[1:]
            pruned_children: list[list | str | None] = []
            removed_flags: list[bool] = []

            for child in children:
                pruned_child = self._prune_constants_from_tree(child, removal_ids, counter)
                pruned_children.append(pruned_child)
                removed_flags.append(pruned_child is None)

            arity = len(children)
            if arity == 1:
                if removed_flags[0]:
                    return None
                return [operator, pruned_children[0]]

            if arity == 2:
                if removed_flags[0] or removed_flags[1]:
                    kept_child = pruned_children[0] if not removed_flags[0] else pruned_children[1]
                    if kept_child is None:
                        return None
                    return self._collapse_binary_after_constant_prune(operator, kept_child, removed_flags[0], removed_flags[1])
                return [operator, pruned_children[0], pruned_children[1]]

            kept_children = [child for child in pruned_children if child is not None]
            if not kept_children:
                return None
            if len(kept_children) == 1:
                return kept_children[0]
            return [operator, *kept_children]

        token = node
        if self._is_constant_token(token):
            idx = counter[0]
            counter[0] += 1
            if idx in removal_ids:
                return None

        return token

    def _generate_constant_pruning_variants(self, expression: list[str]) -> list[list[str]]:
        constant_count = sum(1 for token in expression if self._is_constant_token(token))
        if constant_count == 0:
            return [expression]

        try:
            tree = self._prefix_to_tree(expression)
        except ValueError:
            return [expression]

        variants: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        for mask in range(1 << constant_count):
            removal_ids = {idx for idx in range(constant_count) if mask & (1 << idx)}
            pruned_tree = self._prune_constants_from_tree(tree, removal_ids, counter=[0])
            if pruned_tree is None:
                continue

            prefix_variant = self._tree_to_prefix(pruned_tree)
            variant_key = tuple(prefix_variant)
            if variant_key in seen:
                continue

            seen.add(variant_key)
            variants.append(prefix_variant)

        return variants

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
            length_penalty: float = 0.05,
            constants_penalty: float = 0.0,
            likelihood_penalty: float = 0.0,
            refiner_workers: int | None = None,
            prune_constant_budget: float | int = 16):
        self.simplipy_engine = simplipy_engine
        self.flash_ansr_model = flash_ansr_model.eval()
        self.tokenizer = tokenizer

        if refiner_p0_noise_kwargs == 'default':
            refiner_p0_noise_kwargs = {'loc': 0.0, 'scale': 5.0}

        if generation_config is None:
            generation_config = SoftmaxSamplingConfig()

        self.generation_config = generation_config
        self.n_restarts = n_restarts
        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        self.refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs) if refiner_p0_noise_kwargs is not None else None
        self.numpy_errors = numpy_errors
        self.length_penalty = length_penalty
        self.constants_penalty = float(constants_penalty)
        self.likelihood_penalty = float(likelihood_penalty)
        self.prune_constant_budget = max(0.0, float(prune_constant_budget))

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
            length_penalty: float = 0.05,
            constants_penalty: float = 0.0,
            likelihood_penalty: float = 0.0,
            device: str = 'cpu',
            refiner_workers: int | None = None,
            prune_constant_budget: float | int = 16) -> "FlashANSR":
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
            resolves to ``{'loc': 0.0, 'scale': 5.0}``.
        numpy_errors : {'ignore', 'warn', 'raise', 'call', 'print', 'log'} or None, optional
            NumPy floating-point error policy applied during refinement.
        length_penalty : float, optional
            Length penalty used when compiling results.
        constants_penalty : float, optional
            Penalty applied per constant present in the expression during
            scoring.
        likelihood_penalty : float, optional
            Penalty applied to the negative log likelihood of each beam.
        device : str, optional
            Torch device where the model weights will be loaded.
        refiner_workers : int or None, optional
            Desired worker-pool size for constant refinement. ``None`` uses the
            number of available CPU cores, integers select an explicit pool size,
            and ``0`` disables multiprocessing. Mirrors the constructor parameter.
        prune_constant_budget : int, optional
            Number of top beams (by FVU) or fraction of beams (if 0<value<=1)
            to prune after initial refinement when pruning is enabled.

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
            length_penalty=length_penalty,
            constants_penalty=constants_penalty,
            likelihood_penalty=likelihood_penalty,
            refiner_workers=refiner_workers,
            prune_constant_budget=prune_constant_budget)

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
                    length_penalty=self.length_penalty,
                    constants_penalty=self.constants_penalty,
                    likelihood_penalty=self.likelihood_penalty,
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
            'constant_count': int(payload.get('constant_count', self._count_constants(payload['expression']))),
            'complexity': payload['complexity'],
            'requested_complexity': payload.get('requested_complexity'),
            'raw_beam': payload['raw_beam'],
            'beam': payload['beam'],
            'raw_beam_decoded': payload['raw_beam_decoded'],
            'function': refiner.expression_lambda,
            'refiner': refiner,
            'fits': copy.deepcopy(refiner._all_constants_values),
            'prompt_metadata': copy.deepcopy(payload.get('prompt_metadata')) if payload.get('prompt_metadata') is not None else None,
            'pruned_variant': bool(payload.get('pruned_variant', False)),
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

            with torch.no_grad():
                memory_for_scoring = self.flash_ansr_model._create_memory(data_tensor)

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

                constant_count = self._count_constants(beam_decoded)

                job: dict[str, Any] = {
                    'raw_beam': raw_beam,
                    'raw_beam_decoded': raw_beam_decoded,
                    'beam': beam,
                    'expression': beam_decoded,
                    'log_prob': log_prob,
                    'constant_count': constant_count,
                    'pruned_variant': False,
                    'n_variables': self.n_variables,
                    'n_restarts': self.n_restarts,
                    'method': self.refiner_method,
                    'p0_noise': self.refiner_p0_noise,
                    'p0_noise_kwargs': copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                    'converge_error': converge_error,
                    'numpy_errors': self.numpy_errors,
                    'y_variance': y_variance,
                    'length_penalty': self.length_penalty,
                    'constants_penalty': self.constants_penalty,
                    'likelihood_penalty': self.likelihood_penalty,
                    'complexity': complexity,
                    'metadata_snapshot': metadata_snapshot,
                }
                refinement_jobs.append(job)

            if refinement_jobs:
                input_dim = X_np.shape[1]
                self._input_dim = input_dim

                def _run_refinement_jobs(jobs: list[dict[str, Any]]) -> None:
                    if not jobs:
                        return

                    available_methods = mp.get_all_start_methods()
                    max_workers = min(self.refiner_workers, len(jobs))
                    use_parallel = max_workers > 1 and 'fork' in available_methods

                    if max_workers > 1 and not use_parallel:
                        warnings.warn("Parallel refinement requires the 'fork' start method; falling back to serial execution.")

                    if use_parallel:
                        ctx = mp.get_context('fork')
                        seed_sequence = np.random.SeedSequence()
                        spawned = seed_sequence.spawn(len(jobs))
                        for job, seq in zip(jobs, spawned):
                            job['seed'] = int(seq.generate_state(1, dtype=np.uint32)[0])

                        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                            futures = [executor.submit(_refine_candidate_worker, job) for job in jobs]
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
                            jobs,
                            total=len(jobs),
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

                with _RefinementContext(self.simplipy_engine, X_np, y_np):
                    _run_refinement_jobs(refinement_jobs)

                    prune_count_candidates = [r for r in self._results if np.isfinite(r.get('fvu', np.nan))]
                    top_k_resolved = self._resolve_prune_count(len(prune_count_candidates))

                    if top_k_resolved > 0:
                        sorted_results = sorted(prune_count_candidates, key=lambda r: r.get('fvu', np.inf))
                        top_results = sorted_results[:top_k_resolved]

                        seen_expressions = {tuple(r['expression']) for r in self._results}
                        variant_records: list[tuple[list[str], list[int], float, int]] = []

                        for base_result in top_results:
                            variants = self._generate_constant_pruning_variants(base_result['expression'])
                            for variant in variants:
                                variant_key = tuple(variant)
                                if variant_key in seen_expressions:
                                    continue

                                simplified_variant = self.simplipy_engine.simplify(list(variant), max_pattern_length=4)
                                if not self.simplipy_engine.is_valid(simplified_variant):
                                    continue

                                simplified_key = tuple(simplified_variant)
                                if simplified_key in seen_expressions:
                                    continue

                                seen_expressions.add(simplified_key)

                                try:
                                    encoded_variant = self.tokenizer.encode(simplified_variant, return_tensors=False)
                                except KeyError:
                                    continue
                                inherited_log_prob = float(base_result.get('log_prob', float('nan')))
                                constant_count = self._count_constants(simplified_variant)
                                variant_records.append((simplified_variant, encoded_variant, inherited_log_prob, constant_count))

                        pruning_jobs: list[dict[str, Any]] = []
                        if variant_records:
                            scored_log_probs = self._score_log_probs_batch(
                                sequences=[rec[1] for rec in variant_records],
                                prompt_prefix=prompt_prefix,
                                memory=memory_for_scoring,
                                device=data_tensor.device,
                            )

                            for (variant_expr, encoded_variant, inherited_lp, variant_constant_count), new_lp in zip(variant_records, scored_log_probs):
                                log_prob_value = float(new_lp) if np.isfinite(new_lp) else inherited_lp
                                pruning_job: dict[str, Any] = {
                                    'raw_beam': encoded_variant,
                                    'raw_beam_decoded': variant_expr,
                                    'beam': encoded_variant,
                                    'expression': variant_expr,
                                    'log_prob': log_prob_value,
                                    'constant_count': variant_constant_count,
                                    'pruned_variant': True,
                                    'n_variables': self.n_variables,
                                    'n_restarts': self.n_restarts,
                                    'method': self.refiner_method,
                                    'p0_noise': self.refiner_p0_noise,
                                    'p0_noise_kwargs': copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                                    'converge_error': converge_error,
                                    'numpy_errors': self.numpy_errors,
                                    'y_variance': y_variance,
                                    'length_penalty': self.length_penalty,
                                    'constants_penalty': self.constants_penalty,
                                    'likelihood_penalty': self.likelihood_penalty,
                                    'complexity': complexity,
                                    'metadata_snapshot': metadata_snapshot,
                                }
                                pruning_jobs.append(pruning_job)

                        _run_refinement_jobs(pruning_jobs)

            self.compile_results()

            np.seterr(**numpy_errors_before)

    def compile_results(
            self,
            length_penalty: float | None = None,
            *,
            constants_penalty: float | None = None,
            likelihood_penalty: float | None = None) -> None:
        """Aggregate refiner outputs into a tidy `pandas.DataFrame`.

        Parameters
        ----------
        length_penalty : float, optional
            Length penalty applied during score recomputation. Defaults to the
            current ``length_penalty`` value on the model.
        constants_penalty : float, optional
            Constant-count penalty applied during score recomputation. Defaults
            to the current ``constants_penalty`` value on the model.
        likelihood_penalty : float, optional
            Negative log-likelihood penalty applied during score recomputation.
            Defaults to the current ``likelihood_penalty`` value on the model.

        Raises
        ------
        ConvergenceError
            If no beams converged during refinement.
        """
        if not self._results:
            raise ConvergenceError("The optimization did not converge for any beam")

        self.initial_length_penalty = getattr(self, 'length_penalty', 0.0)
        self.initial_constants_penalty = getattr(self, 'constants_penalty', 0.0)
        self.initial_likelihood_penalty = getattr(self, 'likelihood_penalty', 0.0)

        if length_penalty is not None:
            self.length_penalty = float(length_penalty)
        if constants_penalty is not None:
            self.constants_penalty = float(constants_penalty)
        if likelihood_penalty is not None:
            self.likelihood_penalty = float(likelihood_penalty)

        # Compute the new score for each result
        for result in self._results:
            if 'score' in result:
                fvu = result.get('fvu', np.nan)
                log_prob = result.get('log_prob')
                constant_count = int(result.get('constant_count', self._count_constants(result.get('expression', []))))
                if np.isfinite(fvu):
                    result['score'] = self._score_from_fvu(
                        float(fvu),
                        len(result['expression']),
                        constant_count,
                        log_prob,
                        self.length_penalty,
                        self.constants_penalty,
                        self.likelihood_penalty,
                    )
                else:
                    result['score'] = np.nan

        # Sort the results by the best loss of each beam
        self._results = list(sorted(self._results, key=lambda x: (
            x['score'] if not np.isnan(x['score']) else float('inf'),
            np.isnan(x['score'])
        )))

        # Attach only the best fit (lowest loss) per beam for readability
        best_fit_payloads: list[tuple[np.ndarray, np.ndarray | None, float]] = []
        for result in self._results:
            fits_list = result.get('fits', [])
            if fits_list:
                best_fit = cast(tuple[np.ndarray, np.ndarray | None, float], min(fits_list, key=lambda tpl: tpl[2]))
            else:
                best_fit = cast(tuple[np.ndarray, np.ndarray | None, float], (np.array([]), None, float('nan')))
            best_fit_payloads.append(best_fit)

        self.results = pd.DataFrame(self._results)
        if not self.results.empty:
            self.results['beam_id'] = self.results.index
            self.results['fit_constants'] = [bf[0] for bf in best_fit_payloads]
            self.results['fit_covariances'] = [bf[1] for bf in best_fit_payloads]
            self.results['fit_loss'] = [bf[2] for bf in best_fit_payloads]
            self.results.drop(columns=['fits'], inplace=True, errors='ignore')

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
            "length_penalty": self.length_penalty,
            "constants_penalty": self.constants_penalty,
            "likelihood_penalty": self.likelihood_penalty,
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

        length_penalty = float(metadata.get("length_penalty", getattr(self, "length_penalty", 0.0)))
        constants_penalty = float(metadata.get("constants_penalty", getattr(self, "constants_penalty", 0.0)))
        likelihood_penalty = float(metadata.get("likelihood_penalty", getattr(self, "likelihood_penalty", 0.0)))
        n_variables = int(metadata.get("n_variables", self.n_variables))
        input_dim = int(metadata.get("input_dim", n_variables))

        self._input_dim = input_dim
        self.length_penalty = length_penalty
        self.constants_penalty = constants_penalty
        self.likelihood_penalty = likelihood_penalty
        self.variable_mapping = metadata.get("variable_mapping", self.variable_mapping)

        restored = deserialize_results_payload(
            payload,
            simplipy_engine=self.simplipy_engine,
            n_variables=n_variables,
            input_dim=input_dim,
            rebuild_refiners=rebuild_refiners,
        )

        self._results = restored
        self.compile_results(
            length_penalty=length_penalty,
            constants_penalty=constants_penalty,
            likelihood_penalty=likelihood_penalty,
        )

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
