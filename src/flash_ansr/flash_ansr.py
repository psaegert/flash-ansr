import os
import copy
import math
import time
import hashlib
import numbers
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Any, Iterable, Iterator, TypedDict, Callable, Tuple, Sequence, TypeVar, cast
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.base import BaseEstimator

from simplipy import SimpliPyEngine

from flash_ansr._refine_pool import RecoverableForkPool
from flash_ansr.generation import run_beam_search, run_softmax_sampling, run_mcts_generation
from flash_ansr.model import FlashANSRModel, Tokenizer
from flash_ansr.decoding.mcts import MCTSConfig
from flash_ansr.preprocessing import PromptPrefix, prepare_prompt_prefix
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.scoring import compute_fvu, count_constants, is_constant_token, normalize_variance, score_from_fvu
from flash_ansr.model.flash_ansr_model import _VRAM_GUARD_FRACTION
from flash_ansr.utils.generation import GenerationConfig, SoftmaxSamplingConfig, suggest_batch_size, suggest_batch_size_dims, _FULL_CAP_MIN_VRAM_GB, _spill_over_budget
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


def _candidate_refine_seed(refine_seed: int | None, tokens: Sequence[Any]) -> int:
    """Per-candidate p0-noise seed derived INTRINSICALLY from the candidate's tokens.

    Deriving each seed from a stable hash of the candidate (not its position in the job list)
    makes refinement reproducible AND robust to changes in the candidate SET: if generation
    drops/adds one candidate (CUDA ``multinomial`` is not bit-reproducible run-to-run), a
    positional ``SeedSequence().spawn(len(jobs))`` would shift every downstream seed; intrinsic
    seeds do not. This is also what lets a re-scheduled (overlapped) refinement reproduce the
    serial result candidate-for-candidate. ``refine_seed=None`` -> fresh OS entropy (legacy).
    """
    if refine_seed is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    digest = hashlib.blake2b(repr([str(t) for t in tokens]).encode("utf-8"), digest_size=8).digest()
    token_hash = int.from_bytes(digest, "little")
    return int(np.random.SeedSequence([int(refine_seed), token_hash]).generate_state(1, dtype=np.uint32)[0])


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


# --- parallel post-generation simplify (the c=32k generation lever; ~11x on the simplify portion) ---
_SIMPLIFY_PARALLEL_THRESHOLD = 4096   # only fork for the simplify when choices >= this (else serial)
_SIMPLIFY_ENGINE: Any = None          # per-worker engine, set by the pool initializer (inherited via fork)
_WORKER_THREAD_LIMITER: Any = None    # persistent-pool worker: kept-alive threadpoolctl limiter (single-thread BLAS)


def _simplify_pool_init(engine: Any) -> None:
    global _SIMPLIFY_ENGINE
    _SIMPLIFY_ENGINE = engine


def _persistent_pool_init(engine: Any) -> None:
    """Initializer for the persistent pre-CUDA pool: seed BOTH worker-global engines once.

    The persistent pool serves refine (``_refine_candidate_worker`` -> ``_GLOBAL_SIMPLIPY_ENGINE``)
    AND simplify (``_simplify_pool_worker`` -> ``_SIMPLIFY_ENGINE``). The engine is
    problem-independent, so it is set ONCE per worker here; per-problem ``X``/``y`` travel in each
    refine job payload instead, so two concurrently-scheduled refinements (the Step-4 overlap engine)
    never race a shared ``_GLOBAL_REFINEMENT_DATA`` module global.

    Also pins this worker to single-thread BLAS for its lifetime: 8 refine workers each spinning a
    multi-thread LAPACK pool would oversubscribe cores (the one-sklearn-job-at-a-time landmine) and is
    deployment-matched (the cluster mandates ``OMP/MKL/OPENBLAS=1``). The limiter object is kept alive
    in a module global so the limit persists past this function.
    """
    global _GLOBAL_SIMPLIPY_ENGINE, _SIMPLIFY_ENGINE, _WORKER_THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits
        _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:
        pass
    _GLOBAL_SIMPLIPY_ENGINE = engine
    _SIMPLIFY_ENGINE = engine


def _simplify_pool_worker(raw_expr: tuple) -> tuple:
    """Simplify one raw expression (pure CPU; deterministic -> byte-identical to serial)."""
    return tuple(_SIMPLIFY_ENGINE.simplify(list(raw_expr), max_pattern_length=4))


@dataclass
class GenState:
    """Self-contained output of the GPU generation phase of ``fit`` (``_fit_generate``).

    Carries everything the CPU refinement phase needs, so the two phases share NO instance state.
    This is what lets the overlapped engine run generation for problem N+1 while problem N refines.
    """
    raw_beams: list
    log_probs: list
    X_np: np.ndarray
    y_np: np.ndarray
    y_variance: float
    prompt_prefix: Any
    metadata_snapshot: Any
    memory_for_scoring: Any
    device: Any
    variable_mapping: dict
    complexity: Any
    generation_time: float


@dataclass
class FitResult:
    """Self-contained output of the CPU refinement phase (``_fit_refine``).

    Everything ``fit`` commits to instance state via ``_apply_fit_result``; nothing is written to
    ``self`` during refinement, so a re-scheduled (overlapped) refinement cannot clobber ``self``.
    """
    results: list
    results_df: "pd.DataFrame"
    refinement_time: float
    generation_time: float
    variable_mapping: dict
    input_dim: int | None
    prompt_metadata: Any


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
    refiner_p0_noise : {'uniform', 'normal', 'cauchy', 'magspan'}, optional
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
        return normalize_variance(variance)

    @classmethod
    def _compute_fvu(cls, loss: float, sample_count: int, variance: float) -> float:
        return compute_fvu(loss, sample_count, variance)

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
        return score_from_fvu(
            fvu, complexity, constant_count, log_prob,
            length_penalty, constants_penalty, likelihood_penalty)

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
        prefix_len = len(base_tokens)
        try:
            eos_id = int(self.tokenizer['<eos>'])
        except KeyError:
            eos_id = None
        pad_id = int(self.tokenizer['<pad>']) if '<pad>' in self.tokenizer else 0

        # Score the SAME token span the generation path scores, so a rescored (pruned) candidate's
        # log_prob is comparable to a generated candidate's. A training/generation beam is
        # [<bos>, <expression>, body, </expression>, <eos>]; ``sequences`` here are body-only
        # (tokenizer.encode of the simplified variant, no markers). We therefore (a) re-insert the
        # closing </expression> before <eos> when the prefix opened <expression> -- omitting it left
        # <eos> conditioned on the last body token, an off-distribution context that made the rescored
        # log_prob spuriously (and systematically) too negative -- and (b) sum log-probs only over the
        # generated positions (everything after the prefix), exactly as sample_top_kp does.
        end_expr_id = None
        if '<expression>' in self.tokenizer and '</expression>' in self.tokenizer:
            if int(self.tokenizer['<expression>']) in base_tokens:
                end_expr_id = int(self.tokenizer['</expression>'])

        full_sequences: list[list[int]] = []
        lengths: list[int] = []
        for seq in sequences:
            extended = list(base_tokens) + list(seq)
            if end_expr_id is not None:
                extended = extended + [end_expr_id]
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

        # Sum from the first generated position (mirrors generation, which never scores the prompt prefix).
        start = max(prefix_len, 1)
        gathered: list[float] = []
        for i, seq_len in enumerate(lengths):
            if seq_len <= start:
                gathered.append(float('-inf'))
                continue

            targets = input_tokens[i, start:seq_len]
            step_log_probs = log_probs[i, start - 1:seq_len - 1]
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
        return is_constant_token(token)

    @classmethod
    def _count_constants(cls, expression: Sequence[str]) -> int:
        return count_constants(expression)

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
            refiner_p0_noise: Literal['uniform', 'normal', 'cauchy', 'magspan'] | None = 'normal',
            refiner_p0_noise_kwargs: dict | None | Literal['default'] = 'default',
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            length_penalty: float = 0.05,
            constants_penalty: float = 0.0,
            likelihood_penalty: float = 0.0,
            refiner_workers: int | None = None,
            prune_constant_budget: float | int = 0):
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
        # Parallelize the post-generation simplify across `refiner_workers` (gated to choices >=
        # _SIMPLIFY_PARALLEL_THRESHOLD; byte-identical to serial). Set False to force serial.
        self.parallel_simplify = True

        self._results: list[Result] = []
        self.results: pd.DataFrame = pd.DataFrame()
        self._mcts_cache: dict[Tuple[int, ...], dict[str, Any]] = {}

        self._input_dim: int | None = None

        self.variable_mapping: dict[str, str] = {}
        self._prompt_prefix: PromptPrefix | None = None
        self._prompt_metadata: dict[str, list[list[str]]] | None = None

        # Optional persistent pre-CUDA fork pool (inference-speed Step 3). When set (via
        # ``load(persistent_refine_pool=True)``) refinement + simplify route onto it instead of
        # forking a fresh pool per call; ``None`` keeps the legacy per-call-fork path (the default).
        self._refine_pool: RecoverableForkPool | None = None

        # Set True by ``OverlappedEvaluationEngine`` for the duration of an overlapped run (inference-
        # speed Step 4). While True a GPU-owner thread runs generation concurrently with refinement, so
        # ``_fit_refine`` MUST keep all per-candidate work in forked pool workers (whose global-RNG
        # reseed is process-isolated) and MUST NOT fork a fresh pool on the calling thread (that would
        # be a fork-after-CUDA-while-another-thread-is-live hazard). See ``_run_refinement_jobs``.
        self._overlap_mode: bool = False

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
            refiner_p0_noise: Literal['uniform', 'normal', 'cauchy', 'magspan'] | None = 'normal',
            refiner_p0_noise_kwargs: dict | None | Literal['default'] = 'default',
            numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
            length_penalty: float = 0.05,
            constants_penalty: float = 0.0,
            likelihood_penalty: float = 0.0,
            device: str = 'cpu',
            refiner_workers: int | None = None,
            prune_constant_budget: float | int = 0,
            persistent_refine_pool: bool = False) -> "FlashANSR":
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
        refiner_p0_noise : {'uniform', 'normal', 'cauchy', 'magspan'}, optional
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
        persistent_refine_pool : bool, optional
            When ``True`` a single persistent ``fork`` worker pool is forked BEFORE any CUDA init and
            reused across ``fit()`` calls for both refinement and the parallel post-generation
            simplify (instead of forking a fresh pool per call). For a CUDA ``device`` the weights are
            loaded on CPU first, the engine is warmed, the pool is forked, and only then is the model
            moved to the device -- the structural mitigation for the fork-after-CUDA deadlock family.
            Requires the ``fork`` start method and ``refiner_workers > 1``; otherwise it is a no-op and
            the legacy per-call-fork path is used. Opt-in; the default preserves today's behaviour
            byte-for-byte.

        Returns
        -------
        model : FlashANSR
            Fully initialized regressor ready for inference.
        """
        directory = substitute_root_path(directory)

        flash_ansr_model_path = os.path.join(directory, 'model.yaml')
        tokenizer_path = os.path.join(directory, 'tokenizer.yaml')

        # When a persistent pre-CUDA pool is requested for a non-CPU device, defer the device move:
        # load weights on CPU (incl. map_location, which also touches CUDA otherwise) so the pool can
        # be forked while CUDA is still uninitialized. The flag-off path is unchanged.
        defer_cuda = persistent_refine_pool and str(device) != 'cpu'
        load_device = 'cpu' if defer_cuda else device

        model = FlashANSRModel.from_config(flash_ansr_model_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True, map_location=load_device))
        model.eval().to(load_device)

        tokenizer = Tokenizer.from_config(tokenizer_path)

        nsr = cls(
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

        if persistent_refine_pool:
            # Warm the engine + fork the pool (pre-CUDA), then move the model to the target device.
            nsr._enable_persistent_refine_pool(target_device=device)

        return nsr

    def _enable_persistent_refine_pool(self, target_device: str) -> None:
        """Warm the engine, fork the persistent pool (PRE-CUDA), then move the model to the device.

        This is the production form of the validated fork-safety sequence:
        ``load(device='cpu')`` -> warm ``simplipy.engine`` in the parent -> fork the pool (workers
        COW-inherit the warm engine, no CUDA initialized yet) -> ``.to(device)``. With
        ``refiner_workers <= 1`` or no ``fork`` start method there is nothing to parallelize, so the
        pool is skipped and the legacy per-call path is used (the model is still moved to the device).
        """
        available_methods = mp.get_all_start_methods()
        if self.refiner_workers <= 1 or 'fork' not in available_methods:
            if self.refiner_workers > 1 and 'fork' not in available_methods:
                warnings.warn("persistent_refine_pool requires the 'fork' start method; using per-call refinement.")
            self.to(target_device)
            return

        # Warm + fork under a single-thread BLAS limit. The warm refinement runs scipy curve_fit
        # (LAPACK SVD), which can spin a multi-thread BLAS pool in the PARENT; forking across a live
        # BLAS thread pool is the fork-after-THREADS deadlock family. Limiting to 1 thread keeps the
        # parent single-threaded across the warm AND the fork (the workers are independently pinned in
        # _persistent_pool_init). threadpoolctl is a scikit-learn dependency; degrade gracefully if
        # absent (the deployment also pins OMP/MKL/OPENBLAS=1 in the environment).
        try:
            from threadpoolctl import threadpool_limits
            _blas_limit: Any = threadpool_limits(limits=1)
        except Exception:
            _blas_limit = None
            warnings.warn("threadpoolctl unavailable; cannot pin single-thread BLAS for the pre-CUDA fork. "
                          "Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 in the environment.")
        try:
            # 1. Warm the engine in the PARENT so the COW-forked workers inherit a ready engine.
            self._warm_refine_engine()

            # 2. Fork the pool NOW -- before any CUDA op. The initializer seeds the worker engine
            #    globals once (problem-independent); per-problem X/y travel in each refine job payload.
            self._refine_pool = RecoverableForkPool(
                self.refiner_workers,
                initializer=_persistent_pool_init,
                initargs=(self.simplipy_engine,),
            )
        finally:
            if _blas_limit is not None:
                _blas_limit.restore_original_limits()

        # 3. Only now move the model to the target device (this may initialize CUDA, AFTER the fork).
        self.to(target_device)

    def _warm_refine_engine(self) -> None:
        """Warm the simplipy engine IN-PARENT before forking the persistent pool (fork-safety #1).

        Ensures the engine's lambda-eval module globals are populated (``import_modules``) AND that one
        real refine lambda has been built (``code_to_lambda``) in the parent, so the COW-forked workers
        inherit a ready engine and never ``NameError`` on first refine. Best-effort and side-effect
        free: the warm fit need not converge, the parent RNG state is saved/restored, and the throwaway
        job carries ``seed=None`` so ``_refine_candidate_worker`` does not reseed the parent.
        """
        try:
            self.simplipy_engine.import_modules()
        except Exception:
            pass

        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            n_vars = max(1, self.n_variables)
            X = np.ones((8, n_vars), dtype=np.float64)
            y = np.ones((8, 1), dtype=np.float64)
            warm_expression = ['*', '<constant>', 'x1']
            warm_job: dict[str, Any] = {
                'raw_beam': [],
                'raw_beam_decoded': list(warm_expression),
                'beam': [],
                'expression': list(warm_expression),
                'log_prob': 0.0,
                'constant_count': 1,
                'pruned_variant': False,
                'n_variables': self.n_variables,
                'n_restarts': 1,
                'method': self.refiner_method,
                'p0_noise': self.refiner_p0_noise,
                'p0_noise_kwargs': copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                'converge_error': 'ignore',
                'numpy_errors': self.numpy_errors,
                'y_variance': 1.0,
                'length_penalty': self.length_penalty,
                'constants_penalty': self.constants_penalty,
                'likelihood_penalty': self.likelihood_penalty,
                'complexity': None,
                'metadata_snapshot': None,
                'seed': None,
                'X': X,
                'y': y,
                'simplipy_engine': self.simplipy_engine,
            }
            _refine_candidate_worker(warm_job)
        except Exception:
            pass
        finally:
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

    def close(self) -> None:
        """Release the persistent refine pool, if any. Idempotent; safe to call multiple times."""
        pool = getattr(self, '_refine_pool', None)
        if pool is not None:
            pool.shutdown()
            self._refine_pool = None

    def __del__(self) -> None:  # best-effort cleanup; callers should prefer close()
        try:
            self.close()
        except Exception:
            pass

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

    def _parallel_build_simplify_map(self, unique_raw: set, n_workers: int) -> dict:
        """Simplify every unique raw expression in ONE parallel fork pass -> {raw_tuple: simplified}.

        Forked HERE (scoped to this generate() call) and shut down before fit()'s refinement fork, so
        the two fork pools never coexist (avoids the fork-after-CUDA / nested-fork deadlock family).
        Byte-identical to serial simplify: engine.simplify is a deterministic pure function, so the
        map values equal the inline values; _postprocess_sampled then looks them up in beam order.
        """
        raw_list = list(unique_raw)
        if not raw_list:
            return {}
        chunksize = max(1, len(raw_list) // (n_workers * 8))
        if self._refine_pool is not None:
            # Route onto the persistent pre-CUDA pool (engine already in the worker globals via the
            # pool initializer) -> no fork ever happens after CUDA. Byte-identical to serial simplify.
            # recover=False (this runs post-CUDA inside generate): on a worker death dispose the pool
            # and fall back to the legacy per-call fork below, never re-forking the pool post-CUDA.
            try:
                simplified = self._refine_pool.map_ordered(_simplify_pool_worker, raw_list, chunksize=chunksize, recover=False)
                return dict(zip(raw_list, simplified))
            except BrokenProcessPool:
                if getattr(self, '_overlap_mode', False):
                    # Step 4b: under the overlap engine a GPU producer thread is live and the consumer
                    # is using THIS SAME pool for refine. Do NOT close it (that would break the
                    # consumer mid-run) or fork a per-call pool (fork-after-CUDA on the live GPU
                    # thread is the deadlock hazard). Re-raise so the producer aborts the overlap run
                    # cleanly (engine checkpoints + surfaces an actionable error).
                    raise
                warnings.warn("Persistent refine pool broke (worker death) during simplify; disabling it "
                              "and falling back to a per-call fork pool for the rest of this run.")
                self.close()
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=_simplify_pool_init,
                                 initargs=(self.simplipy_engine,)) as executor:
            simplified = list(executor.map(_simplify_pool_worker, raw_list, chunksize=chunksize))
        return dict(zip(raw_list, simplified))

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
        memory: torch.Tensor | None = None,
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
        memory : torch.Tensor, optional
            Pre-computed encoder memory for ``data``. When provided, sampling
            skips the encoder forward inside the model; required for the outer
            batching loop to reuse one encoder pass across chunks.

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
                choices_target = int(generation_kwargs.get('choices', 1))
                generation_kwargs = dict(generation_kwargs)  # local copy; resolve sentinels into it

                # --- Resolve static-decode (multi-chunk regime) + the c-adaptive batch TOGETHER ---
                # The static arm's chunk batch is BOTH the regime probe (static iff batch < choices) AND the
                # batch the static decode uses -> compute it ONCE here, thread it into the resolver, and reuse
                # it below (so the regime decision and the decode can never disagree on a free-VRAM race).
                # No per-(scale,c) table, no hardware gate: the regime is read from the card's own free VRAM,
                # which is hardware-general (cross-GPU validated 2026-06-24). Static is a GPU lever -> the
                # auto-static regime is only considered on CUDA; CPU/unknown -> dynamic.
                _static_kwarg = generation_kwargs.get('static_decode', None)
                _raw_bs = generation_kwargs.get('batch_size', choices_target)
                _dev = self.flash_ansr_model.device
                _is_cuda = getattr(_dev, 'type', None) == 'cuda'
                _explicit_bs = _raw_bs if isinstance(_raw_bs, int) else None

                _static_batch = None
                if _is_cuda:
                    if _explicit_bs is not None:
                        _static_batch = _explicit_bs
                    else:
                        # Architecture/VRAM-driven cap. "Available" = physical FREE + this process's REUSABLE
                        # held pool (reserved - allocated): the caching allocator reuses freed-but-held blocks
                        # WITHOUT growing physical, so sizing by raw free alone would shrink the cap every call
                        # as the pool grows (WSL2 never releases it) even though the decode fits.
                        _free = None
                        try:
                            _free = torch.cuda.mem_get_info(_dev)[0]
                            _free += torch.cuda.memory_reserved(_dev) - torch.cuda.memory_allocated(_dev)
                        except Exception:
                            _free = None
                        _d = self.flash_ansr_model.decoder
                        _static_batch = suggest_batch_size_dims(
                            choices_target,
                            n_layers=len(_d.layers),
                            n_heads=_d.layers[0].self_attention.n_heads,
                            head_dim=_d.layers[0].self_attention.head_dim,
                            max_len=self.flash_ansr_model.decoder_max_seq_len,
                            static_decode=True, free_bytes=_free)

                _static = self.flash_ansr_model._resolve_static_decode(
                    _static_kwarg, choices=choices_target, static_batch=_static_batch)
                generation_kwargs['static_decode'] = _static

                # Resolve the ACTUAL chunk batch. Static reuses _static_batch (the exact value the regime used);
                # the dynamic 'auto' path keeps the measured per-model lookup (a dims estimate would REGRESS
                # its validated cap). An explicit int batch_size bypasses this entirely.
                if isinstance(_raw_bs, str) and _raw_bs == 'auto':
                    if _static and _static_batch is not None:
                        _raw_bs = _static_batch
                    else:
                        if getattr(self, '_n_params', None) is None:
                            self._n_params = sum(p.numel() for p in self.flash_ansr_model.parameters())
                        _vram_gb = 24.0
                        if _is_cuda:
                            try:
                                _vram_gb = torch.cuda.get_device_properties(_dev).total_memory / 1e9
                            except Exception:
                                pass
                        _raw_bs = suggest_batch_size(choices_target, self._n_params, _vram_gb)
                    if verbose:
                        print(f"[auto batch] choices={choices_target} static={_static} -> batch_size={_raw_bs}")
                # Clamp to >= 1 so a misconfigured batch_size <= 0 cannot stall the batched loop below
                # (this_chunk would be 0 and ``drawn`` never advance). Write the RESOLVED int back so no
                # downstream path (esp. the single-shot path, which forwards generation_kwargs verbatim)
                # sees the 'auto' string -> sample_top_kp range(0, n, 'auto') TypeError.
                batch_size = max(1, int(_raw_bs))
                generation_kwargs['batch_size'] = batch_size

                # Legacy single-shot path: no outer loop, no extra dedup.
                if batch_size >= choices_target or choices_target <= 0:
                    beams, log_probs, completed, rewards = run_softmax_sampling(
                        self.flash_ansr_model,
                        data=data,
                        verbose=verbose,
                        prompt_prefix=effective_prompt,
                        generation_kwargs=generation_kwargs,
                        memory=memory,
                    )
                    return beams, log_probs, completed, rewards

                # Batched path: draw ``choices_target`` candidates in ``batch_size`` chunks,
                # freeing each chunk's KV cache before the next, and dedupe across chunks.
                # Behaviour matches the legacy single-shot dedup (draw N independent samples,
                # dedupe -> <= N unique), so candidate quality is statistically equivalent with
                # bounded peak memory. A sample-until-N-unique variant is intentionally deferred
                # (it would need a max-attempts cap to bound compute on low-entropy inputs).
                unique = bool(generation_kwargs.get('unique', True))
                if memory is None:
                    memory = self.flash_ansr_model._create_memory(data)

                # --- Dynamic-path VRAM spill-guard (sub-24 GiB cards ONLY) ----------------------------------
                # The measured auto-caps and the validated >= 24 GiB path stay UNCHANGED (no guard there ->
                # no false-fire on a 4090/A100). On a smaller, UNTESTED card the conservative cap or an
                # explicit oversized batch_size can spill to system RAM (WSL2, ~28x slower) or OOM; this
                # backstop measures the FIRST chunk's ACTUAL peak allocation (no overhead calibration) and
                # fails loud rather than crawl through the remaining chunks. NOT exercised on the dev 4090
                # (gated off at >= 24 GiB); worst case it tells a small-card user to pass an explicit smaller
                # batch_size (the documented fallback) -- it can never regress a >= 24 GiB run. Gated on
                # `not _static`: the static decode path has its OWN spill-guard (generate_static), so this
                # would only double-cover it -- keep this guard purely for the dynamic decode path.
                _sg_dev = self.flash_ansr_model.device
                _sg_state: tuple[float, int] | None = None
                if (not _static) and getattr(_sg_dev, 'type', None) == 'cuda':
                    try:
                        if torch.cuda.get_device_properties(_sg_dev).total_memory / 1e9 < _FULL_CAP_MIN_VRAM_GB:
                            # Available = physical FREE + this process's REUSABLE held pool (reserved - alloc),
                            # matching the static guard; reset the allocated peak so it reflects THIS decode.
                            _sg_alloc0 = torch.cuda.memory_allocated(_sg_dev)
                            _sg_avail = torch.cuda.mem_get_info(_sg_dev)[0] + (torch.cuda.memory_reserved(_sg_dev) - _sg_alloc0)
                            torch.cuda.reset_peak_memory_stats(_sg_dev)
                            _sg_state = (float(_sg_avail), int(_sg_alloc0))
                    except Exception:
                        _sg_state = None

                def _spill_guard_check(_bsz: int) -> None:
                    # Call once, right after the first chunk's decode. Raises if its added working set (peak
                    # ALLOCATED delta over the decode-start baseline) exceeds _VRAM_GUARD_FRACTION of available
                    # VRAM. Allocated-based (not the process-global reserved high-water a prior wide run inflates).
                    if _sg_state is None:
                        return
                    _added = torch.cuda.max_memory_allocated(_sg_dev) - _sg_state[1]
                    if _spill_over_budget(_added, _sg_state[0], _VRAM_GUARD_FRACTION):
                        raise RuntimeError(
                            f"dynamic decode batch={_bsz} added {_added / 1024**3:.1f} GB > "
                            f"{_VRAM_GUARD_FRACTION:.0%} of {_sg_state[0] / 1024**3:.1f} GB available VRAM on a "
                            f"sub-24GB card; pass a smaller explicit batch_size (auto-batch is tuned for >= 24 GiB cards).")

                # --- PARALLEL post-gen simplify (gated): the deployed simplify is already a post-pass
                # inside sample_top_kp; here we parallelize it. Generate raw per chunk, simplify ALL
                # candidates in ONE fork pass (forked here, shut down before fit()'s refinement fork),
                # then post-process per chunk with the precomputed map. Byte-identical to the serial
                # path (same engine.simplify values, same dedup/sort order); ~11x on simplify at high c.
                _simplify_setting = generation_kwargs.get('simplify', True)
                _valid_only_setting = bool(generation_kwargs.get('valid_only', True))
                _n_workers = min(16, max(1, int(getattr(self, 'refiner_workers', 1) or 1)))
                _do_parallel = (
                    getattr(self, 'parallel_simplify', True)
                    and _simplify_setting is True
                    and choices_target >= _SIMPLIFY_PARALLEL_THRESHOLD
                    and _n_workers > 1
                    and 'fork' in mp.get_all_start_methods()
                )
                if _do_parallel:
                    chunks_raw: list[tuple[list[list[int]], list[float]]] = []
                    drawn = 0
                    while drawn < choices_target:
                        this_chunk = min(batch_size, choices_target - drawn)
                        ck = dict(generation_kwargs)
                        ck['choices'] = this_chunk
                        ck['batch_size'] = this_chunk
                        ck.pop('return_raw', None)
                        raw_seqs, raw_scores = self.flash_ansr_model.sample_top_kp(
                            data=data, verbose=verbose, prompt_prefix=effective_prompt,
                            memory=memory, return_raw=True, **ck)
                        chunks_raw.append((raw_seqs, raw_scores))
                        if drawn == 0:
                            _spill_guard_check(this_chunk)
                        drawn += this_chunk

                    unique_raw: set[tuple] = set()
                    for _seqs, _ in chunks_raw:
                        for _e in self.flash_ansr_model.extract_valid_raw_expressions(_seqs):
                            unique_raw.add(tuple(_e))
                    simplify_map = (self._parallel_build_simplify_map(unique_raw, _n_workers)
                                    if unique_raw else {})

                    par_seen: set[tuple[int, ...]] = set()
                    par_beams: list[list[int]] = []
                    par_log_probs: list[float] = []
                    par_completed: list[bool] = []
                    for raw_seqs, raw_scores in chunks_raw:
                        beams_c, log_probs_c, completed_c = self.flash_ansr_model._postprocess_sampled(
                            raw_seqs, raw_scores, simplify=_simplify_setting, unique=unique,
                            valid_only=_valid_only_setting, verbose=verbose, simplify_map=simplify_map)
                        for rb, lp, cp in zip(beams_c, log_probs_c, completed_c):
                            if unique:
                                try:
                                    enc_expr, _, _ = self.flash_ansr_model.tokenizer.extract_expression_from_beam(rb)
                                except (ValueError, IndexError):
                                    enc_expr = None
                                key = tuple(enc_expr) if enc_expr else tuple(rb)
                                if key in par_seen:
                                    continue
                                par_seen.add(key)
                            par_beams.append(rb)
                            par_log_probs.append(lp)
                            par_completed.append(cp)
                    rewards = [float('nan')] * len(par_beams)
                    return par_beams, par_log_probs, par_completed, rewards

                seen: set[tuple[int, ...]] = set()
                acc_beams: list[list[int]] = []
                acc_log_probs: list[float] = []
                acc_completed: list[bool] = []
                drawn = 0
                while drawn < choices_target:
                    this_chunk = min(batch_size, choices_target - drawn)
                    chunk_kwargs = dict(generation_kwargs)
                    chunk_kwargs['choices'] = this_chunk
                    chunk_kwargs['batch_size'] = this_chunk
                    beams_c, log_probs_c, completed_c, _ = run_softmax_sampling(
                        self.flash_ansr_model,
                        data=data,
                        verbose=verbose,
                        prompt_prefix=effective_prompt,
                        generation_kwargs=chunk_kwargs,
                        memory=memory,
                    )
                    if drawn == 0:
                        _spill_guard_check(this_chunk)
                    for rb, lp, cp in zip(beams_c, log_probs_c, completed_c):
                        if unique:
                            # Dedupe on the extracted <expression>, not the raw beam: the
                            # legacy single-shot sample_top_kp keys its `seen` set on the
                            # parsed/simplified expression, and each chunk's returned beam is
                            # reconstructed from it — so extracting it here removes cross-chunk
                            # duplicates exactly as one big run would. Fall back to the raw beam
                            # for unparseable sequences (only reachable when valid_only=False).
                            try:
                                enc_expr, _, _ = self.flash_ansr_model.tokenizer.extract_expression_from_beam(rb)
                            except (ValueError, IndexError):
                                enc_expr = None
                            key = tuple(enc_expr) if enc_expr else tuple(rb)
                            if key in seen:
                                continue
                            seen.add(key)
                        acc_beams.append(rb)
                        acc_log_probs.append(lp)
                        acc_completed.append(cp)
                    drawn += this_chunk

                rewards = [float('nan')] * len(acc_beams)
                return acc_beams, acc_log_probs, acc_completed, rewards
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
            exclude_terms: Iterable[Sequence[Any]] | None = None,
            refine_seed: int | None = None) -> None:
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
        refine_seed : int or None, optional
            Keyword-only seed for the constant-refinement ``p0`` noise. When
            provided, the per-candidate refiner seeds are derived deterministically
            from it (via ``np.random.SeedSequence(refine_seed)``), so refinement is
            reproducible and independent of completion order. When ``None`` (the
            default) fresh OS entropy is used, preserving the legacy behaviour.

        Raises
        ------
        ValueError
            If ``y`` has more than one output dimension or cannot be reshaped.
        """
        # TODO: Support lists
        # TODO: Support 0-d and 1-d tensors

        # Reset per-fit instance state up front so a ConvergenceError mid-fit leaves a clean (not
        # stale) view for library callers; the eval adapter ignores self on the error path.
        self._results = []
        self.variable_mapping = {}
        self._generation_time = 0.0
        self._refinement_time = 0.0

        # Adopt the configured floating-point error policy for refinement; try/finally restores it
        # even when refinement raises ConvergenceError (which must still propagate to the caller).
        numpy_errors_before = np.geterr()
        np.seterr(all=self.numpy_errors)
        try:
            gen_state = self._fit_generate(
                X, y, variable_names,
                complexity=complexity,
                allowed_terms=allowed_terms,
                include_terms=include_terms,
                exclude_terms=exclude_terms,
                verbose=verbose,
            )
            # Generation-phase state, applied so callers see it even if refinement raises.
            self.variable_mapping = gen_state.variable_mapping
            self._generation_time = gen_state.generation_time
            self._prompt_metadata = copy.deepcopy(gen_state.metadata_snapshot) if gen_state.metadata_snapshot is not None else None

            fit_result = self._fit_refine(
                gen_state,
                converge_error=converge_error,
                refine_seed=refine_seed,
                verbose=verbose,
            )
            self._apply_fit_result(fit_result)
        finally:
            np.seterr(**numpy_errors_before)

    def _fit_generate(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto',
            *,
            complexity: int | float | None = None,
            allowed_terms: Iterable[Sequence[Any]] | None = None,
            include_terms: Iterable[Sequence[Any]] | None = None,
            exclude_terms: Iterable[Sequence[Any]] | None = None,
            verbose: bool = False) -> "GenState":
        """GPU generation phase of :meth:`fit`: prepare inputs, sample candidates, return a GenState.

        Writes NOTHING to ``self`` (the serial ``fit`` and the overlapped engine apply the returned
        state), so generation for problem N+1 cannot clobber problem N's refinement. (NB
        ``_prepare_prompt_prefix`` still sets ``self._prompt_prefix``; refinement reads the prefix
        from the returned GenState, not from ``self``, so that write is benign.)
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif y.shape[-1] != 1:
            raise ValueError("The target data must have a single output dimension")

        X = self._truncate_input(X)

        # Default: No mapping
        variable_mapping: dict[str, str] = {}

        if isinstance(variable_names, list):
            # column i -> variable_names[i]
            variable_mapping = {f"x{i + 1}": name for i, name in enumerate(variable_names)}

        elif isinstance(variable_names, dict):
            if isinstance(X, pd.DataFrame):
                # column i -> variable_names[column i]
                variable_mapping = {f"x{i + 1}": variable_names[c] for i, c in enumerate(X.columns)}
            else:
                # custom mapping
                variable_mapping = variable_names

        elif variable_names == 'auto':
            if isinstance(X, pd.DataFrame):
                # column i -> column name
                variable_mapping = {f"x{i + 1}": name for i, name in enumerate(X.columns)}

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

            memory_for_scoring = self.flash_ansr_model._create_memory(data_tensor)

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

            _t_gen = time.time()
            raw_beams, log_probs, _completed_flags, _rewards = self.generate(
                data_tensor,
                prompt_prefix=prompt_prefix,
                complexity=complexity,
                verbose=verbose,
                memory=memory_for_scoring,
            )
            generation_time = time.time() - _t_gen

            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()

        return GenState(
            raw_beams=raw_beams,
            log_probs=log_probs,
            X_np=X_np,
            y_np=y_np,
            y_variance=y_variance,
            prompt_prefix=prompt_prefix,
            metadata_snapshot=metadata_snapshot,
            memory_for_scoring=memory_for_scoring,
            device=data_tensor.device,
            variable_mapping=variable_mapping,
            complexity=complexity,
            generation_time=generation_time,
        )

    def _fit_refine(
            self,
            gen_state: "GenState",
            *,
            converge_error: Literal['raise', 'ignore', 'print'] = 'ignore',
            refine_seed: int | None = None,
            verbose: bool = False) -> "FitResult":
        """CPU refinement phase of :meth:`fit`: build jobs, fit constants, prune, compile.

        Operates on LOCAL state and returns a FitResult; writes NOTHING to ``self``. With
        ``prune_constant_budget == 0`` it touches no GPU at all. NOTE: not yet overlap-SAFE on its
        own -- ``_RefinementContext`` sets a module global for fork-COW that two concurrent calls
        would race; that global is removed (per-job X/y on the persistent pool) in a later step.
        """
        gs = gen_state
        results: list[Result] = []
        refinement_time = 0.0
        input_dim = gs.X_np.shape[1]

        beams = [self.flash_ansr_model.tokenizer.extract_expression_from_beam(raw_beam)[0] for raw_beam in gs.raw_beams]

        raw_beams_decoded = [self.tokenizer.decode(raw_beam, special_tokens='<constant>') for raw_beam in gs.raw_beams]
        beams_decoded = [self.tokenizer.decode(beam, special_tokens='<constant>') for beam in beams]

        refinement_jobs: list[dict[str, Any]] = []
        beam_iterator = zip(gs.raw_beams, raw_beams_decoded, beams, beams_decoded, gs.log_probs)
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
                'y_variance': gs.y_variance,
                'length_penalty': self.length_penalty,
                'constants_penalty': self.constants_penalty,
                'likelihood_penalty': self.likelihood_penalty,
                'complexity': gs.complexity,
                'metadata_snapshot': gs.metadata_snapshot,
            }
            refinement_jobs.append(job)

        if refinement_jobs:
            def _run_refinement_jobs(jobs: list[dict[str, Any]]) -> None:
                if not jobs:
                    return

                available_methods = mp.get_all_start_methods()
                max_workers = min(self.refiner_workers, len(jobs))
                use_parallel = max_workers > 1 and 'fork' in available_methods

                if max_workers > 1 and not use_parallel:
                    warnings.warn("Parallel refinement requires the 'fork' start method; falling back to serial execution.")

                # Assign each job an INTRINSIC p0-noise seed (hashed from the candidate's own
                # raw_beam tokens, not its position) BEFORE the parallel/serial split, so both
                # code paths refine identically, the result is independent of completion order,
                # and a dropped/added candidate does not cascade-shift other seeds. The raw_beam
                # is the dedup-unique candidate identity (the decoded `expression` can collide).
                for job in jobs:
                    key_tokens = job.get('raw_beam') or job.get('expression', [])
                    job['seed'] = _candidate_refine_seed(refine_seed, key_tokens)

                ran_on_pool = False
                # In overlap mode route EVERY problem (including a single-candidate problem, for which
                # ``use_parallel`` is False because ``max_workers = min(workers, len(jobs)) == 1``) onto
                # the persistent pool: the per-candidate global-RNG reseed in ``_refine_candidate_worker``
                # then happens only in forked worker processes (isolated copies), never on the consumer
                # thread where it would race the GPU-owner thread's ``torch.multinomial`` draws.
                if self._refine_pool is not None and (use_parallel or self._overlap_mode):
                    # Persistent pre-CUDA pool: the engine is already in the worker globals (pool
                    # initializer); per-problem X/y travel in each job payload (no _RefinementContext
                    # fork-COW race). recover=False: this runs AFTER generation has initialized CUDA, so
                    # the pool must NOT re-fork on a worker death (that would reintroduce fork-after-CUDA).
                    for job in jobs:
                        job['X'] = gs.X_np
                        job['y'] = gs.y_np
                    chunksize = max(1, len(jobs) // (max(1, max_workers) * 8))
                    try:
                        outcomes = self._refine_pool.map_ordered(
                            _refine_candidate_worker,
                            jobs,
                            chunksize=chunksize,
                            recover=False,
                            wrap=(lambda it: _iterate_with_progress(it, total=len(jobs), verbose=verbose, desc="Fitting Constants")),
                        )
                    except BrokenProcessPool:
                        if self._overlap_mode:
                            # A GPU-owner thread is live; we must NOT fork a fresh pool here. Surface the
                            # break so OverlappedEvaluationEngine quiesces the producer and degrades.
                            raise
                        warnings.warn("Persistent refine pool broke (worker death); disabling it and "
                                      "falling back to a per-call fork pool for the rest of this run.")
                        self.close()
                    else:
                        for result, warning_msg in outcomes:
                            if warning_msg and converge_error == 'print':
                                print(warning_msg)
                            if result is not None:
                                entry = self._create_result_entry(payload=result, input_dim=input_dim)
                                if entry is not None:
                                    results.append(entry)
                        ran_on_pool = True

                if ran_on_pool:
                    pass
                elif use_parallel:
                    ctx = mp.get_context('fork')
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
                                    results.append(entry)
                else:
                    for job in _iterate_with_progress(
                        jobs,
                        total=len(jobs),
                        verbose=verbose,
                        desc="Fitting Constants",
                    ):
                        serial_payload = job.copy()
                        serial_payload.update({'X': gs.X_np, 'y': gs.y_np, 'simplipy_engine': self.simplipy_engine})
                        result, warning_msg = _refine_candidate_worker(serial_payload)
                        if warning_msg and converge_error == 'print':
                            print(warning_msg)
                        if result is not None:
                            entry = self._create_result_entry(payload=result, input_dim=input_dim)
                            if entry is not None:
                                results.append(entry)

            with _RefinementContext(self.simplipy_engine, gs.X_np, gs.y_np):
                _t_ref = time.time()
                _run_refinement_jobs(refinement_jobs)
                refinement_time += time.time() - _t_ref

                prune_count_candidates = [r for r in results if np.isfinite(r.get('fvu', np.nan))]
                top_k_resolved = self._resolve_prune_count(len(prune_count_candidates))

                if top_k_resolved > 0:
                    sorted_results = sorted(prune_count_candidates, key=lambda r: r.get('fvu', np.inf))
                    top_results = sorted_results[:top_k_resolved]

                    seen_expressions = {tuple(r['expression']) for r in results}
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
                            prompt_prefix=gs.prompt_prefix,
                            memory=gs.memory_for_scoring,
                            device=gs.device,
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
                                'y_variance': gs.y_variance,
                                'length_penalty': self.length_penalty,
                                'constants_penalty': self.constants_penalty,
                                'likelihood_penalty': self.likelihood_penalty,
                                'complexity': gs.complexity,
                                'metadata_snapshot': gs.metadata_snapshot,
                            }
                            pruning_jobs.append(pruning_job)

                    _t_prune_ref = time.time()
                    _run_refinement_jobs(pruning_jobs)
                    refinement_time += time.time() - _t_prune_ref

        sorted_results, results_df = self._compile_results_pure(
            results, self.length_penalty, self.constants_penalty, self.likelihood_penalty)

        return FitResult(
            results=sorted_results,
            results_df=results_df,
            refinement_time=refinement_time,
            generation_time=gs.generation_time,
            variable_mapping=gs.variable_mapping,
            input_dim=input_dim,
            prompt_metadata=gs.metadata_snapshot,
        )

    def _apply_fit_result(self, fit_result: "FitResult") -> None:
        """Commit a FitResult to instance state (the serial ``fit`` and the overlap engine's commit)."""
        self._results = fit_result.results
        self.results = fit_result.results_df
        self._refinement_time = fit_result.refinement_time
        self._generation_time = fit_result.generation_time
        self._input_dim = fit_result.input_dim
        self.variable_mapping = fit_result.variable_mapping
        self._prompt_metadata = copy.deepcopy(fit_result.prompt_metadata) if fit_result.prompt_metadata is not None else None

    def compile_results(
            self,
            length_penalty: float | None = None,
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

        self._results, self.results = self._compile_results_pure(
            self._results, self.length_penalty, self.constants_penalty, self.likelihood_penalty)

    def _compile_results_pure(
            self,
            results: list[Any],
            length_penalty: float,
            constants_penalty: float,
            likelihood_penalty: float) -> tuple[list[Any], "pd.DataFrame"]:
        """Pure core of :meth:`compile_results`: score + sort + build the DataFrame for a results
        list, returning ``(sorted_results, results_df)``.

        Reads ``self`` only for the read-only scoring helpers and config; writes NOTHING to ``self``,
        so the refinement phase (and the overlap engine) can compile a problem's results without
        touching shared state. Raises ConvergenceError if ``results`` is empty.
        """
        if not results:
            raise ConvergenceError("The optimization did not converge for any beam")

        # Compute the new score for each result
        for result in results:
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
                        length_penalty,
                        constants_penalty,
                        likelihood_penalty,
                    )
                else:
                    result['score'] = np.nan

        # Sort the results by the best loss of each beam. The third key is a deterministic
        # tie-break on the expression tokens: `sorted` is stable, so without it two exactly
        # equal scores keep their insertion order, which is the (non-deterministic) parallel
        # refinement completion order. Candidates are deduplicated to distinct expressions
        # (unique=True), so the token tuple is a total order over ties -> the final ranking is
        # independent of completion order (a prerequisite for byte-identical overlap).
        sorted_results = list(sorted(results, key=lambda x: (
            x['score'] if not np.isnan(x['score']) else float('inf'),
            np.isnan(x['score']),
            tuple(map(str, x.get('expression', [])))
        )))

        # Attach only the best fit (lowest loss) per beam for readability
        best_fit_payloads: list[tuple[np.ndarray, np.ndarray | None, float]] = []
        for result in sorted_results:
            fits_list = result.get('fits', [])
            if fits_list:
                best_fit = cast(tuple[np.ndarray, np.ndarray | None, float], min(fits_list, key=lambda tpl: tpl[2]))
            else:
                best_fit = cast(tuple[np.ndarray, np.ndarray | None, float], (np.array([]), None, float('nan')))
            best_fit_payloads.append(best_fit)

        results_df = pd.DataFrame(sorted_results)
        if not results_df.empty:
            results_df['beam_id'] = results_df.index
            results_df['fit_constants'] = [bf[0] for bf in best_fit_payloads]
            results_df['fit_covariances'] = [bf[1] for bf in best_fit_payloads]
            results_df['fit_loss'] = [bf[2] for bf in best_fit_payloads]
            results_df.drop(columns=['fits'], inplace=True, errors='ignore')

        return sorted_results, results_df

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
