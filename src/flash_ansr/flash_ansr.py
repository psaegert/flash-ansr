"""The :class:`FlashANSR` estimator: end-to-end amortized neural symbolic regression.

This module wires the transformer backbone (:class:`~flash_ansr.model.FlashANSRModel`),
tokenizer, prompt preprocessing, candidate generation (softmax sampling)
and the constant :class:`~flash_ansr.refine.Refiner` into a single scikit-learn-style estimator
whose :meth:`FlashANSR.fit` / :meth:`FlashANSR.predict` / :meth:`FlashANSR.infer` methods recover
closed-form expressions from ``(X, y)`` data.
"""
import os
import copy
import math
import time
import hashlib
import numbers
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Any, Iterable, Iterator, Mapping, TypedDict, Callable, Sequence, TypeVar, cast
import warnings

import numpy as np
import pandas as pd
import torch

from flash_ansr.utils.numeric import NUMERIC_DTYPE
from flash_ansr.utils.weights import load_weights
from tqdm import tqdm

from sklearn.base import BaseEstimator

from simplipy import SimpliPyEngine
from symbolic_data.token_ops import normalize_skeleton

from flash_ansr._refine_pool import RecoverableForkPool
from flash_ansr.generation import run_softmax_sampling
from flash_ansr.model import FlashANSRModel, Tokenizer
from flash_ansr.preprocessing import (
    CapabilityUnavailable, PromptPrefix, apply_emission_flag, prepare_prompt_prefix)
from flash_ansr.refine import (Refiner, ConvergenceError, fit_sort_key, RefineScope,
                               DEFAULT_REFINE_SCOPE, refinement_slots, literal_value,
                               TypedSpanPolicy, TYPED_SPAN_POLICIES, DEFAULT_TYPED_SPAN_POLICY,
                               typed_literal_sites, thaw_typed_literals, typed_thaw_subsets)
from flash_ansr.tasks import (
    DEFAULT_SAMPLES, ComplexityDistribution, ValueDistribution, predict_complexity,
    predict_constants, predict_y, score_outliers)
from flash_ansr.spelling import ConstantLadderConfig, ladder_floor, respell_fitted_candidate
from flash_ansr.scoring import (
    PARETO_RANK_NOT_COMPUTED,
    RANKING_METRICS,
    RankingConfig,
    RankingError,
    compute_fvu,
    count_constants,
    is_constant_token,
    non_dominated_ranks,
    normalize_variance,
    objective_vector,
    order_rows,
    resolve_ranking,
    score_from_fvu,
    score_row,
)
from flash_ansr.model.flash_ansr_model import _VRAM_GUARD_FRACTION
from flash_ansr.utils.generation import GenerationConfig, SoftmaxSamplingConfig, suggest_batch_size, suggest_batch_size_dims, _FULL_CAP_MIN_VRAM_GB, _spill_over_budget
from flash_ansr.utils.paths import substitute_root_path
from flash_ansr.utils.skeleton import NonFiniteExpressionError, record_non_finite_drop, simplify_and_mask
from flash_ansr.data.serialization import TAGGED_DELIMITER_TOKENS
from flash_ansr.utils.ieee754 import IEEE754_START_TOKEN
from flash_ansr.utils.tensor_ops import pad_input_set
from flash_ansr.inference import Candidate, FitResult, build_candidate_ledger, _best_constants
from flash_ansr.estimator_config import RefineConfig, ComputeConfig, ranking_from
from simplipy.engine import Mode



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
    pruned_variant: bool
    #: simplipy mu of the REALIZED expression in milli-bits, or None when the candidate could not be
    #: priced. Distinct from `Candidate.mu`, which is the masked-SKELETON unit `fit(complexity=)`
    #: consumes -- see RANKING_SPEC.md section 3.
    mdl: float | None
    #: The model's own decoded constants, before refinement. Carried since the v24 constants lane but
    #: never declared here.
    constants_emitted: list[float] | None
    #: Front index under a `pareto` ranking (0 = non-dominated); PARETO_RANK_NOT_COMPUTED (-1) when
    #: the row was ordered by a scalar score instead.
    pareto_rank: int
    spelling: str | None      # constant re-spelling record of a ladder variant; None for a fitted draw
    replaces_parent: bool     # the variant stands in for its parent (a tie), rather than beside it
    #: How many model-predicted literals were kept VERBATIM in a typed position instead of being
    #: refined (`refiner_typed_spans`); 0 on the shipped path.
    typed_frozen: int
    #: The typed token indices this row re-fitted after the frozen round (the `freeze_then_free`
    #: duplicate), space separated; None for a row that is not such a duplicate.
    typed_thaw: str | None


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


@contextmanager
def _seeded_generation(seed: int | None, *, prior_sampler: Any = None) -> Iterator[None]:
    """Seed one draw: the prior sampler's stream, and torch's generators for softmax sampling
    (saved and restored around the call, so a seeded fit leaves the global RNG state as it found
    it). ``seed=None`` touches nothing."""
    if seed is None:
        yield
        return
    if prior_sampler is not None:
        prior_sampler.reseed(int(seed))
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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


#: FVU at or below which a T11 verbatim (model-predicted) init is accepted as final and the
#: multi-restart refinement is skipped. float32 eps is the same bar srbf's `is_perfect_fit` uses,
#: so a skipped fallback can never cost a recovery the benchmark would have counted.
_T11_ACCEPT_FVU: float = float(np.finfo(np.float32).eps)


def _score_or_inf(result: Mapping[str, Any]) -> float:
    score = result.get('score')
    if score is None:
        return float('inf')
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float('inf')
    return value if np.isfinite(value) else float('inf')


def price_realized(simplipy_engine: Any, refiner: Refiner, expression_tokens: Sequence[str]) -> float | None:
    """The MDL price (milli-bits) of the REALIZED expression: the fitted constants substituted into
    the skeleton, priced certified / f64 / canon default -- the ranking currency (see
    RANKING_SPEC.md section 2: mode= and canon= written out, never inherited). ``None`` for an
    unpriceable candidate: it keeps its fvu and its place in the scalar ranking, the scorer decides
    what a missing price means. A partially substituted expression (a leftover ``<constant>``,
    which `transform` leaves in place when there are fewer fitted values than sites) is refused
    rather than priced at the flat skeleton rate -- a wrong price is the invisible failure."""
    try:
        realized_tokens = list(refiner.transform(
            expression=list(expression_tokens), return_prefix=True, variable_mapping=None))
        if any(tok == '<constant>' for tok in realized_tokens):
            raise ValueError("realized expression still carries a <constant> placeholder")
        return float(simplipy_engine.complexity(
            realized_tokens, certified=True, mode=Mode.f64, canon='default'))
    except Exception:
        return None


_price_realized = price_realized   # the private spelling, kept for the refine worker's call sites


def canonicalize_fitted(simplipy_engine: Any, refiner: Refiner, expression: Sequence[str], X: np.ndarray, *,
                        n_variables: int, refine_scope: RefineScope = DEFAULT_REFINE_SCOPE
                        ) -> tuple[Refiner, list[str], bool, bool]:
    """The fitted candidate in simplipy's canonical form: the expression the certified price describes.

    Generation simplifies the SKELETON, where ``tanh(x)^c1 / tanh(x)^c2`` cannot cancel; the refiner then
    fits both constants to 2 and the realized expression collapses to ``exp(-x^2)`` -- which is what
    :func:`price_realized` prices (the certified price canonicalizes), while the emitted expression stayed
    the skeleton with the numbers filled in (measured 2026-09-11 on the T8-20M r-sweep readings: 15 % of
    the emitted answers longer than their canonical form, 2.5 % of the rows an exact recovery hidden by
    it; ``y = 1.2`` came out as a nine-constant expression whose ``x / (... / log(1.0))`` term is zero).

    Realizes the fit, simplifies the realized tokens, re-abstracts the canonical form's fittable literals
    into ``<constant>`` slots under ``refine_scope`` (typed literals stay verbatim, so a fitted exponent
    that folded to ``2`` becomes the literal ``2``), carries the values over as the one fit (the loss is
    the fitted one; no covariance) and verifies the prediction on ``X`` is unchanged. Returns
    ``(refiner, expression, changed, same_slot_count)``: the input pair unchanged when the canonical form
    is not strictly shorter than the realized one (a re-spelling, not a collapse), when a slot value is
    not a plain number, when the carried refiner is not valid or when the prediction differs on a
    support point (a removable singularity the data hits).
    """
    tokens = list(expression)
    try:
        realized = list(refiner.transform(expression=tokens, return_prefix=True, variable_mapping=None))
        if any(tok == '<constant>' for tok in realized):
            return refiner, tokens, False, True
        simplified = list(simplipy_engine.simplify(realized))
    except Exception:  # noqa: BLE001 - an expression the engine cannot read or simplify stays as fitted
        return refiner, tokens, False, True
    # Only a STRICTLY SHORTER canonical form is taken: the canon also re-spells exact rationals
    # (``* 1.5 sin x1`` -> ``/ * 3 sin x1 2``), which is not a collapse, changes the slot structure
    # and is the ladder's business (its surprise test decides whether a fraction is worth it).
    if len(simplified) >= len(realized) or not simplipy_engine.is_valid(simplified):
        return refiner, tokens, False, True
    slots = refinement_slots(simplified, simplipy_engine, refine_scope)
    slot_set = set(slots)
    canonical = ['<constant>' if i in slot_set else tok for i, tok in enumerate(simplified)]
    try:
        values = np.asarray([float(simplified[i]) for i in slots], dtype=float)
    except (TypeError, ValueError):   # a slot spelled by a symbol (np.pi): keep the fitted spelling
        return refiner, tokens, False, True
    best = refiner._all_constants_values[0]
    old_values = np.asarray(best[0], dtype=float).ravel()
    loss = float(best[2])
    try:
        carried = Refiner.from_serialized(simplipy_engine=simplipy_engine, n_variables=int(n_variables), expression=canonical,
                                          n_inputs=int(X.shape[1]), fits=[(values, None, loss)], refine_scope='placeholders')
        if not carried.valid_fit:
            return refiner, tokens, False, True
        with np.errstate(all='ignore'):
            y_old = np.asarray(refiner.predict(X), dtype=float).reshape(-1)
            y_new = np.asarray(carried.predict(X), dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001
        return refiner, tokens, False, True
    if y_old.shape != y_new.shape or not np.allclose(y_old, y_new, rtol=1e-9, atol=1e-12, equal_nan=True):
        return refiner, tokens, False, True
    return carried, canonical, True, values.size == old_values.size


def _serialize_fits(refiner: Refiner) -> list[tuple[np.ndarray, np.ndarray | None, float]]:
    serialized: list[tuple[np.ndarray, np.ndarray | None, float]] = []
    for constants, constants_cov, fit_loss in refiner._all_constants_values:
        cov_payload: np.ndarray | None
        if constants_cov is None or getattr(constants_cov, 'size', 0) == 0:
            cov_payload = None
        else:
            cov_payload = np.asarray(constants_cov)
        serialized.append((np.asarray(constants), cov_payload, float(fit_loss)))
    return serialized


def respell_result(payload: dict[str, Any], simplipy_engine: Any, refiner: Refiner, X: np.ndarray, y: np.ndarray,
                   result: dict[str, Any]) -> dict[str, Any] | None:
    """The constant ladder on one fitted result: the re-spelled variant as a second result dict
    (fitted under ``refine_scope='placeholders'`` so its spelled literals stay frozen), or ``None``
    when no spelling beats the parent's score. The variant keeps the parent's beam and
    provenance; ``spelling`` records what changed. ``constant_ladder`` None/False -> no ladder.

    Public because a candidate fitted OUTSIDE the generation loop (another method's expression
    joining the pool) goes through the same pass: ``payload`` carries ``constant_ladder``,
    ``ranking_weights``, ``expression`` (the abstracted tokens), ``log_prob``, ``y_variance``,
    ``n_variables``, ``method``, ``n_restarts``, ``p0_noise``, ``p0_noise_kwargs``; ``result``
    carries the parent's ``fvu``, ``mdl``, ``score``."""
    ladder = payload.get('constant_ladder')
    if ladder is None or result.get('mdl') is None:
        return None
    config = ladder if isinstance(ladder, ConstantLadderConfig) else ConstantLadderConfig.from_mapping(ladder)
    if config is None:
        return None
    weights = payload['ranking_weights']
    expression_tokens = list(payload['expression'])
    log_prob = payload.get('log_prob')

    def score_fn(fvu: float, mdl: float, constant_count: int) -> float:
        return score_row({'fvu': fvu, 'expression': expression_tokens, 'constant_count': constant_count,
                          'log_prob': log_prob, 'mdl': mdl}, weights)

    # Rounds: a re-spelled variant is itself a fitted candidate (a redundant pair of constants
    # collapses into one float in the first round, which then deserves its own spelling). Each
    # round must beat the previous one; at most one round per constant.
    current_refiner, current_expression = refiner, expression_tokens
    current = {'fvu': float(result['fvu']), 'mdl': result['mdl'], 'score': float(result['score'])}
    variant: dict[str, Any] | None = None
    records: list[str] = []
    replaces_parent = True
    n_rounds = max(1, len(getattr(refiner, 'slot_indices', []) or []))
    for _ in range(n_rounds):
        try:
            step = respell_fitted_candidate(
                refiner=current_refiner, expression=current_expression, X=X, y=y,
                y_variance=float(payload['y_variance']),
                parent_fvu=current['fvu'], parent_mdl=current['mdl'], parent_score=current['score'],
                score_fn=score_fn, compute_fvu=FlashANSR._compute_fvu,
                price_realized=lambda r, toks: _price_realized(simplipy_engine, r, toks),
                simplipy_engine=simplipy_engine, n_variables=int(payload['n_variables']), method=payload['method'],
                full_fit={'n_restarts': payload['n_restarts'], 'p0_noise': payload['p0_noise'],
                          'p0_noise_kwargs': payload['p0_noise_kwargs']},
                config=config)
        except Exception:
            step = None
        if step is None:
            break
        variant = step
        records.append(step['spelling'])
        replaces_parent = replaces_parent and bool(step.get('replaces_parent', False))
        current_refiner, current_expression = step['refiner'], step['expression']
        current = {'fvu': step['fvu'], 'mdl': step['mdl'], 'score': step['score']}
        if not step['constant_count']:
            break
    if variant is None:
        return None

    # The ladder MINTS collapses (#163): re-spelling two near-equal constants to exactly equal makes
    # `c * x / c` cancel, and a near-zero to exactly 0 makes `0 * x` vanish -- but the ladder runs
    # AFTER `canonicalize_fitted`, so nothing re-simplifies what it created. The certified price is
    # unaffected (the pricer canonicalizes internally), so the row scores correctly while the EMITTED
    # answer carries junk: `(x - 0.3)**2` came out as `pow(x + atan(1/(-(0 * x) - 3.2327)), 2)`,
    # priced at the law's own 19,585 mB. Canonicalize the winning variant, which is the one thing the
    # ladder leaves behind.
    variant_refiner, variant_expression = variant['refiner'], list(variant['expression'])
    variant_constant_count, variant_complexity = variant['constant_count'], variant['complexity']
    variant_score = variant['score']
    try:
        carried, canonical, canonicalized, _same = canonicalize_fitted(
            simplipy_engine, variant_refiner, variant_expression, X,
            n_variables=int(payload['n_variables']), refine_scope='placeholders')
    except Exception:  # noqa: BLE001 -- an un-canonicalizable variant is emitted as the ladder left it
        canonicalized = False
    if canonicalized:
        variant_refiner, variant_expression = carried, list(canonical)
        variant_constant_count = sum(1 for tok in variant_expression if FlashANSR._is_constant_token(tok))
        variant_complexity = len(variant_expression)
        # fvu and mdl are unchanged -- canonicalize_fitted verifies the prediction and the price is
        # taken on the realized form -- but the LENGTH metrics moved, so the score is re-taken.
        variant_score = score_fn(float(variant['fvu']), variant['mdl'], variant_constant_count)

    child = dict(result)
    child.update({
        'expression': variant_expression,
        'constant_count': variant_constant_count,
        'complexity': variant_complexity,
        'mdl': variant['mdl'],
        'fvu': variant['fvu'],
        'score': variant_score,
        'fits': _serialize_fits(variant_refiner),
        'valid_fit': True,
        'refine_scope': 'placeholders',
        'spelling': ' | '.join(records),
        'respelled': None,
        # A re-spelling of a frozen row is still that row: it inherits the freeze provenance from
        # `dict(result)` above, but the parent's typed-span DUPLICATES are not its own.
        'thawed': None,
        # every round only tied its parent: the same candidate, spelled canonically -> it takes the
        # parent's place in the pool; a strict improvement stands beside the parent
        'replaces_parent': replaces_parent,
    })
    return child


_respell_result = respell_result   # the private spelling, kept for the refine worker's call sites


_RESPELL_PARENT_KEYS = ('log_prob', 'fvu', 'score', 'expression', 'constant_count', 'mdl', 'complexity',
                        'requested_complexity', 'raw_beam', 'beam', 'raw_beam_decoded', 'constants_emitted',
                        'pruned_variant', 'typed_frozen', 'typed_thaw')


def _respell_candidate_worker(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The constant ladder on one ALREADY FITTED candidate (the pool-bound path): rebuild its refiner
    from the serialized fits and run the same ``_respell_result`` the fit worker runs inline."""
    simplipy_engine = payload.get('simplipy_engine') or _GLOBAL_SIMPLIPY_ENGINE
    if simplipy_engine is None:
        raise RuntimeError("Re-spelling worker does not have access to a SimpliPyEngine instance.")
    numpy_errors = payload.get('numpy_errors')
    numpy_state = np.geterr()
    if numpy_errors is not None:
        np.seterr(all=numpy_errors)
    X, y = _resolve_refinement_arrays(payload)
    seed = payload.get('seed')
    numpy_rng_state = None
    if seed is not None:
        numpy_rng_state = np.random.get_state()
        np.random.seed(seed)
    try:
        refiner = Refiner.from_serialized(
            simplipy_engine=simplipy_engine, n_variables=payload['n_variables'], expression=payload['expression'],
            n_inputs=int(X.shape[1]), fits=payload['fits'], refine_scope=payload.get('refine_scope', DEFAULT_REFINE_SCOPE))
        if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
            return None, None
        result = {key: payload.get(key) for key in _RESPELL_PARENT_KEYS}
        result.update({'fits': payload['fits'], 'valid_fit': True, 'spelling': None, 'respelled': None})
        child = _respell_result(payload, simplipy_engine, refiner, X, y, result)
    finally:
        np.seterr(**numpy_state)
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)
    return child, None


def _thawed_variants(payload: dict[str, Any], simplipy_engine: Any, X: np.ndarray, y: np.ndarray,
                     parent: dict[str, Any]) -> list[dict[str, Any]]:
    """The typed-span THAW: the frozen candidate refitted with its typed literals free again.

    Round one froze the model's predicted exponent and fitted only the constants an optimizer can
    actually move (`freeze_typed_slots`). This is round two of ``refiner_typed_spans =
    'freeze_then_free'``: the SAME candidate with the exponent turned back into a slot, seeded at
    the value it was frozen at, and every other slot seeded at round one's optimum. It is a
    DUPLICATE -- its own result dict, its own price, its own ladder pass -- so both spellings stand
    in the pool and the ranking decides; nothing is replaced. ``'combinations'`` asks for one
    duplicate per non-empty subset of the typed sites instead of the single all-thawed one.

    Returns ``[]`` whenever the handshake does not line up: no typed literal was frozen, the
    parent's fits do not match its slots, or a seed is missing for some slot. A duplicate is an
    extra candidate, never a reason to lose the parent.
    """
    policy = payload.get('typed_spans', DEFAULT_TYPED_SPAN_POLICY)
    if policy not in ('freeze_then_free', 'combinations'):
        return []
    fits = parent.get('fits') or []
    if not fits:
        return []
    expression = list(parent['expression'])
    scope: RefineScope = payload.get('refine_scope', DEFAULT_REFINE_SCOPE)
    parent_values = np.asarray(fits[0][0], dtype=float).ravel()
    try:
        parent_slots = refinement_slots(expression, simplipy_engine, scope)
    except Exception:  # noqa: BLE001
        return []
    if len(parent_slots) != parent_values.size:
        return []
    seed_by_index = {index: float(value) for index, value in zip(parent_slots, parent_values)}

    variants: list[dict[str, Any]] = []
    for subset in typed_thaw_subsets(expression, simplipy_engine, policy):
        thawed, thawed_values = thaw_typed_literals(expression, simplipy_engine, subset)
        if not thawed_values:
            continue
        try:
            slots = refinement_slots(thawed, simplipy_engine, scope)
        except Exception:  # noqa: BLE001
            continue
        # p0 per slot IN SLOT ORDER: the predicted value for a thawed exponent, round one's fitted
        # value for everything else. A slot with neither has no defensible seed, so the whole
        # variant is dropped rather than fitted from a silently wrong init.
        p0: list[float] = []
        for index in slots:
            if index in thawed_values:
                p0.append(float(thawed_values[index]))
            elif index in seed_by_index:
                p0.append(seed_by_index[index])
            else:
                break
        if not p0 or len(p0) != len(slots):
            continue
        child_payload = dict(payload)
        child_payload.update({
            'expression': thawed,
            'constant_count': sum(1 for tok in thawed if FlashANSR._is_constant_token(tok)),
            'p0': p0,
            # The duplicate is fitted on the shipped path: it must not freeze again (its typed
            # literals are the very slots it exists to fit) nor spawn duplicates of its own.
            'typed_spans': 'refine',
            'typed_frozen': 0,
            # The mark goes into the PAYLOAD, not onto the result afterwards: `_fit_one_candidate` reads it
            # into the result and the constant ladder clones the result for its child, so a ladder child of
            # a thawed duplicate carries the mark too. Set on the result after the fact, the child was
            # already cloned without it and read as a plain candidate's child (2026-09-12 attribution).
            'typed_thaw': ' '.join(str(index) for index in subset),
            'seed': _candidate_refine_seed(payload.get('seed'), ('thaw', *map(str, subset))),
        })
        child, _warning = _fit_one_candidate(child_payload, simplipy_engine, X, y)
        if child is None:
            continue
        variants.append(child)
    return variants


def _refine_candidate_worker(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    simplipy_engine = payload.get('simplipy_engine') or _GLOBAL_SIMPLIPY_ENGINE
    if simplipy_engine is None:
        raise RuntimeError("Refinement worker does not have access to a SimpliPyEngine instance.")

    X, y = _resolve_refinement_arrays(payload)
    result, warning = _fit_one_candidate(payload, simplipy_engine, X, y)
    if result is not None and payload.get('typed_frozen'):
        # Only a candidate that actually FROZE something has a thaw to run; the rest are already
        # fitted with every constant free.
        variants = _thawed_variants(payload, simplipy_engine, X, y, result)
        if variants:
            result['thawed'] = variants
    return result, warning


def _fit_one_candidate(payload: dict[str, Any], simplipy_engine: Any,
                       X: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any] | None, str | None]:
    """One candidate all the way through: verbatim init, restart search, canonicalization, price,
    score and the constant ladder. Split out of :func:`_refine_candidate_worker` so a duplicate of
    the same candidate (the typed-span thaw) travels the identical path -- same handshake, same
    canonicalization, same pricer, same ladder -- rather than a parallel copy of it."""
    numpy_errors = payload.get('numpy_errors')
    numpy_state = np.geterr()
    if numpy_errors is not None:
        np.seterr(all=numpy_errors)

    warning = None
    seed = payload.get('seed')
    numpy_rng_state = None
    if seed is not None:
        numpy_rng_state = np.random.get_state()
        np.random.seed(seed)

    try:
        refiner = Refiner(simplipy_engine=simplipy_engine, n_variables=payload['n_variables'])
        p0_values = payload.get('p0')
        verbatim_fits: list | None = None
        verbatim_fvu = float('inf')
        if p0_values is not None:
            # v24 handshake (contract T11): the beam's predicted float32 constants are the
            # init, VERBATIM -- no noise, one restart (float32 embeds exactly in float64,
            # so np.asarray is bit-preserving).
            refiner.fit(
                expression=payload['expression'],
                X=X,
                y=y,
                n_restarts=1,
                method=payload['method'],
                p0=np.asarray(p0_values, dtype=float),
                p0_noise=None,
                p0_noise_kwargs=None,
                converge_error='ignore',
                refine_scope=payload.get('refine_scope', DEFAULT_REFINE_SCOPE),
            )
            if refiner.valid_fit:
                verbatim_fits = list(refiner._all_constants_values)
                verbatim_fvu = FlashANSR._compute_fvu(
                    float(refiner.loss), int(y.shape[0]), payload['y_variance'])

        # T11 is an INIT, not a replacement for the search. The gate used to be finiteness alone,
        # so any converged-but-catastrophic local optimum cancelled the 8-restart refinement:
        # measured on y = 1 + 3*sin(2x) with the model predicting frequency 9 instead of 2, the
        # single verbatim restart returned loss 4.64 (fvu 0.99954, score +0.3998) and the fallback
        # never ran, while the same skeleton with the fallback reaches loss 0.0 (score -15.25) --
        # and a strictly worse `<constant> * x1` decoy scored +0.2224 and was ranked FIRST.
        # Skip the fallback only when the verbatim fit already explains the data to the bar the
        # benchmark itself calls a perfect recovery; nothing downstream can improve on that.
        if p0_values is None or not (verbatim_fvu <= _T11_ACCEPT_FVU):
            # Exploratory refinement -- and the T11 fallback on a bad init: the verbatim
            # attempt consumes no RNG, so this call sees the same stream an init-free
            # worker would and reproduces its fits exactly.
            refiner.fit(
                expression=payload['expression'],
                X=X,
                y=y,
                n_restarts=payload['n_restarts'],
                method=payload['method'],
                p0=None,
                p0_noise=payload['p0_noise'],
                p0_noise_kwargs=payload['p0_noise_kwargs'],
                converge_error=payload['converge_error'],
                refine_scope=payload.get('refine_scope', DEFAULT_REFINE_SCOPE),
            )
            if verbatim_fits:
                # Keep the BETTER of the two, never the last one: `fit` resets the fit list, so
                # without this merge a good verbatim fit is discarded by a worse exploration.
                refiner._assign_fits(list(refiner._all_constants_values) + verbatim_fits)
    except ConvergenceError:
        if payload['converge_error'] == 'raise':
            raise
        warning = f"Failed to converge for beam: {payload['expression']}" if payload['converge_error'] == 'print' else None
    except (NameError, KeyError) as exc:
        # An UNREALIZABLE candidate, not a failed optimization: the expression names an
        # operator the loaded engine's realization does not define (measured: a
        # `pow1_5` sugar operator under an acj-4 engine -> NameError from lambdify;
        # KeyError from the realization table). One such candidate must not kill the
        # whole problem's refinement -- it is dropped like any invalid fit.
        if payload['converge_error'] == 'raise':
            raise ConvergenceError(f"Unrealizable candidate {payload['expression']}") from exc
        warning = (f"Unrealizable candidate ({exc!r}): {payload['expression']}"
                   if payload['converge_error'] == 'print' else None)
    finally:
        np.seterr(**numpy_state)
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)

    if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
        warning = f"Failed to converge for beam: {payload['expression']}" if payload['converge_error'] == 'print' else None
        if payload['converge_error'] == 'raise':
            raise ConvergenceError("The optimization did not converge")
        return None, warning

    loss = float(refiner._all_constants_values[0][-1])
    sample_count = int(y.shape[0])
    fvu = FlashANSR._compute_fvu(loss, sample_count, payload['y_variance'])

    # The fitted candidate in canonical form (owner 2026-09-11): the constants the refiner chose can make
    # a spelling collapsible that the skeleton was not, and the certified price already prices that
    # collapsed form -- so it is the collapsed form that gets emitted (and that the ladder starts from).
    expression_as_fitted = list(payload['expression'])
    refiner, canonical_expression, canonicalized, same_slots = canonicalize_fitted(
        simplipy_engine, refiner, expression_as_fitted, X,
        n_variables=int(payload['n_variables']), refine_scope=payload.get('refine_scope', DEFAULT_REFINE_SCOPE))
    if canonicalized:
        payload['expression'] = canonical_expression
        payload['constant_count'] = 0
        if not same_slots:
            p0_values = None   # the model's emitted constants no longer align with the slots

    expression_tokens = payload['expression']
    constant_count = int(payload.get('constant_count', 0))
    if constant_count <= 0:
        constant_count = sum(1 for tok in expression_tokens if FlashANSR._is_constant_token(tok))
        payload['constant_count'] = constant_count

    # `mdl` is priced on the REALIZED expression -- the fitted constants substituted in -- per the
    # owner's ruling of 2026-09-04 (RANKING_SPEC.md section 2). The emitted spelling carries only
    # <constant> placeholders at a flat 67,000 mB each, which makes mu a re-parameterisation of
    # (n_nodes, n_constants) to 99% of within-problem pairs; the realized spelling additionally
    # prices constant PRECISION, which no count can express and which a future coarse-graining of
    # constants must be able to move. Measured cost: 3.9 us for the transform + 30.9 us for the
    # pricer per candidate = 0.04% of fit wall-clock.
    #
    # mode= and canon= are written out, never inherited: mode routes the parse and moves mu
    # (permissive differs on 1.0-1.7% of holdout laws, up to +62%), and training data is generated
    # at simplify_mode: permissive.
    mdl: float | None = _price_realized(simplipy_engine, refiner, expression_tokens)
    score = score_row(
        {'fvu': fvu, 'expression': expression_tokens, 'constant_count': constant_count,
         'log_prob': payload.get('log_prob'), 'mdl': mdl},
        payload['ranking_weights'],
    )

    serialized_fits = _serialize_fits(refiner)

    result = {
        'log_prob': payload['log_prob'],
        'fvu': fvu,
        'score': score,
        'expression': payload['expression'],
        'constant_count': payload['constant_count'],
        'mdl': mdl,
        'complexity': len(payload['expression']),
        'requested_complexity': payload['complexity'],
        'raw_beam': payload['raw_beam'],
        'beam': payload['beam'],
        'raw_beam_decoded': payload['raw_beam_decoded'],
        'fits': serialized_fits,
        'valid_fit': refiner.valid_fit,
        'pruned_variant': bool(payload.get('pruned_variant', False)),
        # The model's OWN constants (contract T11's verbatim init). They were decoded, used as the
        # optimizer seed and then dropped on the floor -- so nothing downstream could compare what
        # the model predicted against what the refiner made of it. Measured on FastSRB, refinement
        # improves FVU on only 55% of comparable rows, so the emitted values are not a curiosity.
        'constants_emitted': list(p0_values) if p0_values is not None else None,
        'spelling': None,
        'respelled': None,
        # How many PREDICTED literals were kept verbatim in a typed position, and (on a thawed
        # duplicate) which typed sites it re-fitted. Without these the pool cannot tell the frozen
        # row from its duplicate.
        'typed_frozen': int(payload.get('typed_frozen', 0) or 0),
        'typed_thaw': payload.get('typed_thaw'),
        'thawed': None,
        # the spelling the refiner fitted, when the canonical form emitted above differs from it
        'expression_as_fitted': expression_as_fitted if canonicalized else None,
    }

    # The ladder refits variants, so it consumes the RNG -- and on this inline path it used to run
    # AFTER the seeded block was restored, which made it the one irreproducible stage: two identical
    # runs over one candidate pool agreed on every FIT row and differed on 4,729 ladder rows. The
    # pool-bound path (`_respell_candidate_worker`) has always seeded it; this makes the two agree.
    ladder_seed = _candidate_refine_seed(seed, ('ladder', *(str(t) for t in payload.get('raw_beam') or ())))
    ladder_rng_state = None
    if seed is not None:
        ladder_rng_state = np.random.get_state()
        np.random.seed(ladder_seed)
    try:
        result['respelled'] = _respell_result(payload, simplipy_engine, refiner, X, y, result)
    finally:
        if ladder_rng_state is not None:
            np.random.set_state(ladder_rng_state)
    return result, None


# --- parallel post-generation simplify (the c=32k generation lever; ~11x on the simplify portion) ---
_SIMPLIFY_PARALLEL_THRESHOLD = 4096   # only fork for the simplify when draws >= this (else serial)
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


def _simplify_pool_worker(raw_expr: tuple) -> tuple | None:
    """Simplify one raw expression (pure CPU; deterministic -> byte-identical to serial).
    simplipy.simplify is the equivalence loop only; mask() relabels any emitted literals to
    <constant> for the model's vocabulary (matching the pre-carve-out masked output).

    Returns ``None`` when the expression folds to a non-finite skeleton. The worker does NOT drop
    the candidate itself: it runs in a forked process, so it can neither raise across the fork
    without killing the run nor increment the parent's drop count. Reporting by omission leaves
    the key out of the lookup map, and the parent then takes its normal serial path -- which
    raises, drops and counts in the process that can be observed."""
    try:
        return tuple(simplify_and_mask(_SIMPLIFY_ENGINE, list(raw_expr)))
    except NonFiniteExpressionError:
        return None


@dataclass
class Generation:
    """The output of :meth:`FlashANSR.generate`: the raw draws of one problem plus everything the
    refinement phase needs, so the two phases share NO instance state (which is what lets an
    overlapped engine generate problem N+1 while problem N refines).

    ``raw_beams`` are the deduplicated draws as token ids (prompt prefix included), ``log_probs``
    their summed token log-likelihoods; ``X`` / ``y`` the support set the model read (numpy, padded
    to the model's width); ``memory`` the encoder memory the decoder attended to (the learned null
    memory under ``guidance_weight=0``); ``prompt_prefix`` the prompt the draws continued.
    """
    raw_beams: list
    log_probs: list
    X: np.ndarray
    y: np.ndarray
    y_variance: float
    prompt_prefix: Any
    memory: Any
    device: Any
    variable_mapping: dict
    complexity: Any
    generation_time: float
    draws: int
    seed: int | None = None


@dataclass
class _RefineOutcome:
    """Output of the CPU refinement phase (``_fit_refine``): the ordered result rows (with their
    refiners) and the phase time. Nothing is written to ``self`` during refinement."""
    results: list
    refinement_time: float
    generation_time: float
    variable_mapping: dict
    input_dim: int | None


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
    ranking_mode : {'mdl', 'weighted', 'pareto'}, optional
        How refined candidates are ordered (RANKING_SPEC.md). ``'mdl'`` (default): ``log10(fvu)``
        plus ``mdl_strength`` times the description length, in bits, of the REALIZED expression.
        ``'weighted'``: ``log10(fvu)`` plus ``ranking_weights`` over the metric registry.
        ``'pareto'``: the non-dominated front over ``ranking_metrics``, ordered within a front by
        ``ranking_tie_break``. Each knob belongs to one mode; passing it with another raises.
    mdl_strength : float or None, optional
        Mode ``'mdl'`` only: decades of FVU per bit. ``None`` -> the engineered default
        :data:`flash_ansr.scoring.MDL_STRENGTH_DEFAULT` (1e-2).
    ranking_weights : dict[str, float] or None, optional
        Mode ``'weighted'`` only: weight per metric name (``n_nodes``, ``n_constants``,
        ``n_constant_placeholders``, ``n_typed_literals``, ``mdl`` (per bit), ``neg_log_prob``).
        Absent metrics weigh 0. The pre-0.14 default ranking is ``{'n_nodes': 0.05}``.
    ranking_metrics : sequence of str or None, optional
        Mode ``'pareto'`` only: the front's axes; must include ``'fvu'``. ``None`` -> ``('fvu', 'n_nodes')``.
    ranking_tie_break : str or None, optional
        Mode ``'pareto'`` only: the metric ordering candidates within a front (may lie outside
        ``ranking_metrics``). ``None`` -> ``'fvu'``.
    constant_ladder : mapping or bool or None, optional
        Constant re-spelling after the fit (``flash_ansr.spelling``). For every fitted candidate each
        constant is offered its cheaper spellings (integer, small fraction, rounding, pi or e
        multiple, zero); the fit's curvature predicts their score, the best are frozen as literals
        and the rest re-fitted once, and the re-spelled candidate joins the pool beside its parent
        only if its score beats the parent's (a tie replaces the parent). ``True`` (default) = the
        defaults: fractions only when the continued-fraction surprise test passes
        (``fraction_surprise='denominator'``) and roundings to 1..8 digits, on EVERY fitted
        candidate so the fit-quality / MDL Pareto front stays intact (``pool_bound=True`` restricts
        the pass to the candidates that could still reach rank 0, for time-budgeted runs where only
        the returned answer matters). ``None``/``False`` = off. A mapping overrides ``fraction_surprise``,
        ``pool_bound``, ``max_denominator`` (1000), ``digits``, ``special_constants``
        (('np.pi', 'np.e')), ``max_relative_step`` (0.5), ``max_decades`` (1.0),
        ``confirm_decades`` (0.1), ``n_restarts`` (1).
    refiner_workers : int or None, optional
        Number of worker processes to run during constant refinement. ``None``
        (the default) uses all available CPU cores, while explicit integers
        select a fixed pool size. Set ``0`` to disable multiprocessing.
    prune_constant_budget : int or float, optional
        Apply constant-pruning refinement to the best beams after the initial
        refinement (ranked by FVU). If ``> 1`` treated as an absolute count; if
        ``0 < value <= 1`` treated as a fraction of beams (ceil). Set to 0 to
        disable. Defaults to 0 (disabled): the pruning path does not run unless a budget is set, so ``Candidate.pruned_variant`` is False throughout.
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
            n_nodes: int,
            constant_count: int,
            log_prob: float | None,
            node_penalty: float,
            constants_penalty: float,
            likelihood_penalty: float,
            mdl: float | None = None,
            mdl_penalty: float = 0.0) -> float:
        return score_from_fvu(
            fvu, n_nodes, constant_count, log_prob,
            node_penalty, constants_penalty, likelihood_penalty, mdl, mdl_penalty)

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

        # The numeric channel must be PRESENT, not None. A mixed-representation checkpoint was
        # trained with input_num at full sequence length on every instance -- NaN at every
        # non-payload position -- so the model added a learned constant vector there throughout.
        # Passing None takes the `numeric_embeddings = None` branch and skips that addition at
        # EVERY position, which scores a rescored candidate under a different model than the one
        # that generated it. The prefix's own numeric payload (_base_num) was also being discarded.
        input_num = None
        if self.flash_ansr_model._has_numeric_channel():
            input_num = torch.full((batch_size, max_len), float('nan'),
                                   device=device, dtype=NUMERIC_DTYPE)
            if _base_num is not None:
                prefix_numeric = torch.tensor(
                    [float(value) for value in _base_num], device=device, dtype=NUMERIC_DTYPE)
                input_num[:, :prefix_numeric.shape[0]] = prefix_numeric

        with torch.no_grad():
            logits = self.flash_ansr_model.forward(input_tokens=input_tokens, data=None, input_num=input_num, memory=memory)
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

    _TAGGED_DELIMITERS = frozenset(TAGGED_DELIMITER_TOKENS)

    def _ensure_explicit_dialect(self, tokens: list[str]) -> list[str] | None:
        """v24 models emit the engine's tagged canonical dialect; the refiner and every
        downstream consumer read explicit binary-prefix. One conversion, at the decode
        boundary. A sequence that carries tagged delimiters but does not parse is a
        malformed emission and returns None (an invalid candidate, not an error)."""
        if not any(t in self._TAGGED_DELIMITERS for t in tokens):
            return tokens
        try:
            return list(self.simplipy_engine.to_prefix(list(tokens)))
        except Exception:
            return None

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

        # The exhaustive 2**constant_count powerset (one full tree-prune per mask) is intractable for a
        # high-constant beam (e.g. 20 constants -> ~1M prunes per beam x top_k beams). Below a threshold
        # keep the exact powerset (behaviour-identical); above it fall back to a bounded, deterministic
        # set covering the useful prunings (remove none, remove all, each single removal, each single
        # keep) -- O(constant_count) instead of O(2**n).
        _MAX_EXHAUSTIVE_PRUNE = 12
        masks: Iterable[int]
        if constant_count <= _MAX_EXHAUSTIVE_PRUNE:
            masks = range(1 << constant_count)
        else:
            _full = (1 << constant_count) - 1
            masks = [0, _full, *(1 << i for i in range(constant_count)), *(_full ^ (1 << i) for i in range(constant_count))]

        for mask in masks:
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
            *,
            generation_config: GenerationConfig | None = None,
            refine: RefineConfig | Mapping[str, Any] | None = None,
            ranking: RankingConfig | Mapping[str, Any] | str | None = None,
            compute: ComputeConfig | Mapping[str, Any] | None = None,
            model_directory: str | None = None):
        """The estimator's POLICY, in four config objects (owner design 2026-09-14): the sampler
        (``generation_config``), the refiner (``refine``), the candidate ranking (``ranking``) and
        where it runs (``compute``). A call to :meth:`fit` describes the problem and the question;
        nothing here changes per problem except the search budget, which ``fit(draws=)`` overrides.
        Each config accepts its object or a plain mapping (a YAML config can spell it)."""
        self.simplipy_engine = simplipy_engine
        self.flash_ansr_model = flash_ansr_model.eval()
        self.tokenizer = tokenizer
        # Where the bundle came from (``load``): the training prior ``prior_sampling`` reads by default.
        self.model_directory = model_directory
        self._prior_sampler_cache: Any = None

        if generation_config is None:
            generation_config = SoftmaxSamplingConfig()

        self.generation_config = generation_config
        self.refine: RefineConfig = RefineConfig.from_mapping(refine)
        # Validated in every mode at construction, so a metric typo cannot lie dormant until
        # someone flips the mode. Read-only from here on; `FitResult.rerank` is call-scoped.
        self.ranking: RankingConfig = ranking_from(ranking)
        self.compute: ComputeConfig = ComputeConfig.from_mapping(compute)
        if getattr(generation_config, 'method', None) == 'prior_sampling':
            # Build the prior's catalog now (its holdout registration takes a minute): setup,
            # not the per-problem generation time a benchmark records.
            self._prior_sampler()

        # Parallelize the post-generation simplify across `refiner_workers` (gated to draws >=
        # _SIMPLIFY_PARALLEL_THRESHOLD; byte-identical to serial). Set False to force serial.
        self.parallel_simplify = True

        #: The result of the last :meth:`fit` (``None`` before the first); `predict`,
        #: `get_expression` and `results` read it.
        self.result_: FitResult | None = None
        self._prompt_prefix: PromptPrefix | None = None

        # Optional persistent pre-CUDA fork pool (inference-speed Step 3). When set (via
        # ``load(persistent_refine_pool=True)``) refinement + simplify route onto it instead of
        # forking a fresh pool per call; ``None`` keeps the legacy per-call-fork path (the default).
        self._refine_pool: RecoverableForkPool | None = None

        # Load is the first moment both the generation config and the tokenizer are in hand, so a
        # decoding option this vocabulary cannot serve is refused HERE, not after an encoder pass.
        self._validate_checkpoint()

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
            *,
            generation_config: GenerationConfig | None = None,
            refine: RefineConfig | Mapping[str, Any] | None = None,
            ranking: RankingConfig | Mapping[str, Any] | str | None = None,
            compute: ComputeConfig | Mapping[str, Any] | None = None) -> "FlashANSR":
        """Instantiate a `FlashANSR` estimator from a checkpoint directory.

        Parameters
        ----------
        directory : str
            Directory that contains ``model.yaml``, ``tokenizer.yaml`` and ``model.safetensors``.
        generation_config : SoftmaxSamplingConfig or PriorSamplingConfig, optional
            The sampler's policy (the draw budget, the emission format, temperature, ...). Default:
            ``SoftmaxSamplingConfig()``.
        refine : RefineConfig or mapping, optional
            The refiner's policy (optimizer, restarts, scope, typed spans, the constant ladder,
            constant pruning, the numpy error policy). Default: ``RefineConfig()``.
        ranking : RankingConfig or mapping or str, optional
            The candidate ranking: ``'mdl'`` (default: ``log10(FVU)`` plus ``1e-2`` decades per bit
            of the refined expression's description length), ``'weighted'`` or ``'pareto'``, with
            that mode's knobs as a mapping (``{'mode': 'mdl', 'mdl_strength': 1e-2}``).
        compute : ComputeConfig or mapping, optional
            The torch ``device``, the refiner ``workers`` (``None`` = every core, ``0`` = serial) and
            ``persistent_pool`` (one worker pool forked BEFORE any CUDA initialization and reused
            across calls: the structural mitigation for the fork-after-CUDA deadlock family; requires
            the ``fork`` start method and more than one worker, otherwise a no-op).

        Returns
        -------
        model : FlashANSR
            Fully initialized estimator, on ``compute.device``.
        """
        directory = substitute_root_path(directory)
        compute_cfg = ComputeConfig.from_mapping(compute)
        device = compute_cfg.device

        flash_ansr_model_path = os.path.join(directory, 'model.yaml')
        tokenizer_path = os.path.join(directory, 'tokenizer.yaml')

        # When a persistent pre-CUDA pool is requested for a non-CPU device, defer the device move:
        # load weights on CPU (incl. map_location, which also touches CUDA otherwise) so the pool can
        # be forked while CUDA is still uninitialized.
        defer_cuda = compute_cfg.persistent_pool and str(device) != 'cpu'
        load_device = 'cpu' if defer_cuda else device

        model = FlashANSRModel.from_config(flash_ansr_model_path)
        load_weights(model, directory, device=load_device)
        model.eval().to(load_device)

        tokenizer = Tokenizer.from_config(tokenizer_path)

        nsr = cls(
            simplipy_engine=model.simplipy_engine,
            flash_ansr_model=model,
            tokenizer=tokenizer,
            model_directory=directory,
            generation_config=generation_config,
            refine=refine,
            ranking=ranking,
            compute=compute_cfg)

        if compute_cfg.persistent_pool:
            # Warm the engine + fork the pool (pre-CUDA), then move the model to the target device.
            nsr._enable_persistent_refine_pool(target_device=device)
        elif defer_cuda:
            nsr.to(device)

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
                'refine_scope': self.refiner_scope,
                'converge_error': 'ignore',
                'numpy_errors': self.numpy_errors,
                'y_variance': 1.0,
                'ranking_weights': self.ranking.effective_weights,
                'complexity': None,
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

    # Read-only views of the config objects (the names the pipeline reads).
    @property
    def n_restarts(self) -> int:
        return self.refine.n_restarts

    @property
    def refiner_method(self) -> str:
        return self.refine.method

    @property
    def refiner_p0_noise(self) -> str | None:
        return self.refine.p0_noise

    @property
    def refiner_p0_noise_kwargs(self) -> dict[str, Any] | None:
        return None if self.refine.p0_noise_kwargs is None else dict(self.refine.p0_noise_kwargs)

    @property
    def refiner_scope(self) -> RefineScope:
        return self.refine.scope

    @property
    def refiner_typed_spans(self) -> TypedSpanPolicy:
        return self.refine.typed_spans

    @property
    def constant_ladder(self) -> ConstantLadderConfig | None:
        return self.refine.ladder

    @property
    def numpy_errors(self) -> str | None:
        return self.refine.numpy_errors

    @property
    def prune_constant_budget(self) -> float:
        return self.refine.prune_constant_budget

    @property
    def refiner_workers(self) -> int:
        return self.compute.resolved_workers

    @property
    def results(self) -> pd.DataFrame:
        """The last fit's refined candidates as a DataFrame (``result_.to_dataframe()``)."""
        return self._require_result().to_dataframe()

    @property
    def variable_mapping(self) -> dict[str, str]:
        """The last fit's variable mapping (``x1..xN`` -> the caller's names)."""
        return dict(self.result_.variable_mapping) if self.result_ is not None else {}

    def _require_result(self) -> FitResult:
        if self.result_ is None:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")
        return self.result_

    def _validate_checkpoint(self) -> None:
        """Refuse a checkpoint this harness does not serve, at LOAD time.

        `SoftmaxSamplingConfig` already validates the compact/constrain/use_cache pairings at config
        time, but nothing checked the vocabulary -- so `constrain_ieee754=True` on a checkpoint
        without the span tokens constructed happily, loaded happily, and failed inside `fit()` only
        after shape coercion, the R1 scan, the prompt build and a full encoder forward. Load is the
        first moment both the config and the tokenizer are in hand, so it is where this belongs
        (design principle 4).
        """
        # This harness serves mixed-representation (v24+) checkpoints only: a vocabulary without
        # <ieee754> is not a model this code can serve. Earlier generations need a 0.14.x release.
        if IEEE754_START_TOKEN not in self.tokenizer:
            raise ValueError(
                f"This checkpoint's vocabulary has no {IEEE754_START_TOKEN} token, so it is not a "
                f"v24 (mixed-representation) model. flash-ansr serves v24+ only; use a 0.14.x "
                f"release to run an earlier checkpoint.")
        guidance_weight = getattr(self.generation_config, 'guidance_weight', None)
        if guidance_weight is not None and not getattr(self.flash_ansr_model, 'optional_condition', False):
            raise CapabilityUnavailable(
                "guidance_weight needs a model trained with optional_condition=True (the learned "
                "null_memory); this checkpoint has no unconditioned mode.")

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
                return {raw: value for raw, value in zip(raw_list, simplified) if value is not None}
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
        # Non-finite folds come back as None and are OMITTED, so _postprocess_sampled misses the
        # key and falls back to its guarded serial simplify (which drops + counts in this process).
        return {raw: value for raw, value in zip(raw_list, simplified) if value is not None}

    def _prior_sampler(self) -> Any:
        """The training-prior sampler of ``prior_sampling``, built once from the generation config
        (its ``catalog``, else ``catalog_train.yaml`` beside the loaded model)."""
        if self._prior_sampler_cache is None:
            from flash_ansr.prior import PriorSampler, resolve_prior_catalog

            config = self.generation_config
            catalog = resolve_prior_catalog(getattr(config, 'catalog', None), self.model_directory)
            self._prior_sampler_cache = PriorSampler(
                catalog, engine=self.simplipy_engine, tokenizer=self.tokenizer,
                decontaminate=bool(getattr(config, 'decontaminate', True)), seed=getattr(config, 'seed', None))
        return self._prior_sampler_cache

    def _sample(
        self,
        data: torch.Tensor,
        *,
        prompt_prefix: PromptPrefix | None = None,
        complexity: int | float | None = None,
        verbose: bool = False,
        memory: torch.Tensor | None = None,
        n_active_variables: int | None = None,
        draws: int | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        """Draw candidate expression beams from the sampler for one prepared data tensor.

        The public entry point is :meth:`generate`, which prepares the data; this is the sampler
        dispatch (the decoder's softmax sampling or the training prior). ``draws`` overrides the
        generation config's budget for this call.

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
        n_active_variables : int, optional
            The problem's number of input columns before padding; ``prior_sampling`` conditions
            its draws on it (``match_variables``). Ignored by the decoder.

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

        generation_kwargs = self.generation_config.to_kwargs()
        if draws is not None:
            generation_kwargs['draws'] = int(draws)
        generation_kwargs.pop('emission', None)   # a prompt-prefix matter, resolved by `generate`

        effective_prompt = prompt_prefix
        if effective_prompt is None and complexity is not None:
            preprocessor = getattr(self.flash_ansr_model, 'preprocessor', None)
            effective_prompt = prepare_prompt_prefix(preprocessor, complexity=complexity)

        match self.generation_config.method:
            case 'prior_sampling':
                # The training prior instead of the decoder: the data is not consulted here (it
                # reaches the refiner and the ranking), and a draw carries no log-probability.
                weights = self.ranking.effective_weights
                if float(weights.get('neg_log_prob', 0.0)) != 0.0:
                    raise RankingError(
                        "prior_sampling candidates carry no log-probability, but the ranking weights "
                        "'neg_log_prob'; use ranking_mode='mdl' or weights without it.")
                match = bool(generation_kwargs.get('match_variables', True))
                return self._prior_sampler().draw(
                    int(generation_kwargs.get('draws', 1)),
                    unique=bool(generation_kwargs.get('unique', True)),
                    valid_only=bool(generation_kwargs.get('valid_only', True)),
                    max_tries=generation_kwargs.get('max_tries'),
                    n_variables=(int(n_active_variables) if (match and n_active_variables is not None) else None),
                )
            case 'softmax_sampling':
                choices_target = int(generation_kwargs.get('draws', 1))
                generation_kwargs = dict(generation_kwargs)  # local copy; resolve sentinels into it

                # --- Resolve static-decode (multi-chunk regime) + the c-adaptive batch TOGETHER ---
                # The static arm's chunk batch is BOTH the regime probe (static iff batch < draws) AND the
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
                    _static_kwarg, draws=choices_target, static_batch=_static_batch)
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
                        # 0.0 until a device reports otherwise: an unqueryable device falls to
                        # _SMALL_CARD_BATCH_CAP rather than earning the caps measured on a 24 GiB card.
                        _vram_gb = 0.0
                        if _is_cuda:
                            try:
                                _vram_gb = torch.cuda.get_device_properties(_dev).total_memory / 1e9
                            except Exception:
                                pass
                        elif getattr(_dev, 'type', None) == 'mps':
                            try:
                                _vram_gb = torch.mps.recommended_max_memory() / 1e9
                            except Exception:
                                pass
                        _raw_bs = suggest_batch_size(choices_target, self._n_params, _vram_gb)
                    if verbose:
                        print(f"[auto batch] draws={choices_target} static={_static} -> batch_size={_raw_bs}")
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
                        ck['draws'] = this_chunk
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
                    chunk_kwargs['draws'] = this_chunk
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
            case _:
                raise ValueError(f"Invalid generation method: {self.generation_config.method}")

    def _prepare_prompt_prefix(
            self,
            *,
            complexity: int | float | None,
            emission: str = 'fittable') -> PromptPrefix | None:
        preprocessor = getattr(self.flash_ansr_model, 'preprocessor', None)
        prompt_prefix = prepare_prompt_prefix(preprocessor, complexity=complexity)

        if emission != 'constants':
            if prompt_prefix is None:
                # No preprocessor -> no prefix was built, but an emission flag still has to reach
                # the model. Materialize the bare generation prefix and flag that, which is what
                # the capability probes did by hand.
                tokens, numeric = self.flash_ansr_model._resolve_generation_prefix(prompt_prefix=None)
                numeric_values = list(numeric) if numeric is not None else [float('nan')] * len(tokens)
                prompt_prefix = PromptPrefix(
                    tokens=list(tokens), numeric=numeric_values,
                    mask=[True] * len(tokens), metadata={})
            prompt_prefix = apply_emission_flag(prompt_prefix, emission, self.tokenizer)

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
            refine_scope=payload.get('refine_scope', self.refiner_scope),
        )

        if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
            return None

        entry: Result = {
            'log_prob': payload['log_prob'],
            'fvu': payload['fvu'],
            'score': payload['score'],
            'expression': payload['expression'],
            'constant_count': int(payload.get('constant_count', self._count_constants(payload['expression']))),
            # `_create_result_entry` is a SELECTIVE key-by-key whitelist, not a dict copy. A key the
            # worker produces but this literal omits is dropped silently: every mdl would be None,
            # every score with mdl_penalty != 0 would be nan, the sort would degenerate to the token
            # tie-break, and the run would rank ALPHABETICALLY while fit() reported success. All
            # three design critiques found this line; all three designs missed it.
            'mdl': payload.get('mdl'),
            'complexity': payload['complexity'],
            'requested_complexity': payload.get('requested_complexity'),
            'raw_beam': payload['raw_beam'],
            'beam': payload['beam'],
            'raw_beam_decoded': payload['raw_beam_decoded'],
            'function': refiner.expression_lambda,
            'refiner': refiner,
            'fits': copy.deepcopy(refiner._all_constants_values),
            'pruned_variant': bool(payload.get('pruned_variant', False)),
            'constants_emitted': payload.get('constants_emitted'),
            'pareto_rank': PARETO_RANK_NOT_COMPUTED,
            'spelling': payload.get('spelling'),
            'replaces_parent': bool(payload.get('replaces_parent', False)),
            'typed_frozen': int(payload.get('typed_frozen', 0) or 0),
            'typed_thaw': payload.get('typed_thaw'),
        }

        return entry

    @staticmethod
    def _dedup_respelled(results: list) -> list:
        """Ladder variants that canonicalize to an expression already in the pool (another variant,
        or a fitted draw) are duplicates: keep the best-scoring row per expression, fitted draws
        always kept. The order of the survivors is preserved."""
        if not any(r.get('spelling') for r in results):
            return results
        best_variant: dict[tuple, int] = {}
        drawn: set[tuple] = set()
        for i, r in enumerate(results):
            key = tuple(map(str, r.get('expression', [])))
            if not r.get('spelling'):
                drawn.add(key)
                continue
            j = best_variant.get(key)
            if j is None or _score_or_inf(r) < _score_or_inf(results[j]):
                best_variant[key] = i
        keep = set(best_variant.values())
        return [r for i, r in enumerate(results)
                if not r.get('spelling') or (i in keep and tuple(map(str, r.get('expression', []))) not in drawn)]

    def _append_result_entries(self, results: list, result: dict[str, Any], input_dim: int) -> None:
        """A worker result, its typed-span duplicates, and each row's re-spelled ladder variant.

        The parent comes first. A ladder variant that only TIED its parent replaces it (the same
        candidate, spelled canonically); a typed-span duplicate never replaces anything -- it is a
        different fit of the same skeleton and has to be ranked against the parent on its own."""
        rows: list[dict[str, Any] | None] = []
        for parent in [result, *(result.get('thawed') or [])]:
            child = parent.get('respelled')
            if child is not None and child.get('replaces_parent'):
                rows.append(child)
            else:
                rows.extend([parent, child])
        for payload in rows:
            if payload is None:
                continue
            entry = self._create_result_entry(payload=payload, input_dim=input_dim)
            if entry is not None:
                results.append(entry)

    def fit(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto',
            *,
            draws: int | None = None,
            complexity: int | float | None = None,
            seed: int | None = None,
            on_empty: Literal['return', 'raise'] = 'return',
            verbose: bool = False) -> FitResult:
        """Symbolic regression on ``(X, y)``: draw candidates, fit their constants, rank them.

        Returns the :class:`~flash_ansr.inference.FitResult` (the score-sorted refined candidates,
        the full candidate ledger, the ranking that ordered them and the phase times) and keeps it
        as ``self.result_``, which :meth:`predict`, :meth:`get_expression` and :attr:`results` read.
        The estimator's policy (sampler, refiner, ranking, compute) is fixed at construction; this
        call carries only the problem and the question.

        Parameters
        ----------
        X : ndarray or Tensor or DataFrame
            Feature matrix where rows index observations and columns variables.
        y : ndarray or Tensor or DataFrame or Series
            Target values. Multi-output targets are unsupported.
        variable_names : list[str] or dict[str, str] or {'auto'} or None, optional
            Names for the columns (``'auto'``: a DataFrame's column names, else ``x1..xN``).
        draws : int, optional
            The search budget for this call, overriding the generation config's ``draws``.
        complexity : int or float or None, optional
            Target complexity in **simplipy mu** -- the unit ``simplipy_engine.complexity(skeleton)``
            returns (roughly 1e3-1e6, NOT a token count); ``Candidate.mu`` reports it back, so a
            value can round-trip. Emitted as the trained bare ``<complexity>`` block.
        seed : int or None, optional
            Seeds the draw and the constant refinement (per-candidate refiner seeds are derived
            from it through ``np.random.SeedSequence``, so refinement is reproducible and
            independent of completion order). Softmax sampling on a GPU is best-effort only.
        on_empty : {'return', 'raise'}, optional
            What to do when NO candidate fitted: return the (empty) result with its ledger, or
            raise :class:`ConvergenceError`. A candidate whose refinement fails is never an error;
            it is a ``FIT_FAILED`` row of the ledger.
        verbose : bool, optional
            Progress bars and the refiner's convergence warnings.

        Raises
        ------
        ValueError
            If ``y`` has more than one output dimension or ``X`` carries non-finite values.
        ConvergenceError
            Under ``on_empty='raise'``, when no candidate fitted.
        """
        if on_empty not in ('return', 'raise'):
            raise ValueError(f"on_empty must be 'return' or 'raise'; got {on_empty!r}")
        self.result_ = None

        numpy_errors_before = np.geterr()
        np.seterr(all=self.numpy_errors)
        try:
            generation = self.generate(X, y, variable_names, draws=draws, complexity=complexity, seed=seed, verbose=verbose)
            outcome = self._fit_refine(
                generation,
                converge_error='print' if verbose else 'ignore',
                refine_seed=seed,
                verbose=verbose,
                allow_empty=True,
            )
        finally:
            np.seterr(**numpy_errors_before)

        result = self._build_result(generation, outcome)
        self.result_ = result
        if on_empty == 'raise' and not result.candidates:
            raise ConvergenceError("The optimization did not converge for any beam")
        return result

    def _build_result(self, generation: "Generation", outcome: "_RefineOutcome") -> FitResult:
        """The public result from the ordered refined rows: plain-data candidates + the ledger."""
        results = outcome.results  # already ordered by _order_results

        def _decode_expr(raw_beam: list[int]) -> list[str] | None:
            expr_ids = self.flash_ansr_model.tokenizer.extract_expression_from_beam(raw_beam)[0]
            return self._ensure_explicit_dialect(self.tokenizer.decode_expression(expr_ids))

        ledger = build_candidate_ledger(
            generation.raw_beams, generation.log_probs, results,
            decode_expr=_decode_expr, is_valid=self.simplipy_engine.is_valid,
        )
        variable_mapping = outcome.variable_mapping
        candidates: list[Candidate] = []
        for rank, r in enumerate(results):
            refiner = r['refiner']
            expression_prefix = refiner.transform(expression=r['expression'], return_prefix=True, variable_mapping=None)
            expression_infix = refiner.transform(expression=r['expression'], return_prefix=False, variable_mapping=variable_mapping)
            skeleton_prefix = normalize_skeleton(r['expression'])
            candidates.append(Candidate(
                raw_beam=list(r['raw_beam']),
                expression=list(r['expression']),
                slots=[int(i) for i in getattr(refiner, 'slot_indices', [])],
                expression_prefix=list(expression_prefix) if expression_prefix is not None else [],
                expression_infix=str(expression_infix),
                skeleton_prefix=list(skeleton_prefix) if skeleton_prefix is not None else [],
                constants=_best_constants(r),
                constants_emitted=(list(r['constants_emitted']) if r.get('constants_emitted') is not None else None),
                log_prob=float(r.get('log_prob', float('nan'))),
                score=float(r.get('score', float('nan'))),
                fvu=float(r.get('fvu', float('nan'))),
                n_nodes=int(r.get('complexity', len(r['expression']))),
                # mu, NOT the token count: fit(complexity=) consumes simplipy mu (1e3-1e6) while
                # `complexity` above is a token count (~1e1). Computed here (bounded by the refined
                # survivors), never per beam.
                mu=self._skeleton_mu(skeleton_prefix),
                # The RANKING currency, priced by the refine worker on the REALIZED expression.
                # Read through, never recomputed here.
                mdl=r.get('mdl'),
                constant_count=int(r.get('constant_count', 0)),
                pruned_variant=bool(r.get('pruned_variant', False)),
                pareto_rank=int(r.get('pareto_rank', PARETO_RANK_NOT_COMPUTED)),
                rank=rank,
                spelling=r.get('spelling'),
                typed_frozen=int(r.get('typed_frozen', 0) or 0),
                typed_thaw=r.get('typed_thaw'),
            ))
        return FitResult(
            candidates=candidates,
            ledger=ledger,
            generation_time=generation.generation_time,
            refinement_time=outcome.refinement_time,
            ranking=self.ranking,
            n_variables=self.n_variables,
            variable_mapping=dict(variable_mapping or {}),
            draws=generation.draws,
            engine=self.simplipy_engine,
        )

    def generate(
            self,
            X: np.ndarray | torch.Tensor | pd.DataFrame,
            y: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series,
            variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto',
            *,
            draws: int | None = None,
            complexity: int | float | None = None,
            seed: int | None = None,
            verbose: bool = False) -> "Generation":
        """The generation phase alone: read ``(X, y)``, draw candidates, return a :class:`Generation`.

        :meth:`fit` is this followed by refinement and ranking. Writes NOTHING to ``self``, so a
        caller that wants the raw draws (their token ids, log-likelihoods, the encoder memory and
        the prompt they continued) gets them without touching the fitted state.

        Parameters
        ----------
        X, y : array-like
            The support set: ``(n_points, n_columns)`` features and ``(n_points,)`` targets.
        variable_names : list[str] or dict[str, str] or {'auto'} or None, optional
            Names for the columns (``'auto'``: a DataFrame's column names, else ``x1..xN``).
        draws : int, optional
            The search budget for this call; ``None`` = the generation config's ``draws``.
        complexity : int or float, optional
            A target complexity in simplipy mu for the trained ``<complexity>`` block.
        seed : int, optional
            Seeds the draw: the prior sampler's stream, or the torch generator for softmax
            sampling (best-effort on a GPU, whose kernels are not bitwise reproducible).
        verbose : bool, optional
            Progress bars.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif y.shape[-1] != 1:
            raise ValueError("The target data must have a single output dimension")

        # R1 (simplipy SIMPLIFICATION_CONTRACT_v2 §3): variables range over R -- nan/inf are never
        # INPUTS. Every simplification rule is certified under that assumption, and an embedded
        # nonfinite input silently changes what a rewrite means (`x0 + (x1 - x1)` with a nan x1
        # column is nan unsimplified but x0 simplified). Fail loudly at the boundary instead.
        _X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else (
            X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X))
        if not np.all(np.isfinite(_X_arr)):
            n_bad = int(np.sum(~np.all(np.isfinite(_X_arr), axis=-1)))
            raise ValueError(
                f"X contains non-finite values in {n_bad} row(s). Variables range over the reals "
                f"(contract R1): drop or impute non-finite rows before calling fit().")

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
            # dtype is NOT the caller's choice: the pre-encoder reinterprets rather than
            # converts, so a float32 tensor handed straight through would be viewed as int64 and
            # emit scrambled bits at the right shape. Normalize both branches.
            if not isinstance(X, torch.Tensor):
                if isinstance(X, pd.DataFrame):
                    X = torch.tensor(X.values, dtype=NUMERIC_DTYPE, device=self.flash_ansr_model.device)
                else:
                    X = torch.tensor(X, dtype=NUMERIC_DTYPE, device=self.flash_ansr_model.device)
            else:
                X = X.to(device=self.flash_ansr_model.device, dtype=NUMERIC_DTYPE)

            if not isinstance(y, torch.Tensor):
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = torch.tensor(y.values, dtype=NUMERIC_DTYPE, device=self.flash_ansr_model.device)
                else:
                    y = torch.tensor(y, dtype=NUMERIC_DTYPE, device=self.flash_ansr_model.device)
            else:
                y = y.to(device=self.flash_ansr_model.device, dtype=NUMERIC_DTYPE)

            if y.dim() == 1:
                y = y.unsqueeze(-1)

            # The R1 gate above ran on the caller's array; the cast on the way in can still
            # MANUFACTURE non-finite values (an overflowing narrowing sends |x| to inf, which
            # enters the encoder and the refiner as inf -- exactly what the gate exists to
            # prevent). v25 reads binary64, so this only bites a caller who arrives wider than
            # that, but the check is what makes the boundary honest either way.
            if not bool(torch.isfinite(X).all()):
                raise ValueError(
                    f"X contains values that are not finite in {NUMERIC_DTYPE}, the dtype the model "
                    f"reads. Rescale the inputs before calling fit().")

            sample_count = y.shape[0]
            # Over the FINITE rows only -- the same mask the refiner fits under (refine.py `_r1`).
            # A single non-finite y made `y.var()` nan, `_compute_fvu` return +inf for EVERY
            # candidate, and `_compile_results_pure` rewrite every score to nan; the sort then fell
            # through to its third key and ranked candidates ALPHABETICALLY while fit() still
            # reported success (measured: one nan row in 64 destroyed the ranking silently).
            y_finite_mask = torch.isfinite(y).all(dim=-1)
            n_finite = int(y_finite_mask.sum().item())
            if n_finite < sample_count:
                warnings.warn(
                    f"y contains {sample_count - n_finite} non-finite value(s); they are excluded "
                    f"from the target variance and from every fit (the refiner masks them too).",
                    RuntimeWarning, stacklevel=2)
            if n_finite <= 1:
                # Variance is undefined for a single sample; skip the reduction so downstream scoring
                # quietly falls back to the residual loss via ``_compute_fvu``.
                y_variance = float('nan')
            else:
                # ddof=0 (biased) to match the residual loss mean(diff**2) and the eval metric
                # numeric.fvu, so the selection FVU equals the evaluation FVU exactly.
                y_variance = y[y_finite_mask].var(dim=0, unbiased=False).item()

            n_input_columns = int(X.shape[1])  # the problem's own input count, before padding
            X = pad_input_set(X, self.n_variables)

            # Concatenate x and y along the feature dimension
            data_tensor = torch.cat([X, y], dim=-1)

            guidance_weight = getattr(self.generation_config, 'guidance_weight', None)
            if guidance_weight is not None and float(guidance_weight) == 0.0:
                # The trained UNCONDITIONED mode (condition_dropout): the learned null_memory
                # replaces the encoder's, so candidates come from the model's PRIOR over
                # expressions rather than from this data set. The data is still used downstream --
                # refinement fits the constants and scores the fits -- which makes this the
                # "propose from the prior, fit to the data" arm, not a blind decode. One problem,
                # so batch 1: the sampler broadcasts a batch-1 memory over its candidates.
                memory_for_scoring = self.flash_ansr_model.null_memory.detach().to(device=data_tensor.device)
            else:
                memory_for_scoring = self.flash_ansr_model._create_memory(data_tensor)

            is_prior = getattr(self.generation_config, 'method', None) == 'prior_sampling'
            emission = 'constants' if is_prior else str(getattr(self.generation_config, 'emission', 'fittable'))
            prompt_prefix = self._prepare_prompt_prefix(
                emission=emission,
                complexity=complexity,
            )

            resolved_draws = int(draws) if draws is not None else int(getattr(self.generation_config, 'draws', 1))
            _t_gen = time.time()
            with _seeded_generation(seed, prior_sampler=(self._prior_sampler() if is_prior else None)):
                raw_beams, log_probs, _completed_flags, _rewards = self._sample(
                    data_tensor,
                    prompt_prefix=prompt_prefix,
                    complexity=complexity,
                    verbose=verbose,
                    memory=memory_for_scoring,
                    n_active_variables=n_input_columns,
                    draws=resolved_draws,
                )
            generation_time = time.time() - _t_gen

            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()

        return Generation(
            raw_beams=raw_beams,
            log_probs=log_probs,
            X=X_np,
            y=y_np,
            y_variance=y_variance,
            prompt_prefix=prompt_prefix,
            memory=memory_for_scoring,
            device=data_tensor.device,
            variable_mapping=variable_mapping,
            complexity=complexity,
            generation_time=generation_time,
            draws=resolved_draws,
            seed=seed,
        )

    @property
    def _ladder_bounded(self) -> bool:
        """The constant ladder runs as a post-fit pass under the pool bound (see ConstantLadderConfig)."""
        ladder = getattr(self, 'constant_ladder', None)
        return ladder is not None and bool(getattr(ladder, 'pool_bound', False))

    def _run_ordered_jobs(self, jobs: list[dict[str, Any]], worker: Any, gs: "Generation", *, desc: str,
                          verbose: bool) -> list[Any]:
        """Run ``worker`` over ``jobs`` and return the outcomes IN ORDER, on the same executor the
        fit phase uses: the persistent pool (per-job X/y), a per-call fork pool, or serially."""
        if not jobs:
            return []
        available_methods = mp.get_all_start_methods()
        max_workers = min(self.refiner_workers, len(jobs))
        use_parallel = max_workers > 1 and 'fork' in available_methods
        if self._refine_pool is not None and (use_parallel or self._overlap_mode):
            for job in jobs:
                job['X'] = gs.X
                job['y'] = gs.y
            chunksize = max(1, len(jobs) // (max(1, max_workers) * 8))
            try:
                return list(self._refine_pool.map_ordered(
                    worker, jobs, chunksize=chunksize, recover=False,
                    wrap=(lambda it: _iterate_with_progress(it, total=len(jobs), verbose=verbose, desc=desc))))
            except BrokenProcessPool:
                if self._overlap_mode:
                    raise
                warnings.warn("Persistent refine pool broke (worker death); disabling it and "
                              "falling back to a per-call fork pool for the rest of this run.")
                self.close()
        if use_parallel:
            ctx = mp.get_context('fork')
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                return list(_iterate_with_progress(executor.map(worker, jobs), total=len(jobs), verbose=verbose, desc=desc))
        outcomes = []
        for job in _iterate_with_progress(jobs, total=len(jobs), verbose=verbose, desc=desc):
            serial_payload = job.copy()
            serial_payload.update({'X': gs.X, 'y': gs.y, 'simplipy_engine': self.simplipy_engine})
            outcomes.append(worker(serial_payload))
        return outcomes

    def _run_bounded_ladder(self, results: list, gs: "Generation", *, input_dim: int, converge_error: str,
                            refine_seed: int | None, verbose: bool, chunk: int = 64) -> None:
        """The constant ladder as a post-fit pass under the POOL BOUND: candidates in score order, in
        chunks; a candidate is re-spelled only while its ``ladder_floor`` (MDL cut to nothing, fit
        improved by ``max_decades``) reaches the running best, which tightens as variants land. A
        variant that ties its parent replaces it in ``results``; a strict improvement stands beside."""
        ladder = self.constant_ladder
        if ladder is None:
            return
        weights = self.ranking.effective_weights
        entries = sorted((r for r in results if np.isfinite(float(r.get('score', np.nan)))), key=lambda r: float(r['score']))
        if not entries:
            return
        running_best = float(entries[0]['score'])
        replaced: list[Any] = []
        for start in range(0, len(entries), max(1, int(chunk))):
            batch = [e for e in entries[start:start + chunk] if ladder_floor(e, weights, ladder, score_row) <= running_best]
            if not batch:
                continue
            jobs = []
            for entry in batch:
                job = {key: entry.get(key) for key in _RESPELL_PARENT_KEYS}
                job.update({
                    'fits': [(np.asarray(c, dtype=float), (np.asarray(cov) if cov is not None and getattr(cov, 'size', None) else None), float(loss))
                             for c, cov, loss in entry['fits']],
                    'refine_scope': self.refiner_scope,
                    'n_variables': self.n_variables,
                    'n_restarts': self.n_restarts,
                    'method': self.refiner_method,
                    'p0_noise': self.refiner_p0_noise,
                    'p0_noise_kwargs': copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                    'constant_ladder': ladder,
                    'converge_error': converge_error,
                    'numpy_errors': self.numpy_errors,
                    'y_variance': gs.y_variance,
                    'ranking_weights': weights,
                    'seed': _candidate_refine_seed(refine_seed, entry.get('raw_beam') or entry.get('expression', [])),
                })
                jobs.append(job)
            outcomes = self._run_ordered_jobs(jobs, _respell_candidate_worker, gs, desc="Re-spelling Constants", verbose=verbose)
            for entry, outcome in zip(batch, outcomes):
                child = outcome[0] if outcome else None
                if child is None:
                    continue
                child_entry = self._create_result_entry(payload=child, input_dim=input_dim)
                if child_entry is None:
                    continue
                if child.get('replaces_parent'):
                    replaced.append(entry)
                results.append(child_entry)
                running_best = min(running_best, float(child_entry['score']))
        if replaced:
            gone = {id(e) for e in replaced}
            results[:] = [r for r in results if id(r) not in gone]

    def _fit_refine(
            self,
            gen_state: "Generation",
            *,
            converge_error: Literal['raise', 'ignore', 'print'] = 'ignore',
            refine_seed: int | None = None,
            verbose: bool = False,
            allow_empty: bool = False) -> "_RefineOutcome":
        """CPU refinement phase of :meth:`fit`: build jobs, fit constants, prune, compile.

        Operates on LOCAL state and returns a FitResult; writes NOTHING to ``self``. With
        ``prune_constant_budget == 0`` it touches no GPU at all. NOTE: not yet overlap-SAFE on its
        own -- ``_RefinementContext`` sets a module global for fork-COW that two concurrent calls
        would race; that global is removed (per-job X/y on the persistent pool) in a later step.
        """
        gs = gen_state
        results: list[Result] = []
        refinement_time = 0.0
        input_dim = gs.X.shape[1]

        beams = [self.flash_ansr_model.tokenizer.extract_expression_from_beam(raw_beam)[0] for raw_beam in gs.raw_beams]

        # The candidate as the model STATED it (owner ruling 2026-09-12): an <ieee754> span becomes
        # the LITERAL it spells, a bare '<constant>' stays a placeholder. So the expression itself
        # says which sites are the model's prediction and which are the refiner's, and there is no
        # side-channel of values to keep aligned -- `refinement_slots` under refine_scope='fittable'
        # keeps a spelled exponent verbatim and frees a spelled coefficient, by construction.
        realized_beams = [self.flash_ansr_model._realize_ieee754_spans(beam) for beam in beams]
        raw_beams_decoded = [self.tokenizer.decode_expression(raw_beam) for raw_beam in gs.raw_beams]
        beams_decoded = [(self._ensure_explicit_dialect(realized) or []) if realized is not None else []
                         for realized in realized_beams]

        refinement_jobs: list[dict[str, Any]] = []
        beam_iterator = zip(gs.raw_beams, raw_beams_decoded, beams, beams_decoded, gs.log_probs)
        for beam_position, (raw_beam, raw_beam_decoded, beam, beam_decoded, log_prob) in enumerate(beam_iterator):
            if not beam_decoded or not self.simplipy_engine.is_valid(beam_decoded):
                continue

            # `refiner_typed_spans` decides what happens to a literal the model predicted in a TYPED
            # position (a pow exponent, a rootn index -- the ones whose value fixes the DOMAIN).
            # 'freeze' is what refine_scope='fittable' already does to a spelled literal, so the
            # policy only has to ACT for the control arm: 'refine' thaws them back into slots, which
            # is what the pipeline did implicitly while post-processing was erasing the spelling.
            try:
                typed_sites = typed_literal_sites(beam_decoded, self.simplipy_engine)
            except Exception:  # noqa: BLE001 -- an untypeable candidate keeps every literal as written
                typed_sites = []
            typed_frozen = len(typed_sites)
            if self.refiner_typed_spans == 'refine' and typed_sites:
                beam_decoded, _thawed = thaw_typed_literals(beam_decoded, self.simplipy_engine)
                typed_frozen = 0

            constant_count = self._count_constants(beam_decoded)

            # Verbatim init: one seed per REFINEMENT SLOT in order of appearance, read off the
            # expression itself. A slot is either a literal the model spelled (seed = that value) or
            # a '<constant>' it masked (no seed -- the model declined to predict it). The two cannot
            # be mixed in one p0 vector, so a candidate carrying ANY masked slot takes the ordinary
            # multi-restart search, which is exactly what such a slot asks for. Under
            # `<mask_fittable>` this is the common case and the split is clean: the typed literals
            # are spelled (and not slots), the fittable ones are masked (and are).
            slot_tokens = [beam_decoded[index] for index
                           in refinement_slots(beam_decoded, self.simplipy_engine, self.refiner_scope)]
            p0: list[float] | None = None
            if slot_tokens and not any(token == '<constant>' for token in slot_tokens):
                p0 = [literal_value(token) for token in slot_tokens]

            job: dict[str, Any] = {
                'raw_beam': raw_beam,
                'raw_beam_decoded': raw_beam_decoded,
                'beam': beam,
                'expression': beam_decoded,
                'log_prob': log_prob,
                'constant_count': constant_count,
                'p0': p0,
                'pruned_variant': False,
                'typed_spans': self.refiner_typed_spans,
                'typed_frozen': typed_frozen,
                'n_variables': self.n_variables,
                'n_restarts': self.n_restarts,
                'method': self.refiner_method,
                'p0_noise': self.refiner_p0_noise,
                'p0_noise_kwargs': copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                'refine_scope': self.refiner_scope,
                # under the pool bound the ladder runs AFTER the fits, on the candidates that can still win
                'constant_ladder': None if self._ladder_bounded else self.constant_ladder,
                'converge_error': converge_error,
                'numpy_errors': self.numpy_errors,
                'y_variance': gs.y_variance,
                'ranking_weights': self.ranking.effective_weights,
                'complexity': gs.complexity,
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
                        job['X'] = gs.X
                        job['y'] = gs.y
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
                                self._append_result_entries(results, result, input_dim)
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
                                self._append_result_entries(results, result, input_dim)
                else:
                    for job in _iterate_with_progress(
                        jobs,
                        total=len(jobs),
                        verbose=verbose,
                        desc="Fitting Constants",
                    ):
                        serial_payload = job.copy()
                        serial_payload.update({'X': gs.X, 'y': gs.y, 'simplipy_engine': self.simplipy_engine})
                        result, warning_msg = _refine_candidate_worker(serial_payload)
                        if warning_msg and converge_error == 'print':
                            print(warning_msg)
                        if result is not None:
                            self._append_result_entries(results, result, input_dim)

            with _RefinementContext(self.simplipy_engine, gs.X, gs.y):
                _t_ref = time.time()
                _run_refinement_jobs(refinement_jobs)
                if self._ladder_bounded:
                    self._run_bounded_ladder(results, gs, input_dim=input_dim, converge_error=converge_error,
                                             refine_seed=refine_seed, verbose=verbose)
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

                            try:
                                simplified_variant = simplify_and_mask(self.simplipy_engine, list(variant))
                            except NonFiniteExpressionError:
                                # Pruning a constant can make a denominator vanish; the variant is
                                # then unusable. Dropped like the is_valid rejection below it.
                                record_non_finite_drop()
                                continue
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
                            memory=gs.memory,
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
                                'refine_scope': self.refiner_scope,
                                'constant_ladder': self.constant_ladder,
                                'converge_error': converge_error,
                                'numpy_errors': self.numpy_errors,
                                'y_variance': gs.y_variance,
                                'ranking_weights': self.ranking.effective_weights,
                                'complexity': gs.complexity,
                            }
                            pruning_jobs.append(pruning_job)

                    _t_prune_ref = time.time()
                    _run_refinement_jobs(pruning_jobs)
                    refinement_time += time.time() - _t_prune_ref

        results = self._dedup_respelled(results)
        if not results and not allow_empty:
            raise ConvergenceError("The optimization did not converge for any beam")
        sorted_results = self._order_results(results)

        return _RefineOutcome(
            results=sorted_results,
            refinement_time=refinement_time,
            generation_time=gs.generation_time,
            variable_mapping=gs.variable_mapping,
            input_dim=input_dim,
        )

    def _order_results(self, results: list[Any]) -> list[Any]:
        """Score and order the refined result rows under the estimator's ranking -- through the
        library's one ordering rule (:func:`flash_ansr.scoring.order_rows`), so
        :meth:`FitResult.rerank` reproduces this order offline bit for bit."""
        return order_rows(results, self.ranking)

    def predict(self, X: np.ndarray | torch.Tensor | pd.DataFrame, rank: int = 0) -> np.ndarray:
        """Evaluate the last fit's candidate at ``rank`` (0 = the answer) on ``X`` -> ``(n_points, 1)``.

        Sugar for ``self.result_.predict(X, rank)``. Raises ``ValueError`` before the first fit or
        when the last fit produced no candidate.
        """
        return self._require_result().predict(self._truncate_input(X), rank)

    def _score_outliers(self, X: Any, y: Any) -> np.ndarray:
        """Per-point outlier probability from the trained outlier head.

        Parameters
        ----------
        X : array-like
            Support-set features, ``(n_points, n_variables)``.
        y : array-like
            Support-set targets, ``(n_points,)``.

        Returns
        -------
        np.ndarray
            One probability in ``[0, 1]`` per point, in input order.

        Raises
        ------
        CapabilityUnavailable
            If this checkpoint was built without an outlier head.

        Notes
        -----
        The head reads the DATA-SET ENCODER only: a score conditions on ``(X, y)`` and never on any
        expression. It asks "could a function have produced this point set", not "does this point
        fit that formula". The published AUROC 0.9888 is POOLED and in-distribution; per-problem
        behaviour is much weaker (a single lone outlier scores a median P around 0.42), and the head
        degrades above roughly 10% contamination -- the ceiling it was trained under.

        Examples
        --------
        >>> p = ansr.score_outliers(X, y)          # doctest: +SKIP
        >>> suspicious = np.argsort(p)[-3:]        # doctest: +SKIP
        """
        return score_outliers(self, X, y)

    def _predict_constants(self, X: Any, y: Any, expression: Sequence[str] | str, *,
                          conditioned: bool = True, n_samples: int = DEFAULT_SAMPLES,
                          temperature: float = 1.0,
                          seed: int | None = None) -> list[ValueDistribution]:
        """Fill the ``'<constant>'`` slots of ``expression`` with the model's own predictions.

        The trained ``<predict_constants>`` block: the harness force-feeds the block openers and the
        model emits the 8 IEEE-754 bytes of each float64, exactly as in training.

        Parameters
        ----------
        X : array-like
            Support-set features the constants should explain.
        y : array-like
            Support-set targets.
        expression : sequence of str or str
            Prefix tokens or an infix string, with ``'<constant>'`` at each slot to fill.
            ``v1..vn`` variable names are mapped to ``x1..xN`` at the boundary.
        conditioned : bool, optional
            ``False`` selects the trained unconditioned mode: the constants the model finds typical
            for this SHAPE, with the data ignored. ``X``/``y`` may then be ``None``.

        Returns
        -------
        list[ValueDistribution]
            One DISTRIBUTION per slot, in slot order -- not a float. Every byte is a softmax
            sample, so a single decode is a single draw; quote ``.median`` with ``.q05``/``.q95``,
            and read ``.agreement`` to see whether any single number represents the draws at all.

        Raises
        ------
        CapabilityUnavailable
            If this checkpoint lacks the ``<predict_constants>`` block or the ieee754 vocabulary.
        ValueError
            If the expression has no ``'<constant>'`` slots or names an unknown token.

        Examples
        --------
        >>> pred = ansr.predict_constants(X, y, ['+', '*', '<constant>', 'x1', '<constant>'])
        ... # doctest: +SKIP
        >>> pred.values                                       # doctest: +SKIP
        [2.5, 1.0]

        See Also
        --------
        fit : refines constants numerically; this verb predicts them directly.
        """
        return predict_constants(self, X, y, expression, conditioned=conditioned,
                                 n_samples=n_samples, temperature=temperature, seed=seed)

    def _predict_y(self, X: Any, y: Any, x_query: Any, *,
                  expression: Sequence[str] | str | None = None, conditioned: bool = True,
                  n_samples: int = DEFAULT_SAMPLES, temperature: float = 1.0,
                  seed: int | None = None) -> list[ValueDistribution]:
        """Predict the target at held-out points using the trained ``<predict_y>`` block.

        Training writes the block in two placements and both are reachable. Without ``expression``
        the model interpolates the point set it was given; with one, the expression is in scope.
        ``expression`` plus ``conditioned=False`` is FUNCTION EVALUATION -- the data is replaced by
        the learned ``null_memory``, so the expression and the query point are all there is.
        The query coordinates ride the numeric channel exactly as training wrote them.

        Parameters
        ----------
        X : array-like or None
            Support-set features the model conditions on; ``None`` only if ``conditioned=False``.
        y : array-like or None
            Support-set targets; ``None`` only if ``conditioned=False``.
        x_query : array-like
            Query coordinates, ``(n_queries, n_variables)`` or one ``(n_variables,)`` point.
        expression : sequence of str or str, optional
            Prefix tokens or an infix string. Places the block AFTER the expression.
        conditioned : bool, optional
            ``False`` selects the trained unconditioned mode, by default ``True``.

        Returns
        -------
        list[ValueDistribution]
            One distribution per query point. A decode is a draw, not the model's answer.

        Raises
        ------
        CapabilityUnavailable
            If this checkpoint lacks the ``<predict_y>`` block.
        ValueError
            If a query point's dimensionality does not match ``X``.

        See Also
        --------
        predict : evaluates a FITTED expression; this verb never forms one.
        """
        return predict_y(self, X, y, x_query, expression=expression, conditioned=conditioned,
                         n_samples=n_samples, temperature=temperature, seed=seed)

    def _predict_complexity(self, X: Any, y: Any, *, conditioned: bool = True,
                           n_samples: int = DEFAULT_SAMPLES, temperature: float = 1.0,
                           seed: int | None = None) -> ComplexityDistribution:
        """Ask the model how complex it thinks the generating expression is.

        Uses the trained hypothesis circumstance: the harness utters ``<hypothesize>`` and
        everything after it is the model's own.

        Parameters
        ----------
        X : array-like
            Support-set features.
        y : array-like
            Support-set targets.

        Returns
        -------
        ComplexityDistribution
            Draws in simplipy complexity units -- the unit ``fit(complexity=...)`` consumes, NOT a
            token count. ``.self_initiated_fraction`` reports how often the model would have opened
            the block unprompted.

        Raises
        ------
        CapabilityUnavailable
            If this checkpoint lacks the ``<hypothesize>`` or ``<complexity>`` tokens.
        """
        return predict_complexity(self, X, y, conditioned=conditioned, n_samples=n_samples,
                                  temperature=temperature, seed=seed)

    def _skeleton_mu(self, skeleton: Sequence[str] | None) -> float | None:
        """simplipy complexity of a skeleton, or None when the engine cannot price it.

        Never raises: a candidate whose dialect the engine will not parse still has a valid
        expression and a valid fit, and losing it over a reporting field would be absurd.
        """
        if not skeleton:
            return None
        try:
            return float(self.simplipy_engine.complexity(list(skeleton)))
        except Exception:
            return None

    def get_expression(self, rank: int = 0, *, return_prefix: bool = False, precision: int | None = None,
                       map_variables: bool = True) -> list[str] | str:
        """The last fit's candidate at ``rank`` with its constants substituted.

        Sugar for ``self.result_.get_expression(...)``: an infix string (default) or the prefix
        tokens (``return_prefix=True``); ``precision`` rounds the constants for display (``None`` =
        the round-trip-exact ``repr``); ``map_variables`` applies the fit's variable names.
        """
        return self._require_result().get_expression(rank, return_prefix=return_prefix, precision=precision, map_variables=map_variables)

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
