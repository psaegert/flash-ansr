"""The estimator's policy, in three config objects (plus the generation config in
:mod:`flash_ansr.utils.generation` and the :class:`~flash_ansr.scoring.RankingConfig`).

The rule (owner design 2026-09-14): the constructor describes the ESTIMATOR, a call to ``fit``
describes the PROBLEM and the QUESTION. Anything that would stay the same across a thousand
problems is policy and lives here; anything that changes with the data at hand is an argument of
``fit``. Each object validates at construction so a typo cannot lie dormant, and each accepts a
plain mapping (``RefineConfig.from_mapping``) so a YAML config can spell it.
"""
from __future__ import annotations

import copy
import numbers
import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Mapping

from flash_ansr.refine import (RefineScope, TypedSpanPolicy, DEFAULT_REFINE_SCOPE, DEFAULT_TYPED_SPAN_POLICY,
                               TYPED_SPAN_POLICIES)
from flash_ansr.scoring import RankingConfig, resolve_ranking
from flash_ansr.spelling import ConstantLadderConfig

RefinerMethod = Literal['curve_fit_lm', 'minimize_bfgs', 'minimize_lbfgsb', 'minimize_neldermead',
                        'minimize_powell', 'least_squares_trf', 'least_squares_dogbox']
REFINER_METHODS = ('curve_fit_lm', 'minimize_bfgs', 'minimize_lbfgsb', 'minimize_neldermead',
                   'minimize_powell', 'least_squares_trf', 'least_squares_dogbox')
P0_NOISES = ('uniform', 'normal', 'cauchy', 'magspan')
NUMPY_ERROR_POLICIES = ('ignore', 'warn', 'raise', 'call', 'print', 'log')


@dataclass(frozen=True)
class RefineConfig:
    """How a generated candidate's constants are fitted, re-spelled and pruned.

    ``method`` / ``n_restarts`` / ``p0_noise`` / ``p0_noise_kwargs`` drive the optimizer;
    ``scope`` says which literals of a candidate the refiner may move (``'fittable'``: the model
    predicts the typed literals -- exponents, root indices -- and the refiner fits the rest);
    ``typed_spans`` what happens to a PREDICTED literal in a typed position; ``constant_ladder``
    the post-fit re-spelling (``True`` = defaults, ``False``/``None`` = off, a mapping overrides);
    ``prune_constant_budget`` how many top candidates (count, or a fraction in (0, 1]) get
    constant-pruning variants; ``numpy_errors`` the floating-point error policy in force during
    refinement.
    """

    method: RefinerMethod = 'curve_fit_lm'
    n_restarts: int = 8
    p0_noise: Literal['uniform', 'normal', 'cauchy', 'magspan'] | None = 'normal'
    p0_noise_kwargs: Mapping[str, Any] | None = field(default_factory=lambda: {'loc': 0.0, 'scale': 5.0})
    scope: RefineScope = DEFAULT_REFINE_SCOPE
    typed_spans: TypedSpanPolicy = DEFAULT_TYPED_SPAN_POLICY
    constant_ladder: Mapping[str, Any] | bool | None = True
    prune_constant_budget: float = 0.0
    numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore'

    def __post_init__(self) -> None:
        if self.method not in REFINER_METHODS:
            raise ValueError(f"RefineConfig.method must be one of {REFINER_METHODS}; got {self.method!r}")
        if not isinstance(self.n_restarts, numbers.Integral) or int(self.n_restarts) < 1:
            raise ValueError(f"RefineConfig.n_restarts must be a positive integer; got {self.n_restarts!r}")
        if self.p0_noise is not None and self.p0_noise not in P0_NOISES:
            raise ValueError(f"RefineConfig.p0_noise must be one of {P0_NOISES} or None; got {self.p0_noise!r}")
        if self.scope not in ('placeholders', 'fittable', 'all'):
            raise ValueError(f"RefineConfig.scope must be 'placeholders', 'fittable' or 'all'; got {self.scope!r}")
        if self.typed_spans not in TYPED_SPAN_POLICIES:
            raise ValueError(f"RefineConfig.typed_spans must be one of {TYPED_SPAN_POLICIES}; got {self.typed_spans!r}")
        if self.numpy_errors is not None and self.numpy_errors not in NUMPY_ERROR_POLICIES:
            raise ValueError(f"RefineConfig.numpy_errors must be one of {NUMPY_ERROR_POLICIES} or None; got {self.numpy_errors!r}")
        budget = float(self.prune_constant_budget)
        if budget < 0.0:
            raise ValueError(f"RefineConfig.prune_constant_budget must be >= 0; got {self.prune_constant_budget!r}")
        object.__setattr__(self, 'prune_constant_budget', budget)
        object.__setattr__(self, 'n_restarts', int(self.n_restarts))
        # `scope='all'` frees EVERY literal, typed ones included, so it already does what 'refine'
        # asks and leaves nothing for a freeze to keep or a duplicate to thaw. Coerced rather than
        # refused: asking for the widest scope should not also require restating the policy it implies.
        if self.scope == 'all':
            object.__setattr__(self, 'typed_spans', 'refine')
        object.__setattr__(self, 'p0_noise_kwargs',
                           None if self.p0_noise_kwargs is None else dict(copy.deepcopy(dict(self.p0_noise_kwargs))))
        # Validated once, here: a malformed ladder mapping must not wait for the first fit.
        ConstantLadderConfig.from_mapping(self.constant_ladder)

    @property
    def ladder(self) -> ConstantLadderConfig | None:
        """The resolved constant-ladder configuration (``None`` = off)."""
        return ConstantLadderConfig.from_mapping(self.constant_ladder)

    @classmethod
    def from_mapping(cls, payload: "RefineConfig | Mapping[str, Any] | None") -> "RefineConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        payload = dict(payload)
        unknown = sorted(set(payload) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown RefineConfig keys: {unknown}; known: {[f.name for f in fields(cls)]}")
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out['p0_noise_kwargs'] = None if self.p0_noise_kwargs is None else dict(self.p0_noise_kwargs)
        out['constant_ladder'] = (dict(self.constant_ladder) if isinstance(self.constant_ladder, Mapping)
                                  else self.constant_ladder)
        return out


@dataclass(frozen=True)
class ComputeConfig:
    """Where and how wide the estimator runs: the torch ``device`` of the transformer, the refiner's
    worker pool (``workers``: ``None`` = every CPU core, ``0`` = no multiprocessing) and whether one
    persistent pool is forked BEFORE any CUDA initialization and reused across calls
    (``persistent_pool``; the structural mitigation for the fork-after-CUDA deadlock family)."""

    device: str = 'cpu'
    workers: int | None = None
    persistent_pool: bool = False

    def __post_init__(self) -> None:
        if self.workers is not None and (not isinstance(self.workers, numbers.Integral) or int(self.workers) < 0):
            raise TypeError(f"ComputeConfig.workers must be a non-negative integer or None; got {self.workers!r}")
        object.__setattr__(self, 'device', str(self.device))
        object.__setattr__(self, 'workers', None if self.workers is None else int(self.workers))
        object.__setattr__(self, 'persistent_pool', bool(self.persistent_pool))

    @property
    def resolved_workers(self) -> int:
        return max(1, os.cpu_count() or 1) if self.workers is None else self.workers

    @classmethod
    def from_mapping(cls, payload: "ComputeConfig | Mapping[str, Any] | None") -> "ComputeConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        payload = dict(payload)
        unknown = sorted(set(payload) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown ComputeConfig keys: {unknown}; known: {[f.name for f in fields(cls)]}")
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def ranking_from(payload: "RankingConfig | Mapping[str, Any] | str | None") -> RankingConfig:
    """A :class:`RankingConfig` from the config object itself, a mode name (``'mdl'``), a mapping
    (``{'mode': 'mdl', 'mdl_strength': 1e-2}`` -- the spelling ``RankingConfig.as_dict`` writes,
    with ``weights`` / ``metrics`` / ``tie_break`` for the other modes) or ``None`` (the default)."""
    if payload is None:
        return resolve_ranking('mdl')
    if isinstance(payload, RankingConfig):
        return payload
    if isinstance(payload, str):
        return resolve_ranking(payload)
    return RankingConfig.from_dict(payload)


__all__ = ['RefineConfig', 'ComputeConfig', 'RankingConfig', 'ranking_from', 'REFINER_METHODS']
