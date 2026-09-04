"""Canonical candidate-scoring primitives (single source of truth).

These are owned by flash-ansr because the product needs them at *inference* time to score and
rank decode candidates (``flash_ansr.py`` / ``results.py``). They are
also consumed by the comparison baselines and, after the repo split, by ``srbf`` via the
public API. This module collapses the formerly-triplicated
``_compute_fvu`` / ``_normalize_variance`` / ``_score_from_fvu`` copies
(``flash_ansr.py`` + the srbf comparison baselines) onto one definition.

Note on FVU: this is the *scalar-loss* form. The refiner supplies an already-reduced residual
``loss``, so :func:`compute_fvu` divides by the (epsilon-floored) target variance. The *array* form
``fvu(y_true, y_pred)`` lives in the ``srbf`` evaluation metrics and is a
deliberately separate signature for a different call site: the two share the FVU *definition*
(residual / total variance) but are not interchangeable functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from flash_ansr.utils.ieee754 import IEEE754_END_TOKEN, BYTE_TOKENS

#: Membership set for the expanded-span content alphabet. A frozenset, not the source tuple:
#: is_constant_token runs per token on the scoring hot path, where a linear scan costs 16
#: comparisons today and 256 once the alphabet becomes bytes.
_BYTE_TOKEN_SET = frozenset(BYTE_TOKENS)

#: Floor used to keep variance and FVU strictly positive before a division / log.
FLOAT64_EPS: float = float(np.finfo(np.float64).eps)


def normalize_variance(variance: float) -> float:
    """DEPRECATED. Floor ``variance`` at an absolute :data:`FLOAT64_EPS`.

    No longer used by :func:`compute_fvu`: an absolute floor breaks FVU scale-invariance and, for
    tiny-magnitude targets (``var(y) < FLOAT64_EPS``), spuriously rates any candidate as near-perfect
    (the constant-candidate mis-selection bug). Retained only for backward-compatible imports; do not
    use it to normalize an FVU.
    """
    if not np.isfinite(variance):
        return FLOAT64_EPS
    return max(float(variance), FLOAT64_EPS)


def compute_fvu(loss: float, sample_count: int, variance: float) -> float:
    """Scale-invariant Fraction of Variance Unexplained = ``loss / variance`` (== SS_res / SS_tot).

    Mirrors :func:`flash_ansr.eval.metrics.numeric.fvu` on the reduced ``(loss, variance)`` inputs so
    the selection-score FVU agrees with the evaluation FVU. There is deliberately **no absolute
    variance floor**: an absolute floor breaks scale-invariance and, for tiny-magnitude targets
    (``var(y) < FLOAT64_EPS``), would spuriously rate any candidate as near-perfect.

    Edge cases (matching ``numeric.fvu``'s ``safe_divide`` semantics):
    - ``sample_count <= 1``: variance undefined -> return the raw ``loss``.
    - non-finite ``loss`` or ``variance`` -> ``+inf`` (worst; a diverged/invalid fit).
    - ``variance == 0`` (constant/degenerate target): perfect iff residual is exactly zero, else ``+inf``.
      (The signature lacks the target scale, so only an EXACT-zero residual can be deemed perfect; any
      absolute near-zero tolerance would reintroduce the scale-dependence this fix removes.)
    - otherwise -> ``loss / variance``.
    """
    if sample_count <= 1:
        return float(loss)
    loss = float(loss)
    variance = float(variance)
    if not np.isfinite(loss) or not np.isfinite(variance):
        return float('inf')
    if variance <= 0.0:
        return 0.0 if loss == 0.0 else float('inf')
    return loss / variance


#: simplipy reports complexity in MILLI-bits; `mdl_penalty` is specified per BIT so that its
#: calibrated value (4.5e-3) is legible next to `node_penalty` (0.05) in a config file. 4.5e-6 per
#: milli-bit three lines from 0.05 is a typo waiting to happen. The conversion happens once, here.
MILLIBITS_PER_BIT = 1000.0


def score_from_fvu(
        fvu: float,
        n_nodes: int,
        constant_count: int,
        log_prob: float | None,
        node_penalty: float,
        constants_penalty: float,
        likelihood_penalty: float,
        mdl: float | None = None,
        mdl_penalty: float = 0.0) -> float:
    """Parsimony-penalised selection score ``log10(FVU) + structural penalties`` (lower is better).

    A genuine perfect fit (``fvu == 0``) is floored at :data:`FLOAT64_EPS` so it gets the best FINITE
    score (not ``-inf``). A non-finite or negative FVU is a diverged/invalid candidate and maps to the
    WORST score (``+inf``), NOT the best -- otherwise an invalid candidate would out-rank real fits
    (a ranking inversion). ``log_prob`` contributes ``likelihood_penalty * (-log_prob)`` when finite.
    """
    if fvu == 0.0:
        safe_fvu = FLOAT64_EPS
    elif not np.isfinite(fvu) or fvu < 0.0:
        return float('inf')
    else:
        safe_fvu = max(float(fvu), FLOAT64_EPS)

    likelihood_term = 0.0
    if log_prob is not None and np.isfinite(log_prob):
        likelihood_term = likelihood_penalty * (-float(log_prob))

    # The MDL addend. `mdl` arrives in MILLI-bits (simplipy's native unit, and what the ledger
    # stores); `mdl_penalty` is per BIT. The conversion is explicit and happens exactly here --
    # multiplying the two directly would be wrong by a factor of 1000.
    #
    # UNPRICEABLE CANDIDATES. When `mdl_penalty` is 0 -- the default, and every pre-existing config
    # -- `mdl` is not part of the ranking at all, so a missing price is irrelevant and the term is
    # exactly 0.0: the score stays character-for-character what 0.13.0 produced.
    #
    # When `mdl_penalty` is LIVE, a missing price must never contribute 0. At the calibrated
    # strength a typical candidate pays ~0.63 decades of FVU, so a free pass is not a small mercy --
    # it is a ~4x FVU advantage that would float every unpriceable candidate to the top of the
    # ranking. Such a row maps to +inf instead: it sorts below every priced candidate, and above a
    # diverged fit (whose score is nan), which is the correct relative standing for "fitted, but
    # cannot be judged on the declared criterion". The aggregate case -- NOTHING priceable, so the
    # criterion produced no ranking at all -- is caught by the caller, not here; see
    # `_compile_results_pure`.
    mdl_term = 0.0
    if mdl_penalty != 0.0:
        if mdl is None or not np.isfinite(mdl):
            return float('inf')
        mdl_term = mdl_penalty * (float(mdl) / MILLIBITS_PER_BIT)

    return float(np.log10(safe_fvu)
                 + node_penalty * n_nodes
                 + constants_penalty * max(int(constant_count), 0)
                 + likelihood_term
                 + mdl_term)


def is_constant_token(token: str) -> bool:
    """Return ``True`` if ``token`` denotes a constant in a prefix expression.

    Recognises the ``<constant>`` placeholder, generated ``C_i`` symbols, a small set of named
    literals (signed/unsigned ``0``/``1``, ``np.pi``, ``np.e``, the float specials), any token
    that parses as a Python ``float``, and the v24 ``ieee754_mixed`` constant forms: the compact
    ``<float>`` token and the ``<ieee754>`` span OPENING tag. The byte tokens
    ``<h0>``..``<hf>`` and the closing ``</ieee754>`` tag deliberately do NOT count, so a whole
    10-token expanded span contributes exactly ONE constant to any per-token sum (the count
    that feeds ``constants_penalty`` in :func:`score_from_fvu` must see one per span, not 10).
    """
    if token in ('<constant>', '<float>', '<ieee754>'):
        return True
    if token == IEEE754_END_TOKEN or token in _BYTE_TOKEN_SET:
        return False
    if token.startswith('C_') and token[2:].isdigit():
        return True
    if token in {'0', '1', '(-1)', 'np.pi', 'np.e', 'float("inf")', 'float("-inf")', 'float("nan")'}:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def count_constants(expression: Iterable[str] | None) -> int:
    """Count the constant tokens in ``expression`` (``None`` -> ``0``)."""
    if expression is None:
        return 0
    return sum(1 for token in expression if is_constant_token(str(token)))


# --- ranking modes (RANKING_SPEC.md) --------------------------------------------------------------

#: Every metric a ranking may declare, and how to read it off a result row. All are LOWER IS BETTER
#: -- the front construction and the tie-break both assume it, so a "higher is better" metric would
#: have to enter negated. `fvu` is required in every mode: it is the only term that measures fit, and
#: a ranking without it would order candidates purely on shape.
#:
#: `mdl` is priced on the REALIZED expression; `n_nodes`, `n_constant_placeholders` and
#: `n_typed_literals` are counted on the EMITTED one. That is deliberate (RANKING_SPEC.md section 3):
#: the two spellings answer different questions, and mixing them silently would be the bug.
RANKING_METRICS: dict[str, Any] = {
    'fvu': lambda r: r.get('fvu'),
    # BITS, converted from the milli-bit price the row stores, so every weight in `weighted` mode
    # is per natural unit (a node, a constant, a bit, a nat).
    'mdl': lambda r: (None if r.get('mdl') is None else float(r['mdl']) / MILLIBITS_PER_BIT),
    'n_nodes': lambda r: len(r.get('expression', []) or []),
    # The legacy total (placeholders, spans and typed literals alike -- count_constants), read off
    # the row when the worker already counted it.
    'n_constants': lambda r: (r['constant_count'] if r.get('constant_count') is not None
                              else count_constants(r.get('expression'))),
    'n_constant_placeholders': lambda r: sum(1 for t in (r.get('expression') or []) if t == '<constant>'),
    'n_typed_literals': lambda r: sum(1 for t in (r.get('expression') or [])
                                      if t != '<constant>' and _parses_as_float(str(t))),
    'neg_log_prob': lambda r: (None if r.get('log_prob') is None else -float(r['log_prob'])),
}

#: Above this many candidates the pairwise front is refused rather than silently allocating: the
#: domination matrix is 2 * k^2 bytes (8 MB at k=2048, 134 MB here). The doctrine arm draws 1024.
ND_MAX_CANDIDATES = 8192

#: `pareto_rank` on a row that was ranked by the SCALAR mode. Not 0 -- 0 is the best front, and a
#: scalar row must never be mistaken for a front-0 member by a downstream consumer.
PARETO_RANK_NOT_COMPUTED = -1


def _parses_as_float(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


class RankingError(ValueError):
    """A ranking could not be produced: the declared criterion ordered nothing."""


#: Mode 1's engineered strength, per BIT of realized-expression description length. Calibrated
#: against `node_penalty = 0.05` on the reference population (RANKING_SPEC.md section 4): a
#: typical candidate's ~140 bits then cost ~0.63 decades of FVU. Not a tuning surface: a run that
#: wants a different weight on `mdl` says so in `weighted` mode, where the number is visible.
MDL_STRENGTH_DEFAULT = 4.5e-3

RANKING_MODES = ('mdl', 'weighted', 'pareto')

#: Metrics a `weighted` ranking may put a weight on: everything in the registry except `fvu`,
#: which is the base term of the score (log10) and not a weighted addend.
WEIGHTABLE_METRICS = tuple(m for m in RANKING_METRICS if m != 'fvu')


@dataclass(frozen=True)
class RankingConfig:
    """The resolved ranking actually in force: one of the three modes with ITS knobs, nothing
    dormant. Built only through :func:`resolve_ranking`, which validates."""

    mode: str
    mdl_strength: float | None = None                 # 'mdl' only
    weights: Mapping[str, float] = field(default_factory=dict)   # 'weighted' only
    metrics: tuple[str, ...] = ()                     # 'pareto' only
    tie_break: str | None = None                      # 'pareto' only

    @property
    def effective_weights(self) -> dict[str, float]:
        """The scalar addends this ranking applies (empty for `pareto`, whose score is nan)."""
        if self.mode == 'mdl':
            return {'mdl': float(self.mdl_strength)}
        if self.mode == 'weighted':
            return {k: float(v) for k, v in self.weights.items()}
        return {}

    def as_dict(self) -> dict[str, Any]:
        """Plain, picklable, YAML-able record of the resolved values -- what provenance stores."""
        out: dict[str, Any] = {'mode': self.mode}
        if self.mode == 'mdl':
            out['mdl_strength'] = float(self.mdl_strength)
        elif self.mode == 'weighted':
            out['weights'] = {k: float(v) for k, v in sorted(self.weights.items())}
        else:
            out['metrics'] = list(self.metrics)
            out['tie_break'] = self.tie_break
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RankingConfig":
        payload = dict(payload)
        mode = payload.pop('mode', None)
        if mode is None:
            raise ValueError("a ranking record must carry 'mode'")
        cfg = resolve_ranking(
            mode,
            mdl_strength=payload.pop('mdl_strength', None),
            weights=payload.pop('weights', None),
            metrics=payload.pop('metrics', None),
            tie_break=payload.pop('tie_break', None),
        )
        if payload:
            raise ValueError(f"unknown keys in ranking record: {sorted(payload)}")
        return cfg


def resolve_ranking(
        mode: str = 'mdl',
        *,
        mdl_strength: float | None = None,
        weights: Mapping[str, float] | None = None,
        metrics: Sequence[str] | None = None,
        tie_break: str | None = None) -> RankingConfig:
    """Validate a ranking request and fill the defaults OF ITS MODE. Raises on anything unrecognised.

    Every knob belongs to exactly one mode, and passing a knob to another mode raises rather than
    being ignored: a `ranking_weights` typed next to `ranking_mode='mdl'` would otherwise lie
    dormant until someone flipped the mode, and a `ranking_metrics` typo would wait for the first
    pareto run. The user-facing spellings are ``ranking_mode``, ``mdl_strength``,
    ``ranking_weights``, ``ranking_metrics`` and ``ranking_tie_break`` (RANKING_SPEC.md section 4).
    """
    if mode not in RANKING_MODES:
        raise ValueError(f"unknown ranking_mode {mode!r}; expected one of {RANKING_MODES}")

    def _refuse(name: str, value: Any, owner: str) -> None:
        if value is not None:
            raise ValueError(
                f"{name} belongs to ranking_mode={owner!r} and was given with ranking_mode={mode!r}; "
                f"it would be silently ignored. Drop it, or switch the mode.")

    if mode == 'mdl':
        _refuse('ranking_weights', weights, 'weighted')
        _refuse('ranking_metrics', metrics, 'pareto')
        _refuse('ranking_tie_break', tie_break, 'pareto')
        strength = MDL_STRENGTH_DEFAULT if mdl_strength is None else float(mdl_strength)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError(f"mdl_strength must be a finite non-negative number of decades per bit; got {mdl_strength!r}")
        return RankingConfig(mode='mdl', mdl_strength=strength)

    if mode == 'weighted':
        _refuse('mdl_strength', mdl_strength, 'mdl')
        _refuse('ranking_metrics', metrics, 'pareto')
        _refuse('ranking_tie_break', tie_break, 'pareto')
        resolved = {str(k): float(v) for k, v in dict(weights or {}).items()}
        unknown = sorted(set(resolved) - set(WEIGHTABLE_METRICS))
        if unknown:
            raise ValueError(
                f"unknown ranking_weights {unknown}; weightable metrics: {sorted(WEIGHTABLE_METRICS)}"
                + (" ('fvu' is the base term of the score, not a weighted addend)" if 'fvu' in unknown else ""))
        for k, v in resolved.items():
            if not np.isfinite(v):
                raise ValueError(f"ranking_weights[{k!r}] must be finite; got {v!r}")
        return RankingConfig(mode='weighted', weights=resolved)

    # pareto
    _refuse('mdl_strength', mdl_strength, 'mdl')
    _refuse('ranking_weights', weights, 'weighted')
    declared = tuple(str(m) for m in (metrics if metrics is not None else ('fvu', 'n_nodes')))
    unknown = sorted(set(declared) - set(RANKING_METRICS))
    if unknown:
        raise ValueError(f"unknown ranking_metrics {unknown}; known metrics: {sorted(RANKING_METRICS)}")
    if len(set(declared)) != len(declared):
        raise ValueError(f"ranking_metrics repeats a metric: {declared}")
    if 'fvu' not in declared:
        raise ValueError(
            "ranking_metrics must contain 'fvu': it is the only term that measures FIT, and a front "
            "without it would order candidates on shape alone.")
    # The tie-break may name a metric OUTSIDE the declared set -- that is deliberate (owner ruling):
    # ordering within a front is a separate question from which axes define the front.
    tb = 'fvu' if tie_break is None else str(tie_break)
    if tb not in RANKING_METRICS:
        raise ValueError(f"unknown ranking_tie_break {tb!r}; known metrics: {sorted(RANKING_METRICS)}")
    return RankingConfig(mode='pareto', metrics=declared, tie_break=tb)


def score_row(result: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    """The scalar score of one result row under `weights` (the `effective_weights` of a resolved
    `mdl` or `weighted` ranking): ``log10(fvu) + sum_k w_k * m_k``.

    Routed through :func:`score_from_fvu` for the four terms it knows (nodes, constants,
    likelihood, mdl) so the frozen 0.13 behaviour -- the eps floor, +inf for a non-finite or
    negative fvu, +inf for an unpriced row under a live mdl weight -- is inherited rather than
    re-implemented; the two count metrics it does not know are plain addends. Adding ``0.0``
    leaves a float bit-identical, so an absent weight changes nothing.
    """
    fvu = result.get('fvu', np.nan)
    expression = result.get('expression') or []
    constant_count = result.get('constant_count')
    if constant_count is None:
        constant_count = count_constants(expression)
    base = score_from_fvu(
        float(fvu),
        len(expression),
        int(constant_count),
        result.get('log_prob'),
        float(weights.get('n_nodes', 0.0)),
        float(weights.get('n_constants', 0.0)),
        float(weights.get('neg_log_prob', 0.0)),
        result.get('mdl'),
        float(weights.get('mdl', 0.0)),
    )
    extra = 0.0
    for name in ('n_constant_placeholders', 'n_typed_literals'):
        w = float(weights.get(name, 0.0))
        if w != 0.0:
            extra += w * float(RANKING_METRICS[name](result))
    return base + extra


def objective_vector(results: list[dict[str, Any]], metrics: tuple[str, ...]) -> np.ndarray:
    """(n_candidates, n_metrics) of finite objective values; unrankable entries become +inf.

    +inf, not nan: a candidate that cannot be measured on an axis must LOSE on that axis rather than
    poison every comparison it takes part in (nan compares False both ways, which would silently
    make such a row non-dominated and float it into front 0).
    """
    out = np.full((len(results), len(metrics)), np.inf, dtype=float)
    for j, name in enumerate(metrics):
        read = RANKING_METRICS[name]
        for i, r in enumerate(results):
            v = read(r)
            if v is None:
                continue
            v = float(v)
            if np.isfinite(v):
                out[i, j] = v
    return out


def non_dominated_ranks(V: np.ndarray) -> np.ndarray:
    """Front index per row (0 = non-dominated), by repeated peeling.

    Memory is 2*k^2 bytes rather than k^2*m: the domination matrix accumulates PER OBJECTIVE instead
    of materialising an (k, k, m) comparison tensor.
    """
    n = V.shape[0]
    if n > ND_MAX_CANDIDATES:
        raise RankingError(
            f"non-dominated ranking refused for {n} candidates (limit {ND_MAX_CANDIDATES}): the "
            f"pairwise front is quadratic. Use ranking_mode='mdl' or 'weighted', or reduce `choices`.")
    if n == 0:
        return np.zeros(0, dtype=int)

    le = np.ones((n, n), dtype=bool)      # i is <= j on every objective
    lt = np.zeros((n, n), dtype=bool)     # i is <  j on at least one
    for j in range(V.shape[1]):
        col = V[:, j]
        le &= col[:, None] <= col[None, :]
        lt |= col[:, None] < col[None, :]
    dominates = le & lt                    # i dominates j

    ranks = np.full(n, -1, dtype=int)
    alive = np.ones(n, dtype=bool)
    front = 0
    while alive.any():
        # rows are already restricted to alive dominators, so a second mask would misbroadcast
        dominated_by_alive = dominates[alive, :].any(axis=0)
        current = alive & ~dominated_by_alive
        if not current.any():          # every survivor dominates another: cycle-free math says
            current = alive.copy()     # impossible, but never loop forever on a numeric surprise
        ranks[current] = front
        alive &= ~current
        front += 1
    # An all-unrankable pool peels as one front; callers that need "nothing ordered" raise on that
    # separately. Never return 0 fronts.
    return ranks
