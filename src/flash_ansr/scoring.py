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

from typing import Any, Iterable

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
    'mdl': lambda r: r.get('mdl'),
    'n_nodes': lambda r: len(r.get('expression', []) or []),
    'n_constants': lambda r: r.get('constant_count'),
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


def resolve_ranking(
        mode: str,
        weights: dict[str, float] | None,
        metrics: tuple[str, ...] | None,
        tie_break: str | None) -> tuple[str, dict[str, float], tuple[str, ...], str]:
    """Validate a ranking request and fill its defaults. Raises on anything unrecognised."""
    if mode not in ('mdl', 'weighted', 'pareto'):
        raise ValueError(f"unknown ranking_mode {mode!r}; expected 'mdl', 'weighted' or 'pareto'")

    weights = dict(weights or {})
    unknown = sorted(set(weights) - set(RANKING_METRICS))
    if unknown:
        raise ValueError(f"unknown ranking_weights {unknown}; known metrics: {sorted(RANKING_METRICS)}")

    metrics = tuple(metrics or ('fvu', 'n_nodes'))
    unknown = sorted(set(metrics) - set(RANKING_METRICS))
    if unknown:
        raise ValueError(f"unknown ranking_metrics {unknown}; known metrics: {sorted(RANKING_METRICS)}")
    if 'fvu' not in metrics:
        raise ValueError(
            "ranking_metrics must contain 'fvu': it is the only term that measures FIT, and a front "
            "without it would order candidates on shape alone.")

    # The tie-break may name a metric OUTSIDE the declared set -- that is deliberate (owner ruling):
    # ordering within a front is a separate question from which axes define the front.
    tie_break = tie_break or 'fvu'
    if tie_break not in RANKING_METRICS:
        raise ValueError(f"unknown ranking_tie_break {tie_break!r}; known metrics: {sorted(RANKING_METRICS)}")
    return mode, weights, metrics, tie_break


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
