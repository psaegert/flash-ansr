"""Canonical candidate-scoring primitives (single source of truth).

These are owned by flash-ansr because the product needs them at *inference* time to score and
rank decode candidates (``flash_ansr.py`` / ``generation/mcts.py`` / ``results.py``). They are
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

from typing import Iterable

import numpy as np

from flash_ansr.utils.ieee754 import IEEE754_END_TOKEN, NIBBLE_TOKENS

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


def score_from_fvu(
        fvu: float,
        complexity: int,
        constant_count: int,
        log_prob: float | None,
        length_penalty: float,
        constants_penalty: float,
        likelihood_penalty: float) -> float:
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

    return float(np.log10(safe_fvu)
                 + length_penalty * complexity
                 + constants_penalty * max(int(constant_count), 0)
                 + likelihood_term)


def is_constant_token(token: str) -> bool:
    """Return ``True`` if ``token`` denotes a constant in a prefix expression.

    Recognises the ``<constant>`` placeholder, generated ``C_i`` symbols, a small set of named
    literals (signed/unsigned ``0``/``1``, ``np.pi``, ``np.e``, the float specials), any token
    that parses as a Python ``float``, and the v24 ``ieee754_mixed`` constant forms: the compact
    ``<float>`` token and the ``<ieee754>`` span OPENING tag. The hex-nibble tokens
    ``<h0>``..``<hf>`` and the closing ``</ieee754>`` tag deliberately do NOT count, so a whole
    10-token expanded span contributes exactly ONE constant to any per-token sum (the count
    that feeds ``constants_penalty`` in :func:`score_from_fvu` must see one per span, not 10).
    """
    if token in ('<constant>', '<float>', '<ieee754>'):
        return True
    if token == IEEE754_END_TOKEN or token in NIBBLE_TOKENS:
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
