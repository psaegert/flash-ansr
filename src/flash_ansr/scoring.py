"""Canonical candidate-scoring primitives (single source of truth).

These are owned by flash-ansr because the product needs them at *inference* time to score and
rank decode candidates (``flash_ansr.py`` / ``generation/mcts.py`` / ``results.py``). They are
also consumed by the comparison baselines and, after the planned repo split, by ``srbf`` via the
public API. This module collapses the formerly-triplicated
``_compute_fvu`` / ``_normalize_variance`` / ``_score_from_fvu`` copies
(``flash_ansr.py`` + ``baselines/{brute_force_model,skeleton_pool_model}.py``) onto one definition.

Note on FVU: this is the *scalar-loss* form. The refiner supplies an already-reduced residual
``loss``, so :func:`compute_fvu` divides by the (epsilon-floored) target variance. The *array* form
``fvu(y_true, y_pred)`` lives in the evaluation metrics (``eval/metrics/numeric.py``) and is a
deliberately separate signature for a different call site: the two share the FVU *definition*
(residual / total variance) but are not interchangeable functions.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

#: Floor used to keep variance and FVU strictly positive before a division / log.
FLOAT64_EPS: float = float(np.finfo(np.float64).eps)


def normalize_variance(variance: float) -> float:
    """Return ``variance`` floored at :data:`FLOAT64_EPS` (also for non-finite input)."""
    if not np.isfinite(variance):
        return FLOAT64_EPS
    return max(float(variance), FLOAT64_EPS)


def compute_fvu(loss: float, sample_count: int, variance: float) -> float:
    """Fraction of Variance Unexplained from a precomputed residual ``loss``.

    With a single sample the target variance is undefined, so the raw ``loss`` is returned.
    """
    if sample_count <= 1:
        return float(loss)
    return float(loss) / normalize_variance(variance)


def score_from_fvu(
        fvu: float,
        complexity: int,
        constant_count: int,
        log_prob: float | None,
        length_penalty: float,
        constants_penalty: float,
        likelihood_penalty: float) -> float:
    """Parsimony-penalised selection score ``log10(FVU) + structural penalties`` (lower is better).

    ``fvu`` is floored at :data:`FLOAT64_EPS` so a perfect fit does not send the score to ``-inf``;
    a non-finite or non-positive FVU is treated as the floor. ``log_prob`` contributes
    ``likelihood_penalty * (-log_prob)`` when finite, and nothing otherwise.
    """
    if not np.isfinite(fvu) or fvu <= 0:
        safe_fvu = FLOAT64_EPS
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
    literals (signed/unsigned ``0``/``1``, ``np.pi``, ``np.e``, the float specials), and any token
    that parses as a Python ``float``. This is the constant count that feeds ``constants_penalty``
    in :func:`score_from_fvu`, so it is owned here alongside the rest of the scoring path.
    """
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


def count_constants(expression: Iterable[str] | None) -> int:
    """Count the constant tokens in ``expression`` (``None`` -> ``0``)."""
    if expression is None:
        return 0
    return sum(1 for token in expression if is_constant_token(str(token)))
