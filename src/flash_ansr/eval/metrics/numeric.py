"""Numeric evaluation metrics for comparing predictions to ground truth."""
from __future__ import annotations

import numpy as np


def safe_divide(a: float, b: float) -> float:
    """Divide ``a`` by ``b``, returning 0 for 0/0 and inf for x/0."""
    if b == 0:
        if a == 0:
            return 0.0
        return np.inf
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a / b


def fvu(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> float:
    """Compute Fraction of Variance Unexplained between two arrays.

    Uses numerical scaling to avoid floating-point precision issues.

    Parameters
    ----------
    y_true : np.ndarray or None
        Ground truth values.
    y_pred : np.ndarray or None
        Predicted values.

    Returns
    -------
    float
        FVU value.  Returns ``np.inf`` when inputs are invalid.
    """
    if y_pred is None or y_true is None:
        return np.inf

    if not isinstance(y_pred, np.ndarray) and np.isnan(y_pred):
        return np.inf

    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    # Ground truth finite but prediction not → infinite error
    if np.isfinite(y_true).all() and not np.isfinite(y_pred).all():
        return np.inf

    # Scale by inverse MSE to avoid numerical issues
    ss_res = np.mean((y_true - y_pred) ** 2)
    if ss_res == 0:
        return 0.0

    # Overflow guard. A finite-but-DIVERGENT prediction can make the squared residual
    # overflow to +inf; the 1/ss_res rescale below then drives scale -> 0, collapsing every
    # term to 0 so safe_divide(0, 0) returns 0.0 and is_perfect_fit() spuriously fires
    # (e.g. fvu([1,2,3,4,5], [1,2,3,4,1e167]) used to return 0.0). Decide good-vs-bad robustly
    # by re-scaling the residual and total variance by the GT magnitude (an O(1) normalizer
    # that never collapses a genuine divergence to zero). The finite-ss_res path is left
    # byte-identical, so all non-pathological results are unchanged.
    if not np.isfinite(ss_res):
        denom = np.max(np.abs(y_true))
        if not np.isfinite(denom) or denom == 0.0:
            return np.inf
        ss_res_n = np.mean(((y_true - y_pred) / denom) ** 2)
        ss_tot_n = np.mean(((y_true - np.mean(y_true)) / denom) ** 2)
        if not np.isfinite(ss_res_n):
            return np.inf
        return safe_divide(ss_res_n, ss_tot_n)

    scale = 1.0 / ss_res

    ss_res = np.mean((y_true * scale - y_pred * scale) ** 2)
    ss_tot = np.mean((y_true * scale - np.mean(y_true * scale, keepdims=True)) ** 2)

    return safe_divide(ss_res, ss_tot)


def log10_fvu(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> float:
    """Compute log10 of the Fraction of Variance Unexplained.

    Returns ``-np.inf`` when FVU is exactly zero (perfect fit).
    """
    fvu_value = fvu(y_true, y_pred)
    if fvu_value == 0:
        return -np.inf
    return np.log10(fvu_value)


def is_perfect_fit(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> bool:
    """Check if ``y_pred`` perfectly fits ``y_true`` within float32 epsilon."""
    return fvu(y_true, y_pred) <= np.finfo(np.float32).eps


def naninfmean(a: np.ndarray) -> float:
    """Compute the mean of an array, ignoring NaN and Inf values."""
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a[np.isfinite(a)]))
