"""The GP stage: PySRRegressor over flash-ansr's 23-operator vocabulary, in-process.

The same regressor srbf's PySR baseline worker builds (srbf/worker/models/pysr_worker.py keeps its own
copy: the two must not drift -- 17 unaries + {+, -, *, /, ^, rootn}, upstream defaults otherwise), with one
difference outside the search: the hybrid's PySR builds no sympy/numpy exports of its hall of fame
(``pysr_spec.NoExportExpressionSpec``), because the hybrid parses and evaluates the equation strings itself.
``pysr`` is imported lazily: Julia is fetched and compiled on first use, and ``warmup`` pays the
one-off SymbolicRegression.jl compile on a throwaway fit so the first timed search starts warm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["PySRSettings", "UNARY_OPERATORS", "BINARY_OPERATORS", "ROOTN_JULIA", "create_model", "warmup", "run_pysr"]

UNARY_OPERATORS = [
    "neg", "abs", "inv", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log",
]
# IEEE-754 rootn, matching simplipy.operators.rootn's table: odd integer index = signed root (total on
# R), even = principal (NaN on negatives), negative index = reciprocal, index 0 or non-integer = NaN.
# `abs(n) < 2^53`: every float beyond that is "an integer", and `Int(abs(n))` overflows Int64 for
# |n| >= 2^63 (InexactError) -- a GP mutation or a seed can put 1e33 into the index slot.
ROOTN_JULIA = (
    r"rootn(x::T, n::T) where {T} = (isfinite(n) && n == round(n) && n != 0 && abs(n) < 9.007199254740992e15) ? "
    r"((x >= 0) ? abs(x)^(one(T)/n) : (isodd(Int(abs(n))) ? -abs(x)^(one(T)/n) : T(NaN))) : T(NaN)"
)
BINARY_OPERATORS = ["+", "-", "*", "/", "^", ROOTN_JULIA]


@dataclass
class PySRSettings:
    """PySR's knobs the hybrid forwards: ``maxsize`` / ``parsimony`` (None = PySR's own defaults),
    ``model_selection`` (PySR's own pick, recorded for reference only), ``warmup`` (pay the Julia
    compile before the first timed fit), ``extra`` (further ``PySRRegressor`` kwargs)."""

    maxsize: int | None = None
    parsimony: float | None = None
    model_selection: str = "best"
    warmup: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PySRSettings":
        value = dict(value or {})
        known = {k: value.pop(k) for k in ("maxsize", "parsimony", "model_selection", "warmup") if k in value}
        extra = dict(value.pop("extra", {}) or {})
        extra.update(value)   # any other key is a PySRRegressor kwarg
        return cls(**known, extra=extra)


def _require_pysr() -> Any:
    try:
        from pysr import PySRRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pysr is required for the hybrid's GP stage: pip install pysr") from exc
    return PySRRegressor


def _no_export_spec() -> Any:
    """PySR's default expression spec without the export step (``flash_ansr.hybrid.pysr_spec``, which imports pysr)."""
    from flash_ansr.hybrid.pysr_spec import NoExportExpressionSpec
    return NoExportExpressionSpec()


def create_model(*, timeout_in_seconds: float, niterations: int, settings: PySRSettings | None = None,
                 guesses: Sequence[str] | None = None) -> Any:
    """A PySRRegressor over the 23-operator vocabulary; ``guesses`` (Julia-syntax infix strings in the
    fit's variable names) seed its initial populations (PySR >= 2.0). It builds no sympy/numpy exports
    of its hall of fame (an ``expression_spec`` in ``settings.extra`` replaces that choice), so it has
    no ``predict``: the hybrid evaluates the equation strings with its own engine."""
    PySRRegressor = _require_pysr()
    settings = settings or PySRSettings()
    optional: dict[str, Any] = dict(settings.extra)
    if "expression_spec" not in optional:
        optional["expression_spec"] = _no_export_spec()
    if settings.maxsize is not None:
        optional["maxsize"] = int(settings.maxsize)
    if settings.parsimony is not None:
        optional["parsimony"] = float(settings.parsimony)
    if guesses:
        optional["guesses"] = [str(g) for g in guesses]
    return PySRRegressor(
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=int(max(1, round(timeout_in_seconds))),
        niterations=int(niterations),
        model_selection=settings.model_selection,
        unary_operators=list(UNARY_OPERATORS),
        binary_operators=list(BINARY_OPERATORS),
        extra_sympy_mappings={"rootn": lambda x, n: x ** (1 / n)},   # principal branch on export
        **optional,
    )


def warmup(settings: PySRSettings | None = None) -> None:
    """Pay the Julia startup + SymbolicRegression.jl compile OUTSIDE any timed fit: a throwaway minimal
    fit, best-effort (a failing warmup forfeits only the warmup)."""
    model = create_model(timeout_in_seconds=60, niterations=1, settings=settings)
    x = np.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 1.0
    try:
        model.fit(x, y, variable_names=["x0"])
    except Exception:  # noqa: BLE001 - never blocks evaluation
        pass


def run_pysr(X: np.ndarray, y: np.ndarray, variables: Sequence[str], *, timeout_in_seconds: float, niterations: int,
             guesses: Sequence[str] | None = None, settings: PySRSettings | None = None) -> dict[str, Any]:
    """One PySR search on its own clock. Returns PySR's own pick (``expression``, infix in the fit's
    variable names), the whole hall of fame (``equations``: complexity, loss, score, equation) and the
    wall time; on failure ``error`` and no expression. No curves: the caller evaluates the equation
    strings with its own engine (PySR builds no exports here, see ``create_model``)."""
    X = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).ravel()
    out: dict[str, Any] = {"expression": None, "equations": [], "n_guesses": len(guesses or []),
                           "niterations_used": int(niterations), "error": None}
    t0 = time.time()
    try:
        model = create_model(timeout_in_seconds=timeout_in_seconds, niterations=niterations, settings=settings, guesses=guesses)
        model.fit(X, target, variable_names=list(variables))
        try:
            hof = model.equations_
            out["equations"] = hof[["complexity", "loss", "score", "equation"]].to_dict("records")
        except Exception:  # noqa: BLE001 - the hall of fame is best-effort
            pass
        best = model.get_best()
        out["expression"] = str(best["equation"])
    except Exception as exc:  # noqa: BLE001 - a failed search is recorded, never raised
        out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    out["wall_s"] = time.time() - t0
    return out
