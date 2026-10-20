"""The GP stage: PySRRegressor over flash-ansr's 23-operator vocabulary, in-process.

The same regressor srbf's PySR baseline worker builds (srbf/worker/models/pysr_worker.py keeps its own
copy: the two must not drift -- 17 unaries + {+, -, *, /, ^, rootn}, upstream defaults otherwise), with three
differences outside the search:

* the hybrid's PySR builds no sympy/numpy exports of its hall of fame (``pysr_spec.NoExportExpressionSpec``), because
  the hybrid parses and evaluates the equation strings itself;
* ``rootn`` is defined in Julia ONCE (:func:`define_operators`) and handed to PySR by name. PySR evaluates an inline
  operator definition again in every fit; Julia then replaces the method and recompiles everything compiled against
  it, which cost ~6 s per fit on valkyrie-light (1 iteration: 7.8-9.1 s inline, 1.8-2.1 s defined once, 2026-10-20),
  more than the search itself at small budgets. The definition is the same text, so the search is the same. rootn is
  srbf's addition to PySR's vocabulary; PySR's own code runs as shipped, as in srbf's PySR baseline (owner
  2026-10-20: the hybrid's PySR must not be faster than that baseline's for reasons other than the seeds);
* the seeds may arrive while PySR is being set up (:class:`SeedFuture`): PySR prepares everything that does not
  depend on them (its operator checks, options and data) and waits for them at the one point where it reads its
  guesses, so the hybrid can run that setup while Flash-ANSR is still generating.

``pysr`` is imported lazily: Julia is fetched and compiled on first use, and ``warmup`` pays the one-off
SymbolicRegression.jl compile on a throwaway fit (with a seed, so the seeded path is compiled too) when the hybrid is
created, so the first timed search starts warm.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["PySRSettings", "SeedFuture", "UNARY_OPERATORS", "BINARY_OPERATORS", "ROOTN_JULIA", "define_operators", "create_model",
           "warmup", "run_pysr"]

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
#: PySR's binary operators: rootn by NAME, defined once in Julia by :func:`define_operators` (an inline definition would
#: be evaluated, and everything compiled against it recompiled, in every fit).
BINARY_OPERATORS = ["+", "-", "*", "/", "^", "rootn"]
_OPERATORS_DEFINED = False
_OPERATORS_LOCK = threading.Lock()


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


def _seval(code: str) -> None:
    """Evaluate Julia code in PySR's Julia session (``Main``)."""
    from pysr import jl
    jl.seval(code)


def define_operators() -> None:
    """Define the custom operators (``rootn``) in Julia, once per process: on the first call -- the warmup when the
    hybrid is created, else the first fit -- and never again."""
    global _OPERATORS_DEFINED
    with _OPERATORS_LOCK:
        if not _OPERATORS_DEFINED:
            _seval(ROOTN_JULIA)
            _OPERATORS_DEFINED = True


class SeedsUnavailable(RuntimeError):
    """The seeds a waiting PySR search was promised never came: the stage that makes them failed."""


class SeedFuture:
    """The seeds of a search that is set up before they exist. :func:`run_pysr` starts PySR at once; PySR checks its
    operators, builds its options and converts the data, and at the point where it reads its guesses it waits here
    until the seed stage calls :meth:`set` (an empty list: a search without seeds) or :meth:`fail` (the search ends
    with :class:`SeedsUnavailable`). ``reached_at`` / ``released_at`` (``time.time()``) record when PySR got
    there and when the seeds came, so the record can tell PySR's setup from its wait and its search."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._seeds: list[str] = []
        self._error: BaseException | None = None
        self.reached_at: float | None = None
        self.released_at: float | None = None

    def set(self, seeds: Sequence[str] | None) -> None:
        self._seeds = [str(g) for g in (seeds or [])]
        self._done.set()

    def fail(self, error: BaseException) -> None:
        self._error = error
        self._done.set()

    @property
    def seeds(self) -> list[str]:
        """The seeds handed over (empty until :meth:`set`)."""
        return list(self._seeds)

    def wait(self) -> list[str]:
        self.reached_at = time.time()
        self._done.wait()
        self.released_at = time.time()
        if self._error is not None:
            raise SeedsUnavailable(f"the seed stage failed: {type(self._error).__name__}: {self._error}") from self._error
        return list(self._seeds)


_PENDING: SeedFuture | None = None   # the future of the search running on this process (one at a time)


def _install_seed_hook() -> None:
    """Route PySR's guesses through the pending :class:`SeedFuture`: ``pysr.sr._prepare_guesses_for_julia`` is the one
    place PySR's fit reads ``guesses`` (PySR 2.x), right before the search starts. A PySR without that function cannot
    take late seeds, and refuses here rather than searching without them."""
    import pysr.sr as sr
    original = getattr(sr, "_prepare_guesses_for_julia", None)
    if original is None:
        raise RuntimeError("this PySR has no _prepare_guesses_for_julia: the hybrid cannot hand it seeds late")
    if getattr(original, "_flash_ansr_seed_hook", False):
        return

    def prepare(guesses: Any, *args: Any, **kwargs: Any) -> Any:
        pending = _PENDING
        if pending is not None:
            guesses = pending.wait() or None
        return original(guesses, *args, **kwargs)

    prepare._flash_ansr_seed_hook = True  # type: ignore[attr-defined]
    sr._prepare_guesses_for_julia = prepare


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
    define_operators()
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
    """When the hybrid is created: Julia's startup, the operator definitions and the one-off SymbolicRegression.jl
    compile, on a throwaway minimal fit with one seed (the seeded path compiles on its first use too).
    Best-effort: a failing warmup forfeits only the warmup."""
    define_operators()
    model = create_model(timeout_in_seconds=60, niterations=1, settings=settings, guesses=["(x0 + 1.0)"])
    x = np.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 1.0
    try:
        model.fit(x, y, variable_names=["x0"])
    except Exception:  # noqa: BLE001 - never blocks evaluation
        pass


def run_pysr(X: np.ndarray, y: np.ndarray, variables: Sequence[str], *, timeout_in_seconds: float, niterations: int,
             guesses: Sequence[str] | SeedFuture | None = None, settings: PySRSettings | None = None) -> dict[str, Any]:
    """One PySR search on its own clock. Returns PySR's own pick (``expression``, infix in the fit's
    variable names), the whole hall of fame (``equations``: complexity, loss, score, equation) and the
    wall time; on failure ``error`` and no expression. No curves: the caller evaluates the equation
    strings with its own engine (PySR builds no exports here, see ``create_model``).

    ``guesses`` may be a :class:`SeedFuture` (seeds that another thread is still making): PySR is set up at once and
    waits for them where it reads its guesses; ``setup_s`` (until it got there), ``wait_s`` (for the seeds) and
    ``search_s`` (the rest) split ``wall_s``. Call it from the thread that runs Julia (the main thread)."""
    global _PENDING
    X = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).ravel()
    future: SeedFuture | None = None
    listed: list[str] | None = None
    if isinstance(guesses, SeedFuture):
        future = guesses
    else:
        listed = [str(g) for g in (guesses or [])]
    out: dict[str, Any] = {"expression": None, "equations": [], "n_guesses": len(listed or []),
                           "niterations_used": int(niterations), "error": None}
    t0 = time.time()
    try:
        if future is not None:
            _require_pysr()
            _install_seed_hook()
        model = create_model(timeout_in_seconds=timeout_in_seconds, niterations=niterations, settings=settings, guesses=listed)
        if future is not None:
            _PENDING = future
            try:
                model.fit(X, target, variable_names=list(variables))
            finally:
                _PENDING = None
            if future.reached_at is None:
                raise RuntimeError("PySR's fit never read its guesses: the seeds were not handed over")
            out["n_guesses"] = len(future.seeds)
        else:
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
    t1 = time.time()
    out["wall_s"] = t1 - t0
    if future is not None:
        reached, released = future.reached_at, future.released_at
        out["setup_s"] = (reached - t0) if reached is not None else out["wall_s"]
        out["wait_s"] = (released - reached) if reached is not None and released is not None else 0.0
        out["search_s"] = (t1 - released) if released is not None else 0.0
    return out
