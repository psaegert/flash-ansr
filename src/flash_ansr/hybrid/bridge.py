"""The bridge between the two stages: Flash-ANSR's prefix grammar and PySR's Julia-syntax infix, the
engine-side evaluation of a realized prefix, and the data fingerprint a cached generation pass keys on."""
from __future__ import annotations

import hashlib
import importlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import simplipy
import simplipy.operators  # noqa: F401 - operator realizations reference it by dotted name
from simplipy.utils import codify, safe_f

from flash_ansr.hybrid.pysr_model import UNARY_OPERATORS

__all__ = ["PYSR_UNARY", "PYSR_BINARY", "PYSR_OPERATORS", "SPECIAL_LITERALS", "julia_literal", "literal_value",
           "prefix_to_julia", "evaluate_prefix", "data_fingerprint"]

#: PySR's vocabulary in the engine grammar (the GP stage's, :mod:`flash_ansr.hybrid.pysr_model`): a
#: seed using anything else is not offered
PYSR_UNARY = frozenset(UNARY_OPERATORS)
PYSR_BINARY = frozenset({"+", "-", "*", "/", "pow", "rootn"})
PYSR_OPERATORS = PYSR_UNARY | PYSR_BINARY
SPECIAL_LITERALS = {"np.pi": math.pi, "np.e": math.e, "pi": math.pi, "e": math.e}


def literal_value(token: str) -> float | None:
    """The float a prefix leaf spells (``np.pi`` and friends included); None for a non-literal."""
    if token in SPECIAL_LITERALS:
        return float(SPECIAL_LITERALS[token])
    try:
        return float(token)
    except ValueError:
        return None


def julia_literal(token: str) -> str | None:
    """A prefix leaf as a Julia literal: floats as ``repr``, rationals ``p/q`` evaluated, negatives in
    parentheses; None for a non-literal or a non-finite value."""
    if token in SPECIAL_LITERALS:
        return repr(float(SPECIAL_LITERALS[token]))
    try:
        value = float(token)
    except ValueError:
        if "/" in token:
            num, _, den = token.partition("/")
            try:
                value = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return None
        else:
            return None
    if not math.isfinite(value):
        return None
    text = repr(value)
    return f"({text})" if value < 0 else text


def prefix_to_julia(tokens: Sequence[str], arity: Mapping[str, int], variables: Sequence[str]) -> str | None:
    """A realized prefix expression (constants inlined, ``x1..xn`` variables) as the Julia-syntax
    infix string PySR's ``guesses`` parser reads: unaries as calls, ``pow`` as ``^``, ``rootn`` as
    a call, negative literals in parentheses, ``x<i>`` mapped onto the fit's variable names.
    ``None`` when a token is outside PySR's vocabulary (the seed is then not offered)."""
    tokens = list(tokens)

    def render(i: int) -> tuple[str, int]:
        tok = tokens[i]
        n = int(arity.get(tok, 0))
        if n == 0:
            if len(tok) > 1 and tok[0] == "x" and tok[1:].isdigit():
                k = int(tok[1:]) - 1
                if k < 0 or k >= len(variables):
                    raise ValueError("variable outside the problem")
                return str(variables[k]), i + 1
            lit = julia_literal(tok)
            if lit is None:
                raise ValueError(f"unknown leaf {tok!r}")
            return lit, i + 1
        if tok not in PYSR_OPERATORS:
            raise ValueError(f"operator {tok!r} is not in PySR's vocabulary")
        args = []
        j = i + 1
        for _ in range(n):
            s, j = render(j)
            args.append(s)
        if tok in ("+", "-", "*", "/"):
            return f"({args[0]} {tok} {args[1]})", j
        if tok == "pow":
            return f"({args[0]} ^ {args[1]})", j
        return f"{tok}({', '.join(args)})", j

    try:
        text, end = render(0)
    except (ValueError, IndexError):
        return None
    return text if end == len(tokens) else None


def evaluate_prefix(engine: Any, prefix: Sequence[str], variables: Sequence[str], *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Evaluate a prefix expression (numeric constants inlined) on each array of rows with the
    engine's own operator realizations; one ``(n, 1)`` column per array, empty for empty input."""
    realized = engine.operators_to_realizations(list(prefix))
    code_string = engine.prefix_to_infix(realized, realization=True)
    code = codify(code_string, list(variables))
    namespace: dict[str, Any] = {"np": np, "numpy": np, "math": math, "simplipy": simplipy}
    for root in getattr(engine, "_realization_roots", {}) or {}:
        try:
            namespace.setdefault(root, importlib.import_module(root))
        except Exception:  # noqa: BLE001 - an unimportable root surfaces as NameError below
            pass
    function = eval(code, namespace)  # noqa: S307 - the engine's own realization code
    outputs = []
    for array in arrays:
        rows = np.asarray(array, dtype=float)
        if rows.ndim != 2 or rows.shape[0] == 0:
            outputs.append(np.empty((0, 1), dtype=float))
            continue
        with np.errstate(all="ignore"):
            values = safe_f(function, rows)
        outputs.append(np.asarray(values, dtype=float).reshape(-1, 1))
    return tuple(outputs)


def data_fingerprint(X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None) -> str:
    """A digest of the arrays a cached generation pass fitted on and predicted for: support X, the
    fitted y, validation X. A snapshot is only valid on the exact data it was generated on."""
    h = hashlib.sha1()
    for arr in (X, y, X_val if X_val is not None else np.empty((0, 0))):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()
