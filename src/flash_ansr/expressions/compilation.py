"""Helpers for compiling and evaluating expression programs."""
import time
from typing import Callable

import numpy as np

from types import CodeType


def codify(code_string: str, variables: list[str] | None = None) -> CodeType:
    """Compile an infix expression body into a callable lambda."""
    if variables is None:
        variables = []
    func_string = f"lambda {', '.join(variables)}: {code_string}"
    filename = f"<lambdifygenerated-{time.time_ns()}"
    return compile(func_string, filename, "eval")


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    """Evaluate ``f`` on ``X`` while normalising scalar outputs to vectors."""
    if constants is None:
        y = f(*X.T)
    else:
        y = f(*X.T, *constants)
    if not isinstance(y, np.ndarray) or y.shape[0] == 1:
        y = np.full(X.shape[0], y)
    return y
