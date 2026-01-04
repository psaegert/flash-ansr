"""Helpers for compiling and evaluating expression programs."""
import re
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


def get_used_modules(infix_expression: str) -> list[str]:
    """Return the top-level modules referenced by dotted calls in ``infix_expression``."""
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\(")
    matches = pattern.findall(infix_expression)
    modules_set = {match.split(".")[0] for match in matches}
    modules_set.update(["numpy"])
    return list(modules_set)


def safe_f(f: Callable, X: np.ndarray, constants: np.ndarray | None = None) -> np.ndarray:
    """Evaluate ``f`` on ``X`` while normalising scalar outputs to vectors."""
    if constants is None:
        y = f(*X.T)
    else:
        y = f(*X.T, *constants)
    if not isinstance(y, np.ndarray) or y.shape[0] == 1:
        y = np.full(X.shape[0], y)
    return y
