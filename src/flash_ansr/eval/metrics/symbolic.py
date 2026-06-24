"""Symbolic / structural metrics for comparing predicted expressions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def total_nestedness(prefix_skeleton: Sequence[str], operator_arity: Mapping[str, int]) -> int:
    """Count the total nestedness of unary operators in a prefix skeleton.

    Nestedness increments for each consecutive unary operator beyond the
    first in a chain.  For example ``sin(cos(x))`` has nestedness 1,
    while ``sin(cos(exp(x)))`` has nestedness 2.

    Parameters
    ----------
    prefix_skeleton : Sequence[str]
        Expression in prefix notation.
    operator_arity : Mapping[str, int]
        Map from operator name to its arity.

    Returns
    -------
    int
        Total nestedness score.
    """
    nestedness = 0
    current_depth = 0
    for token in prefix_skeleton:
        if operator_arity.get(token, 0) == 1:
            current_depth += 1
        else:
            nestedness += max(0, current_depth - 1)
            current_depth = 0
    return nestedness
