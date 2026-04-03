"""Formatting helpers for evaluation plots and labels."""
from __future__ import annotations

import numpy as np


def arrow_notation(
    objective: int = 1,
    lower: float | str = 0,
    upper: float | str = 1,
    lower_open: bool = False,
    upper_open: bool = False,
) -> str:
    r"""Build a LaTeX arrow annotation indicating metric direction and range.

    Parameters
    ----------
    objective : {1, -1}
        ``1`` for "higher is better" (↑), ``-1`` for "lower is better" (↓).
    lower, upper : float or str
        Range bounds.  ``-np.inf`` and ``np.inf`` are rendered as
        ``-\infty`` / ``\infty`` with open brackets.
    lower_open, upper_open : bool
        Whether the corresponding bound is open (parenthesis) or closed
        (bracket).

    Returns
    -------
    str
        LaTeX string, e.g. ``\uparrow^{[0, 100]}``.
    """
    lower_bracket = '(' if lower_open else '['
    upper_bracket = ')' if upper_open else ']'

    if lower == -np.inf:
        lower_bracket = '('
        lower = '-\\infty'
    if upper == np.inf:
        upper_bracket = ')'
        upper = '\\infty'

    try:
        fl = float(lower)  # type: ignore[arg-type]
        lower = f"{fl:.2g}" if int(fl) != fl else f"{int(fl)}"
    except (ValueError, TypeError):
        pass

    try:
        fu = float(upper)  # type: ignore[arg-type]
        upper = f"{fu:.2g}" if int(fu) != fu else f"{int(fu)}"
    except (ValueError, TypeError):
        pass

    if objective == 1:
        return f"\\uparrow^{{{lower_bracket}{lower}, {upper}{upper_bracket}}}"
    elif objective == -1:
        return f"\\downarrow^{{{lower_bracket}{lower}, {upper}{upper_bracket}}}"
    else:
        raise ValueError("Objective must be 1 (higher is better) or -1 (lower is better).")
