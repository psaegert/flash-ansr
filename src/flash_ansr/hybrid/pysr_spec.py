"""PySR's default expression spec without the export step. Imported lazily: importing pysr starts Julia.

At the end of every fit, PySR turns each hall-of-fame entry into a sympy expression and a numpy function built from
it. The hybrid never uses either: it reads the equation strings and parses and evaluates them with its own engine.
Building them is not free and has no bound: on 2026-09-29 one hall of fame holding a 25-deep tanh/sinh chain kept a
fit busy in sympy for more than 8 hours after an 11-second search. The hybrid's PySR therefore builds no exports; the
search, the hall of fame and PySR's own pick are unchanged.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from pysr.expression_specs import ExpressionSpec

__all__ = ["NoExportExpressionSpec"]


class NoExportExpressionSpec(ExpressionSpec):
    """``ExpressionSpec`` whose ``create_exports`` adds no columns: no sympy, numpy, jax or torch formats. A module-level
    class, so PySR's checkpoint pickle of the regressor keeps working."""

    def create_exports(self, model: Any, equations: pd.DataFrame, search_output: Any, i: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(index=equations.index)
