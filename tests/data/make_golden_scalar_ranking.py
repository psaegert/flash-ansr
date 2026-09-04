"""Regenerate ``golden_scalar_ranking_0130.json`` -- the frozen pre-change scalar ranking.

Run BEFORE any ranking change, from the pre-change tree:

    python tests/data/make_golden_scalar_ranking.py

The golden is captured through :meth:`FlashANSR._compile_results_pure`, NOT through
:func:`score_from_fvu`, and that is the whole point: at ``flash_ansr.py:2149-2156`` a non-finite FVU
is overwritten to ``nan`` and never reaches the scorer, so a scorer-level golden would assert
behaviour the shipped pipeline does not have. The sort's three keys (score, is-nan, expression-token
tie-break) are likewise only observable here.

``_compile_results_pure`` reads ``self`` for exactly two read-only helpers -- ``_score_from_fvu`` and
``_count_constants`` (verified by inspection of the method body) -- so the golden needs no
checkpoint, no refiner, no GPU and no RNG. It is byte-reproducible on any machine, which a
``fit()``-driven golden would not be.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from flash_ansr.flash_ansr import FlashANSR                      # noqa: E402
from flash_ansr.scoring import count_constants, score_from_fvu   # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_scalar_ranking_0130.json")

#: Each case is (expression tokens, fvu, log_prob, constant_count or None to force the fallback).
#: The set deliberately covers every branch in the scoring block and every sort key.
CASES: list[tuple[list[str], float, float | None, int | None]] = [
    (["+", "*", "<constant>", "x1", "<constant>"], 1e-3, -4.0, 2),
    (["*", "<constant>", "x1"], 1e-2, -2.0, 1),
    (["x1"], 0.5, -1.0, 0),
    (["+", "x1", "x2"], 0.5, -1.0, 0),                  # exact score tie with the row above at 0
    (["sin", "x1"], 0.5, -1.0, 0),                      # ... and a third, to force the token order
    (["*", "<constant>", "sin", "x1"], 0.0, -3.0, 1),   # perfect fit -> floored to FLOAT64_EPS
    (["exp", "x1"], float("inf"), -2.0, 0),             # non-finite -> score overwritten to nan
    (["log", "x1"], float("nan"), -2.0, 0),             # nan fvu -> same path
    (["cos", "x1"], -1.0, -2.0, 0),                     # negative but FINITE -> reaches the scorer -> +inf
    (["/", "x1", "<constant>"], 1e-1, None, 1),         # log_prob None -> no likelihood term
    (["pow", "x1", "2"], 1e-1, -6.0, None),             # constant_count absent -> _count_constants
    (["+", "<constant>", "<constant>"], 2e-1, float("inf"), 2),   # non-finite log_prob -> no term
]

PENALTY_GRID = [
    (lp, cp, kp)
    for lp in (0.0, 0.05, 0.2)
    for cp in (0.0, 0.1)
    for kp in (0.0, 0.01)
]


class _Shim:
    """The only two attributes ``_compile_results_pure`` reads off ``self``."""

    _score_from_fvu = staticmethod(score_from_fvu)
    _count_constants = staticmethod(count_constants)


def _fresh_results() -> list[dict]:
    # 'score' MUST be present: the scoring block is guarded by `if 'score' in result`.
    out = []
    for expression, fvu, log_prob, constant_count in CASES:
        row: dict = {"expression": list(expression), "fvu": fvu, "log_prob": log_prob, "score": np.nan}
        if constant_count is not None:
            row["constant_count"] = constant_count
        out.append(row)
    # NOTE (found while capturing this golden, 2026-09-04): a row with NO 'score' key is NOT a
    # supported input, despite the scoring block being guarded by `if 'score' in result`. The sort
    # key at :2165 reads `x['score']` unconditionally, so such a row raises KeyError before it can
    # be ordered. The guard and the sort disagree about whether score-less rows are tolerated. Not
    # reachable in production -- every producer sets 'score' -- so this is a latent inconsistency,
    # not a live defect, and it is asserted explicitly in the golden test rather than encoded here.
    return out


def _jsonable(value: float) -> object:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return repr(value)      # repr round-trips a float64 exactly
    return value


def build() -> dict:
    cells = []
    for lp, cp, kp in PENALTY_GRID:
        sorted_results, _ = FlashANSR._compile_results_pure(_Shim(), _fresh_results(), lp, cp, kp)
        cells.append({
            "node_penalty": lp,          # named for the post-rename world; value is the old length_penalty
            "constants_penalty": cp,
            "likelihood_penalty": kp,
            "order": [
                {
                    "expression": r["expression"],
                    "score": _jsonable(r.get("score", float("nan"))),
                    "fvu": _jsonable(r["fvu"]),
                    "n_nodes": len(r["expression"]),
                    "constant_count": r.get("constant_count", count_constants(r["expression"])),
                }
                for r in sorted_results
            ],
        })
    return {
        "note": "Frozen scalar ranking captured through FlashANSR._compile_results_pure before the "
                "ranking-modes change. Floats are repr() strings so the comparison is exact; "
                "'nan'/'inf'/'-inf' are spelled out.",
        "n_cases": len(_fresh_results()),
        "cells": cells,
    }


if __name__ == "__main__":
    payload = build()
    with open(GOLDEN, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"wrote {GOLDEN}: {len(payload['cells'])} cells x {payload['n_cases']} candidates")
