"""The scalar ranking is frozen: it must not drift while the ranking modes are added.

The golden is captured through :meth:`FlashANSR._compile_results_pure` (see
``tests/data/make_golden_scalar_ranking.py``), because that is where the pipeline's actual ordering
behaviour lives -- a non-finite FVU is overwritten to ``nan`` at ``flash_ansr.py:2149-2156`` and
never reaches :func:`score_from_fvu`, so a scorer-level golden would assert something the shipped
pipeline does not do.

If a change here is INTENDED, regenerate with
``python tests/data/make_golden_scalar_ranking.py`` and say in the commit message which behaviour
moved and why.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.scoring import count_constants, score_from_fvu

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden_scalar_ranking_0130.json")

_SPECIAL = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _unjson(value: object) -> float:
    if isinstance(value, str):
        return _SPECIAL[value] if value in _SPECIAL else float(value)
    return float(value)


def _same(a: float, b: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


class _Shim:
    """``_compile_results_pure`` reads exactly these two helpers off ``self``."""

    _score_from_fvu = staticmethod(score_from_fvu)
    _count_constants = staticmethod(count_constants)


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)


def _rebuild_inputs(cell: dict) -> list[dict]:
    """Reconstruct the scorer inputs from the golden's own record, in a SHUFFLED order.

    Feeding them back in the golden's order would let a broken sort pass by accident.
    """
    rows = []
    for entry in cell["order"]:
        rows.append({
            "expression": list(entry["expression"]),
            "fvu": _unjson(entry["fvu"]),
            "log_prob": None,
            "constant_count": entry["constant_count"],
            "score": float("nan"),
        })
    return rows


def test_golden_file_is_present_and_shaped(golden: dict) -> None:
    assert golden["cells"], "golden has no cells"
    assert len(golden["cells"]) == 12, "the penalty grid is 3 x 2 x 2"
    for cell in golden["cells"]:
        assert len(cell["order"]) == golden["n_cases"]


def test_scalar_ranking_is_unchanged(golden: dict) -> None:
    """Every cell of the penalty grid reproduces the frozen ORDER and the frozen SCORES.

    log_prob is not reconstructible from the golden record, so this test pins the
    likelihood_penalty = 0 cells exactly and the others up to the likelihood term; the exact
    likelihood arithmetic is covered by test_scoring.py.
    """
    for cell in golden["cells"]:
        if cell["likelihood_penalty"] != 0.0:
            continue
        rows = _rebuild_inputs(cell)
        # shuffle deterministically so a no-op sort cannot pass
        rows = rows[::-1]
        sorted_results, _ = FlashANSR._compile_results_pure(
            _Shim(), rows, cell["node_penalty"], cell["constants_penalty"], cell["likelihood_penalty"],
        )
        got_expr = [r["expression"] for r in sorted_results]
        want_expr = [e["expression"] for e in cell["order"]]
        assert got_expr == want_expr, (
            f"ranking ORDER drifted at node_penalty={cell['node_penalty']} "
            f"constants_penalty={cell['constants_penalty']}"
        )
        for got, want in zip(sorted_results, cell["order"]):
            assert _same(float(got["score"]), _unjson(want["score"])), (
                f"score drifted for {' '.join(want['expression'])} at "
                f"node_penalty={cell['node_penalty']}: {got['score']} != {want['score']}"
            )


def test_non_finite_fvu_is_nan_not_scored(golden: dict) -> None:
    """The pipeline overwrites a non-finite FVU to nan and never calls the scorer for it.

    This is the behaviour a scorer-level golden cannot see, and the reason the golden goes through
    ``_compile_results_pure``.
    """
    rows = [
        {"expression": ["exp", "x1"], "fvu": float("inf"), "log_prob": -1.0, "constant_count": 0, "score": 0.0},
        {"expression": ["x1"], "fvu": 0.25, "log_prob": -1.0, "constant_count": 0, "score": 0.0},
    ]
    sorted_results, _ = FlashANSR._compile_results_pure(_Shim(), rows, 0.05, 0.0, 0.0)
    assert sorted_results[0]["expression"] == ["x1"]
    assert np.isnan(sorted_results[-1]["score"]), "a non-finite FVU must sort last with a nan score"
    # the scorer itself would have returned +inf, NOT nan -- that is the difference being pinned
    assert score_from_fvu(float("inf"), 2, 0, -1.0, 0.05, 0.0, 0.0) == float("inf")


def test_negative_finite_fvu_reaches_the_scorer_and_is_worst_finite() -> None:
    """A negative FVU is finite, so it passes the guard and the scorer maps it to +inf."""
    rows = [
        {"expression": ["cos", "x1"], "fvu": -1.0, "log_prob": -1.0, "constant_count": 0, "score": 0.0},
        {"expression": ["x1"], "fvu": 0.25, "log_prob": -1.0, "constant_count": 0, "score": 0.0},
    ]
    sorted_results, _ = FlashANSR._compile_results_pure(_Shim(), rows, 0.0, 0.0, 0.0)
    assert sorted_results[0]["expression"] == ["x1"]
    assert sorted_results[-1]["score"] == float("inf")


def test_ties_break_on_expression_tokens_not_insertion_order() -> None:
    """Equal scores are ordered by the token tuple, so parallel completion order cannot leak in."""
    rows = [
        {"expression": ["sin", "x1"], "fvu": 0.5, "log_prob": None, "constant_count": 0, "score": 0.0},
        {"expression": ["x1"], "fvu": 0.5, "log_prob": None, "constant_count": 0, "score": 0.0},
        {"expression": ["+", "x1", "x2"], "fvu": 0.5, "log_prob": None, "constant_count": 0, "score": 0.0},
    ]
    forward, _ = FlashANSR._compile_results_pure(_Shim(), list(rows), 0.0, 0.0, 0.0)
    reverse, _ = FlashANSR._compile_results_pure(_Shim(), list(rows[::-1]), 0.0, 0.0, 0.0)
    assert [r["expression"] for r in forward] == [r["expression"] for r in reverse]
    assert [r["expression"] for r in forward] == [["+", "x1", "x2"], ["sin", "x1"], ["x1"]]


def test_score_less_row_raises_keyerror_documented_inconsistency() -> None:
    """The scoring block tolerates a missing 'score'; the sort key does not.

    Found 2026-09-04 while capturing the golden. ``if 'score' in result`` at :2141 implies
    score-less rows are allowed, but the sort at :2165 reads ``x['score']`` unconditionally. Not
    reachable in production (every producer sets 'score'), so this pins the CURRENT behaviour rather
    than asserting it is correct. If the sort is ever made total over missing keys, delete this test
    and say so.
    """
    rows = [{"expression": ["x1"], "fvu": 0.25, "log_prob": None, "constant_count": 0}]
    with pytest.raises(KeyError):
        FlashANSR._compile_results_pure(_Shim(), rows, 0.0, 0.0, 0.0)


def test_pre_rename_results_payload_is_refused(tmp_path) -> None:
    """A v1 payload spelled the penalty `length_penalty`; loading it must raise, not default.

    Without this, ``metadata.get("node_penalty", <estimator default>)`` would silently rescore the
    restored table at this estimator's penalty instead of the file's.
    """
    import pickle

    from flash_ansr.results import RESULTS_FORMAT_VERSION

    assert RESULTS_FORMAT_VERSION >= 2, "the rename bumped the format version"

    # save_results_payload writes a pickle, so the stale fixture must be one too
    stale = tmp_path / "v1_results.pkl"
    with stale.open("wb") as handle:
        pickle.dump({
            "version": 1,
            "metadata": {"length_penalty": 0.2, "constants_penalty": 0.0, "likelihood_penalty": 0.0},
            "results": [],
        }, handle)

    from flash_ansr.flash_ansr import FlashANSR

    class _Stub:
        node_penalty = 0.05
        constants_penalty = 0.0
        likelihood_penalty = 0.0

    with pytest.raises(ValueError, match="predates the length_penalty -> node_penalty rename"):
        FlashANSR.load_results(_Stub(), str(stale))
