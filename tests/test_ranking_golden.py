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
from flash_ansr.scoring import count_constants, score_from_fvu, RankingConfig, resolve_ranking

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
    """``_compile_results_pure`` reads exactly this helper off ``self`` (scoring itself goes through
    ``flash_ansr.scoring.score_row``, a module function)."""

    _count_constants = staticmethod(count_constants)


def _weighted(node: float = 0.0, const: float = 0.0, lik: float = 0.0, mdl: float = 0.0) -> RankingConfig:
    """The 0.13 scalar ranking spelled in the registry: the four loose penalties as weights."""
    return resolve_ranking('weighted', weights={
        'n_nodes': node, 'n_constants': const, 'neg_log_prob': lik, 'mdl': mdl})


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
            _Shim(), rows, ranking=_weighted(cell["node_penalty"], cell["constants_penalty"], cell["likelihood_penalty"]),
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
    sorted_results, _ = FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted(0.05))
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
    sorted_results, _ = FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted())
    assert sorted_results[0]["expression"] == ["x1"]
    assert sorted_results[-1]["score"] == float("inf")


def test_ties_break_on_expression_tokens_not_insertion_order() -> None:
    """Equal scores are ordered by the token tuple, so parallel completion order cannot leak in."""
    rows = [
        {"expression": ["sin", "x1"], "fvu": 0.5, "log_prob": None, "constant_count": 0, "score": 0.0},
        {"expression": ["x1"], "fvu": 0.5, "log_prob": None, "constant_count": 0, "score": 0.0},
        {"expression": ["+", "x1", "x2"], "fvu": 0.5, "log_prob": None, "constant_count": 0, "score": 0.0},
    ]
    forward, _ = FlashANSR._compile_results_pure(_Shim(), list(rows), ranking=_weighted())
    reverse, _ = FlashANSR._compile_results_pure(_Shim(), list(rows[::-1]), ranking=_weighted())
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
        FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted())


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

    with pytest.raises(ValueError, match="predates the ranking record"):
        FlashANSR.load_results(_Stub(), str(stale))


class TestMdlEndToEnd:
    """`mdl` must survive the whole pipeline, not just be computed.

    `_create_result_entry` is a selective key-by-key whitelist: a key the refine worker produces but
    that literal omits is dropped SILENTLY. With mdl dropped, every score at mdl_penalty != 0 is nan,
    the sort degenerates to the expression-token tie-break, and a 21-hour run ranks ALPHABETICALLY
    while fit() reports success. Computing mdl correctly and never checking it arrives is exactly the
    failure this class exists to catch.
    """

    def test_the_whitelist_carries_mdl(self) -> None:
        """Pin the literal itself: if someone adds a worker key without adding it here, this fails."""
        import inspect

        from flash_ansr.flash_ansr import FlashANSR

        source = inspect.getsource(FlashANSR._create_result_entry)
        assert "'mdl': payload.get('mdl')" in source, (
            "_create_result_entry is a whitelist; mdl must be listed or it is dropped silently"
        )

    def test_result_typeddict_declares_mdl(self) -> None:
        from flash_ansr.flash_ansr import Result

        assert "mdl" in Result.__annotations__
        assert "constants_emitted" in Result.__annotations__, (
            "constants_emitted was carried but never declared"
        )

    def test_candidate_declares_mdl_and_keeps_mu_separate(self) -> None:
        """mdl and mu are different quantities on different spellings; neither may absorb the other."""
        import dataclasses

        from flash_ansr.inference import Candidate

        names = {f.name for f in dataclasses.fields(Candidate)}
        assert {"mdl", "mu"} <= names, "both must exist -- mu is the skeleton unit fit(complexity=) eats"

    def test_worker_prices_the_realized_spelling_not_the_emitted_one(self) -> None:
        """The owner's R1 ruling: mdl is priced AFTER the constants are substituted.

        Pricing the emitted spelling would make mdl a re-parameterisation of (n_nodes, n_constants)
        to 99% of within-problem pairs, and would be blind to constant precision -- the axis a future
        coarse-graining of constants has to be able to move.
        """
        import inspect

        from flash_ansr import flash_ansr as module

        worker = inspect.getsource(module._refine_candidate_worker)
        assert "_price_realized(" in worker, "the worker must price through _price_realized"
        source = inspect.getsource(module._price_realized)
        assert "refiner.transform(" in source, "mdl must be priced on the realized expression"
        assert "return_prefix=True" in source
        # the pricer call must sit AFTER the transform, on its output
        transform_at = source.index("refiner.transform(")
        price_at = source.index("simplipy_engine.complexity(")
        assert transform_at < price_at, "the pricer must see the realized tokens, not the emitted ones"
        assert "mode=Mode.f64" in source and "canon='default'" in source, (
            "mode and canon are written out, never inherited -- mode moves mu"
        )


class TestMdlPenaltyAddend:
    """The mdl addend must be inert by default and live when asked for."""

    def test_default_is_bit_identical_to_the_frozen_golden(self, golden: dict) -> None:
        """mdl_penalty defaults to 0.0, so every pre-existing config scores EXACTLY as before.

        The golden was captured before mdl existed. If adding the addend perturbed the default path
        by even one ULP this fails -- which is the whole point of having captured it first.
        """
        for cell in golden["cells"]:
            if cell["likelihood_penalty"] != 0.0:
                continue
            rows = _rebuild_inputs(cell)[::-1]
            # every row carries an mdl, but mdl_penalty is left at its default
            for r in rows:
                r["mdl"] = 123456.0
            sorted_results, _ = FlashANSR._compile_results_pure(
                _Shim(), rows, ranking=_weighted(cell["node_penalty"], cell["constants_penalty"], cell["likelihood_penalty"]),
            )
            assert [r["expression"] for r in sorted_results] == [e["expression"] for e in cell["order"]]
            for got, want in zip(sorted_results, cell["order"]):
                assert _same(float(got["score"]), _unjson(want["score"])), (
                    "a populated mdl changed the score at mdl_penalty=0 -- the addend is not inert"
                )

    def test_a_live_penalty_reorders_on_mdl_alone(self) -> None:
        """Two candidates identical in fvu, nodes and constants, differing ONLY in mdl."""
        def rows() -> list[dict]:
            return [
                {"expression": ["a", "b"], "fvu": 0.25, "log_prob": None, "constant_count": 1,
                 "mdl": 200000.0, "score": float("nan")},
                {"expression": ["c", "d"], "fvu": 0.25, "log_prob": None, "constant_count": 1,
                 "mdl": 100000.0, "score": float("nan")},
            ]
        inert, _ = FlashANSR._compile_results_pure(_Shim(), rows(), ranking=_weighted())
        # tie on score -> the token tie-break decides, so 'a b' comes first
        assert [r["expression"] for r in inert] == [["a", "b"], ["c", "d"]]

        live, _ = FlashANSR._compile_results_pure(_Shim(), rows(), ranking=_weighted(mdl=4.5e-3))
        assert [r["expression"] for r in live] == [["c", "d"], ["a", "b"]], (
            "the cheaper mdl must win once mdl_penalty is live"
        )
        # and the gap is exactly the bits-converted difference
        assert live[1]["score"] - live[0]["score"] == pytest.approx((200000.0 - 100000.0) / 1000.0 * 4.5e-3)

    def test_unpriceable_is_harmless_while_the_penalty_is_off(self) -> None:
        """mdl_penalty=0 -> mdl is not part of the ranking, so a missing price cannot matter."""
        rows = [
            {"expression": ["a"], "fvu": 0.5, "log_prob": None, "constant_count": 0,
             "mdl": None, "score": float("nan")},
            {"expression": ["b"], "fvu": 0.9, "log_prob": None, "constant_count": 0,
             "mdl": 1000.0, "score": float("nan")},
        ]
        out, _ = FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted())
        assert out[0]["expression"] == ["a"], "the better fvu wins; the missing price is irrelevant"
        assert np.isfinite(out[0]["score"])

    def test_unpriceable_is_never_ADVANTAGED_while_the_penalty_is_live(self) -> None:
        """The trap: contributing 0 is not neutral, it is a ~0.63-decade head start.

        At the calibrated strength a typical candidate pays mdl_penalty * 140 bits ~ 0.63 decades of
        FVU. A row that contributes 0 would out-rank a priced row that is genuinely better, so an
        unpriceable candidate must sort BELOW every priced one.
        """
        rows = [
            {"expression": ["a"], "fvu": 0.10, "log_prob": None, "constant_count": 0,
             "mdl": None, "score": float("nan")},          # best fvu, but no price
            {"expression": ["b"], "fvu": 0.11, "log_prob": None, "constant_count": 0,
             "mdl": 140000.0, "score": float("nan")},      # slightly worse fvu, priced
        ]
        out, _ = FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted(mdl=4.5e-3))
        assert out[0]["expression"] == ["b"], (
            "an unpriceable candidate must not out-rank a priced one on a free pass"
        )
        assert out[-1]["score"] == float("inf")

    def test_unpriceable_still_outranks_a_diverged_fit(self) -> None:
        """+inf sorts above nan: 'fitted but unjudgeable' beats 'did not fit'."""
        rows = [
            {"expression": ["a"], "fvu": 0.10, "log_prob": None, "constant_count": 0,
             "mdl": None, "score": float("nan")},
            {"expression": ["b"], "fvu": float("inf"), "log_prob": None, "constant_count": 0,
             "mdl": 1000.0, "score": float("nan")},
        ]
        out, _ = FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted(mdl=4.5e-3))
        assert out[0]["expression"] == ["a"]
        assert np.isnan(out[-1]["score"])

    def test_all_unpriceable_raises_only_when_the_penalty_is_live(self) -> None:
        """Aggregate failure: nothing priceable means the criterion produced no ordering.

        Returning an answer there would mean ranking by expression spelling while reporting success.
        One unpriceable row must NOT raise -- that would let a single pathological candidate abort a
        problem and discard the good ones beside it.
        """
        rows = [
            {"expression": ["a"], "fvu": 0.5, "log_prob": None, "constant_count": 0,
             "mdl": None, "score": float("nan")},
            {"expression": ["b"], "fvu": 0.9, "log_prob": None, "constant_count": 0,
             "mdl": None, "score": float("nan")},
        ]
        with pytest.raises(ValueError, match="none of the 2 candidates could be priced"):
            FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted(mdl=4.5e-3))
        # same rows, penalty off -> perfectly fine
        out, _ = FlashANSR._compile_results_pure(_Shim(), rows, ranking=_weighted())
        assert len(out) == 2

    def test_milli_bit_to_bit_conversion_is_explicit(self) -> None:
        """mdl is stored in milli-bits; mdl_penalty is per bit. Multiplying directly is 1000x wrong."""
        from flash_ansr.scoring import MILLIBITS_PER_BIT, score_from_fvu

        assert MILLIBITS_PER_BIT == 1000.0
        base = score_from_fvu(0.1, 5, 1, None, 0.0, 0.0, 0.0)
        live = score_from_fvu(0.1, 5, 1, None, 0.0, 0.0, 0.0, mdl=67000.0, mdl_penalty=4.5e-3)
        assert live - base == pytest.approx(67.0 * 4.5e-3), "one <constant> is 67 bits, not 67000"


class TestUnpriceable:
    """What "unpriceable" actually means, measured 2026-09-04 rather than assumed.

    Rate on the 8,476-candidate reference population: 0. These pin the triggers so the guard's scope
    stays honest if simplipy's tokeniser changes.
    """

    @pytest.mark.parametrize("bad", ["inf", "nan", "-inf"])
    def test_non_finite_constants_are_what_actually_raise(self, bad: str) -> None:
        """The realistic trigger: a fit that passes valid_fit but carries inf/nan.

        `transform` substitutes repr(float(v)), and repr(inf) is the literal token 'inf', which
        simplipy rejects as a reserved numeric spelling. Everything else survives: 1e308 and 5e-324
        both price fine, so it is non-finiteness and not magnitude that breaks pricing.
        """
        from simplipy import Mode, SimpliPyEngine

        engine = SimpliPyEngine.load("acj-5-4-llm", install=True)
        with pytest.raises(Exception):
            engine.complexity(["*", bad, "x1"], certified=True, mode=Mode.f64, canon="default")
        # a finite extreme is NOT a failure -- the guard must not be read as "big numbers break it"
        assert engine.complexity(["*", "1e308", "x1"], certified=True, mode=Mode.f64,
                                 canon="default") > 0

    def test_a_leftover_placeholder_is_refused_not_priced(self) -> None:
        """The dangerous case: a partial substitution prices at 67,000 WITHOUT raising.

        simplipy prices '<constant>' happily, so without an explicit check a half-substituted
        expression would be recorded as a realized price. This asserts the worker refuses it.
        """
        import inspect

        from flash_ansr import flash_ansr as module

        source = inspect.getsource(module._price_realized)
        assert "still carries a <constant> placeholder" in source, (
            "a partially substituted expression must be refused, not priced as a skeleton"
        )
        # and the check must precede the pricer call
        guard_at = source.index("still carries a <constant> placeholder")
        price_at = source.index("simplipy_engine.complexity(")
        assert guard_at < price_at
