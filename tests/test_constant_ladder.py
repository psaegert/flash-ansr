"""The constant ladder (``flash_ansr.spelling``): the spelling menus, the curvature model, the
re-spelling step through the refinement worker, the ledger rows it adds, and the off-by-default
contract."""
import math

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import flash_ansr as harness
from flash_ansr.inference import build_candidate_ledger
from flash_ansr.scoring import resolve_ranking
from flash_ansr.spelling import (
    ladder_floor,
    ConstantLadderConfig, FitCurvature, SpellingPricer, _convergents, spelling_menu,
)
from flash_ansr.utils.tensor_ops import pad_input_set

N_VARIABLES = 3


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


class TestConfig:
    def test_off_unless_asked(self):
        assert ConstantLadderConfig.from_mapping(None) is None
        assert ConstantLadderConfig.from_mapping(False) is None
        assert ConstantLadderConfig.from_mapping({"enabled": False}) is None
        assert ConstantLadderConfig.from_mapping(True) == ConstantLadderConfig()
        assert ConstantLadderConfig.from_mapping({}) == ConstantLadderConfig()

    def test_overrides_and_validation(self):
        cfg = ConstantLadderConfig.from_mapping({"max_denominator": 64, "digits": [2, 4], "special_constants": ["np.pi"]})
        assert cfg.max_denominator == 64 and cfg.digits == (2, 4) and cfg.special_constants == ("np.pi",)
        assert cfg.to_dict()["digits"] == [2, 4]
        with pytest.raises(ValueError):
            ConstantLadderConfig.from_mapping({"rungs": [1]})
        with pytest.raises(ValueError):
            ConstantLadderConfig.from_mapping({"special_constants": ["tau"]})


class TestMenu:
    def test_convergents_are_the_best_rationals(self):
        assert [str(f) for f in _convergents(math.pi, 1000)] == ["3", "22/7", "333/106", "355/113"]
        assert [str(f) for f in _convergents(0.5, 1000)] == ["0", "1/2"]
        assert [str(f) for f in _convergents(-2.75, 1000)] == ["-3", "-11/4"]

    def test_menu_offers_the_simple_spellings(self):
        cfg = ConstantLadderConfig()
        labels = {s.label for s in spelling_menu(0.5000000001, cfg)}
        assert "/ 1 2" in labels
        labels = {s.label for s in spelling_menu(1 / 137 + 3e-12, cfg)}
        assert "/ 1 137" in labels
        labels = {s.label for s in spelling_menu(2.00000004, cfg)}
        assert "2" in labels
        labels = {s.label for s in spelling_menu(math.pi / 2 + 1e-9, cfg)}
        assert "/ np.pi 2" in labels
        labels = {s.label for s in spelling_menu(1 / (2 * math.pi), cfg)}
        assert "/ 1 * 2 np.pi" in labels
        labels = {s.label for s in spelling_menu(-3.7e-9, cfg)}
        assert "0" in labels

    def test_float_first_and_the_radius_is_honoured(self):
        cfg = ConstantLadderConfig()
        menu = spelling_menu(9.80665, cfg, tolerance=1e-4)
        assert menu[0].kind == "float" and menu[0].value == 9.80665
        assert all(abs(s.value - 9.80665) <= 1e-4 for s in menu[1:])
        assert "10" not in {s.label for s in menu}
        # a unit constant is not within reach of zero, a tiny one is
        assert "0" not in {s.label for s in spelling_menu(1.0000001, cfg)}
        assert "0" in {s.label for s in spelling_menu(1e-7, cfg)}

    def test_surprise_rule_keeps_real_fractions_and_drops_disguised_floats(self):
        from flash_ansr.spelling import fraction_surprise
        from fractions import Fraction
        # an arbitrary number: its convergent 769/404 is only generically close (next quotient ~ 3)
        arbitrary = 769 / 404 + 2e-6
        assert fraction_surprise(arbitrary, Fraction(769, 404)) < 404
        # a number that IS the fraction, fitted to 1e-12: the surprise is ~ 1 / (1e-12 * 137^2)
        assert fraction_surprise(1 / 137 + 3e-12, Fraction(1, 137)) > 1e6
        assert fraction_surprise(0.5, Fraction(1, 2)) == math.inf
        off = ConstantLadderConfig(fraction_surprise=None)
        rule = ConstantLadderConfig(fraction_surprise="denominator")
        assert "/ 769 404" in {s.label for s in spelling_menu(arbitrary, off, tolerance=1e-3)}
        assert "/ 769 404" not in {s.label for s in spelling_menu(arbitrary, rule, tolerance=1e-3)}
        kept = {s.label for s in spelling_menu(1 / 137 + 3e-12, rule)}
        assert "/ 1 137" in kept
        assert "/ 1 2" in {s.label for s in spelling_menu(0.5000000001, rule)}
        assert "/ 3 2" in {s.label for s in spelling_menu(1.5 + 2e-9, rule)}
        assert "2" in {s.label for s in spelling_menu(2.0004, rule, tolerance=1e-3)}   # integers are never filtered
        # the special families go through the same test on x / s
        assert "/ np.pi 2" in {s.label for s in spelling_menu(math.pi / 2 + 1e-9, rule)}
        fixed = ConstantLadderConfig(fraction_surprise=32)
        assert "/ 769 404" not in {s.label for s in spelling_menu(arbitrary, fixed, tolerance=1e-3)}
        assert ConstantLadderConfig.from_mapping({"fraction_surprise": "denominator"}).fraction_surprise == "denominator"
        assert ConstantLadderConfig.from_mapping({"fraction_surprise": 32}).to_dict()["fraction_surprise"] == 32
        with pytest.raises(ValueError):
            ConstantLadderConfig.from_mapping({"fraction_surprise": "always"})
        with pytest.raises(ValueError):
            ConstantLadderConfig.from_mapping({"fraction_surprise": -1})

    def test_prices_are_cheaper_for_simpler_spellings(self, engine):
        pricer = SpellingPricer(engine)
        assert pricer.price(("2",)) < pricer.price(("2.0000005208229443",))
        assert pricer.price(("/", "1", "137")) < pricer.price((repr(1 / 137),))
        assert pricer.price(("np.pi",)) < pricer.price((repr(math.pi),))


class TestCurvature:
    def test_linear_model_stiffness_is_the_schur_complement(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, 64)
        predict = lambda c: c[0] * x + c[1]
        curv = FitCurvature.build(predict, np.array([1.3, -0.2]))
        A = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), 64.0]])
        assert np.allclose(curv.A, A, rtol=1e-6)
        expected = A[0, 0] - A[0, 1] ** 2 / A[1, 1]
        assert np.isclose(curv.stiffness([0])[0, 0], expected, rtol=1e-6)
        assert np.allclose(curv.stiffness([0, 1]), A, rtol=1e-6)

    def test_redundant_pair_is_free(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-2, 2, 64)
        predict = lambda c: c[0] * c[1] * x
        curv = FitCurvature.build(predict, np.array([2.0, 3.0]))
        assert curv.stiffness([0])[0, 0] == 0.0 and curv.stiffness([1])[0, 0] == 0.0
        assert curv.stiffness([0, 1])[0, 0] > 0


def _payload(engine, expression, X, y, *, ladder, p0=None, log_prob=-1.0):
    return {
        "simplipy_engine": engine, "X": X, "y": y.reshape(-1, 1), "n_variables": N_VARIABLES,
        "expression": list(expression), "raw_beam": [7, 8, 9], "beam": [7, 8, 9], "raw_beam_decoded": " ".join(expression),
        "log_prob": log_prob, "constant_count": sum(t == "<constant>" for t in expression), "p0": p0,
        "n_restarts": 4, "method": "curve_fit_lm", "p0_noise": "normal", "p0_noise_kwargs": None,
        "refine_scope": "fittable", "constant_ladder": ladder, "converge_error": "ignore", "numpy_errors": "ignore",
        "y_variance": float(np.var(y)), "ranking_weights": resolve_ranking("mdl", mdl_strength=1e-2).effective_weights,
        "complexity": None, "seed": 0,
    }


def _data(seed=0, n=128):
    rng = np.random.default_rng(seed)
    X = pad_input_set(rng.uniform(0.5, 3.0, size=(n, 1)), N_VARIABLES)
    return X


class TestWorker:
    def test_off_by_default(self, engine):
        X = _data(); y = 2.0 * X[:, 0]
        result, warning = harness._refine_candidate_worker(_payload(engine, ["*", "<constant>", "x1"], X, y, ladder=None, p0=[2.0]))
        assert warning is None and result is not None
        assert result["spelling"] is None and result["respelled"] is None

    def test_an_exact_scale_becomes_a_literal(self, engine):
        X = _data(); y = 2.0 * X[:, 0]
        result, _ = harness._refine_candidate_worker(_payload(engine, ["*", "<constant>", "x1"], X, y, ladder=True, p0=[2.0]))
        child = result["respelled"]
        assert child is not None
        assert child["expression"] == ["*", "2", "x1"]
        assert child["spelling"] == "c0=2" and child["constant_count"] == 0 and child["refine_scope"] == "placeholders"
        # the pricer already priced the exact 2.0 as `2`, so this is a tie: the canonical spelling
        # takes the parent's place in the pool
        assert child["score"] <= result["score"] and child["mdl"] <= result["mdl"]
        assert child["replaces_parent"] is True
        assert child["fvu"] <= max(result["fvu"], np.finfo(np.float64).eps)
        assert child["raw_beam"] == result["raw_beam"]   # the parent's beam, provenance

    def test_a_unit_scale_disappears(self, engine):
        X = _data(); y = X[:, 0].copy()
        result, _ = harness._refine_candidate_worker(_payload(engine, ["*", "<constant>", "x1"], X, y, ladder=True, p0=[1.0]))
        child = result["respelled"]
        assert child is not None and child["expression"] == ["x1"] and child["constant_count"] == 0

    def test_a_strict_improvement_stands_beside_its_parent(self, engine):
        # a whisper of noise: the fit lands near 2, the data cannot tell 2.00000003 from 2, and the
        # integer is cheaper -- a strict improvement, so the variant stands beside its parent
        X = _data(); rng = np.random.default_rng(3)
        y = 2.0 * X[:, 0] * (1.0 + 1e-7 * rng.standard_normal(X.shape[0]))
        result, _ = harness._refine_candidate_worker(_payload(engine, ["*", "<constant>", "x1"], X, y, ladder=True, p0=[2.0]))
        child = result["respelled"]
        assert child is not None and child["expression"] == ["*", "2", "x1"]
        assert child["score"] < result["score"] and child["replaces_parent"] is False

    def test_a_constant_only_variant_of_an_informative_parent_is_not_offered(self, engine):
        X = _data(); y = 0.3 * X[:, 0] + 5.0 + 0.05 * np.sin(7 * X[:, 0])
        result, _ = harness._refine_candidate_worker(_payload(engine, ["+", "*", "<constant>", "x1", "<constant>"], X, y, ladder=True, p0=[0.3, 5.0]))
        child = result["respelled"]
        assert child is None or any(t.startswith("x") for t in child["expression"])

    def test_a_half_exponent_becomes_a_root(self, engine):
        X = _data(); y = np.sqrt(X[:, 0])
        result, _ = harness._refine_candidate_worker(_payload(engine, ["pow", "x1", "<constant>"], X, y, ladder=True, p0=[0.5]))
        child = result["respelled"]
        assert child is not None
        assert "rootn" in child["expression"] or child["expression"] == ["pow", "x1", "/", "1", "2"]

    def test_a_precise_constant_stays_a_float(self, engine):
        X = _data(); y = 9.80665 * X[:, 0]
        result, _ = harness._refine_candidate_worker(_payload(engine, ["*", "<constant>", "x1"], X, y, ladder=True, p0=[9.80665]))
        assert result["respelled"] is None

    def test_a_tiny_coefficient_kills_its_term(self, engine):
        X = _data(); y = 3.0 * X[:, 0]
        expression = ["+", "*", "<constant>", "x1", "*", "<constant>", "pow", "x1", "2"]
        result, _ = harness._refine_candidate_worker(_payload(engine, expression, X, y, ladder=True, p0=[3.0, 0.0]))
        child = result["respelled"]
        assert child is not None
        assert child["expression"] == ["*", "3", "x1"], child["expression"]

    def test_pi_is_recognised(self, engine):
        X = _data(); y = math.pi * X[:, 0]
        result, _ = harness._refine_candidate_worker(_payload(engine, ["*", "<constant>", "x1"], X, y, ladder=True, p0=[math.pi]))
        child = result["respelled"]
        assert child is not None and "np.pi" in child["expression"]


class TestPool:
    def test_variant_gets_its_own_ledger_row_with_a_parent(self):
        parent = {"raw_beam": [1, 2, 3], "fvu": 1e-3, "log_prob": -1.0, "fits": [(np.array([2.0000001]), None, 1e-3)],
                  "expression": ["*", "<constant>", "x1"], "constant_count": 1, "mdl": 62000.0, "score": -2.4, "spelling": None}
        child = dict(parent, expression=["*", "2", "x1"], constant_count=0, mdl=11585.0, score=-2.9, spelling="c0=2", fits=[(np.array([]), None, 1e-3)])
        ledger = build_candidate_ledger([[1, 2, 3], [4, 5]], [-1.0, -2.0], [child, parent])
        assert len(ledger) == 3
        assert ledger.spelling == ["", "", "c0=2"] and ledger.parent == [-1, -1, 0]
        assert ledger.token_lists[2] == [1, 2, 3] and ledger.result_index[2] == 0 and ledger.result_index[0] == 1

    def test_duplicate_variants_collapse_and_drawn_expressions_win(self):
        rows = [
            {"expression": ["x1"], "score": -1.0, "spelling": None},
            {"expression": ["*", "2", "x1"], "score": -2.0, "spelling": "c0=2"},
            {"expression": ["*", "2", "x1"], "score": -2.5, "spelling": "c1=2"},
            {"expression": ["x1"], "score": -3.0, "spelling": "c0=1"},
        ]
        kept = harness.FlashANSR._dedup_respelled(rows)
        assert [r.get("spelling") for r in kept] == [None, "c1=2"]


class TestPositions:
    def test_slot_positions_walk_the_prefix_tree(self):
        from flash_ansr.spelling import ZERO_POSITIONS, slot_positions
        arity = {"+": 2, "*": 2, "pow": 2, "log": 1, "/": 2}
        expr = ["+", "*", "<constant>", "x1", "pow", "<constant>", "<constant>"]
        pos = slot_positions(expr, arity)
        assert pos[0] == (None, 0) and pos[2] == ("*", 0) and pos[3] == ("*", 1)
        assert pos[5] == ("pow", 0) and pos[6] == ("pow", 1)
        assert ("*", 0) in ZERO_POSITIONS and ("pow", 1) in ZERO_POSITIONS
        assert ("pow", 0) not in ZERO_POSITIONS and ("/", 1) not in ZERO_POSITIONS and ("log", 0) not in ZERO_POSITIONS

    def test_zero_is_not_offered_as_a_pow_base(self, engine):
        # y = 3 x1: the pow base is a dead subtree with a free exponent; zeroing it would make the
        # fit's arithmetic and the pricer's folding disagree, so zero is never offered there
        X = _data(); y = 3.0 * X[:, 0]
        expression = ["+", "*", "<constant>", "x1", "*", "<constant>", "pow", "<constant>", "<constant>"]
        result, _ = harness._refine_candidate_worker(_payload(engine, expression, X, y, ladder=True, p0=[3.0, 1e-9, 0.2, 6.0]))
        child = result["respelled"]
        if child is not None:
            assert "pow 0" not in " ".join(child["expression"])
            assert child["mdl"] >= 6000.0


class TestPoolBound:
    """The ladder as a post-fit pass: only candidates that can still reach rank 0 are re-spelled, and
    the returned rank 0 is the one the exhaustive pass returns."""

    def _stub(self, engine, ladder, counter):
        from types import SimpleNamespace
        stub = SimpleNamespace(
            refiner_workers=1, _refine_pool=None, _overlap_mode=False, simplipy_engine=engine, n_variables=N_VARIABLES,
            n_restarts=4, refiner_method="curve_fit_lm", refiner_p0_noise="normal", refiner_p0_noise_kwargs=None,
            numpy_errors="ignore", ranking=SimpleNamespace(effective_weights=resolve_ranking("mdl", mdl_strength=1e-2).effective_weights),
            constant_ladder=ladder, refiner_scope="fittable", _count_constants=harness.FlashANSR._count_constants, close=lambda: None)
        stub._create_result_entry = lambda **kw: harness.FlashANSR._create_result_entry(stub, **kw)

        def run_ordered(jobs, worker, gs, **kw):
            counter.append(len(jobs))
            return harness.FlashANSR._run_ordered_jobs(stub, jobs, worker, gs, **kw)
        stub._run_ordered_jobs = run_ordered
        return stub

    def test_bound_skips_hopeless_candidates_and_keeps_rank_zero(self, engine):
        from types import SimpleNamespace
        rng = np.random.default_rng(5)
        X = pad_input_set(rng.uniform(0.5, 3.0, size=(64, 1)), N_VARIABLES)
        y = 2.0 * X[:, 0]
        exprs = [["*", "<constant>", "x1"], ["*", "<constant>", "pow", "x1", "2"], ["+", "*", "<constant>", "x1", "<constant>"]]
        bounded_cfg = ConstantLadderConfig(pool_bound=True)
        exhaustive_cfg = ConstantLadderConfig(pool_bound=False)
        # the fits alone (what the fit phase produces under the bound)
        fitted = [harness._refine_candidate_worker(_payload(engine, e, X, y, ladder=None))[0] for e in exprs]
        assert all(r is not None for r in fitted)
        counter: list[int] = []
        stub = self._stub(engine, bounded_cfg, counter)
        results = [stub._create_result_entry(payload=r, input_dim=N_VARIABLES) for r in fitted]
        gs = SimpleNamespace(X_np=X, y_np=y.reshape(-1, 1), y_variance=float(np.var(y)))
        harness.FlashANSR._run_bounded_ladder(stub, results, gs, input_dim=N_VARIABLES, converge_error="ignore", refine_seed=0, verbose=False)
        # the exact and the constant-free candidates are tried, the hopeless quadratic is not
        assert sum(counter) == 2
        best = min(results, key=lambda r: r["score"])
        assert best["expression"] == ["*", "2", "x1"] and best["spelling"]
        # the quadratic is still in the pool, as fitted, without a variant
        quadratic = [r for r in results if r["expression"][:2] == ["*", "<constant>"] and "pow" in r["expression"]]
        assert len(quadratic) == 1 and not quadratic[0]["spelling"]
        # the exhaustive pass (the ladder inside the fit worker) returns the same rank 0
        exhaustive = []
        for e in exprs:
            r = harness._refine_candidate_worker(_payload(engine, e, X, y, ladder=exhaustive_cfg))[0]
            child = r.get("respelled")
            exhaustive += [child] if (child is not None and child.get("replaces_parent")) else [x for x in (r, child) if x is not None]
        ref = min(exhaustive, key=lambda r: r["score"])
        assert ref["expression"] == best["expression"] and abs(ref["score"] - best["score"]) < 1e-9

    def test_floor_is_the_loosest_score(self):
        weights = resolve_ranking("mdl", mdl_strength=1e-2).effective_weights
        from flash_ansr.scoring import score_row
        cfg = ConstantLadderConfig(max_decades=1.0)
        entry = {"fvu": 1e-4, "log_prob": -3.0, "expression": ["*", "<constant>", "x1"], "constant_count": 1, "mdl": 70000.0}
        floor = ladder_floor(entry, weights, cfg, score_row)
        assert floor < score_row(entry, weights)
        assert abs(floor - score_row({"fvu": 1e-5, "expression": [], "constant_count": 0, "log_prob": -3.0, "mdl": 0.0}, weights)) < 1e-12
        assert ladder_floor({"fvu": float("nan")}, weights, cfg, score_row) == math.inf
