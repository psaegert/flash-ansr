"""PySR's hall of fame joins the Flash-ANSR candidates and Flash-ANSR's sorting picks rank 0: the same
score, the same MDL penalty, for every candidate; PySR's own choice is kept beside it for reference."""
import numpy as np
import pytest
from flash_ansr.scoring import count_constants, score_row
from simplipy.engine import Mode

from flash_ansr.hybrid.bridge import evaluate_prefix
from flash_ansr.hybrid.candidates import pick_prediction, prediction_fvu, pysr_candidates, rank_candidates


def _problem():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, size=(64, 2)); x_val = rng.uniform(0.5, 2.0, size=(16, 2))
    law = lambda a: 2.0 * a[:, 0] ** 2 + 1.0  # noqa: E731
    return x, law(x), x_val, law(x_val)


def _flash(engine, prefix, x, y, x_val, weights):
    """A Flash-ANSR candidate dict as a snapshot stores it, scored the way the refine worker scores."""
    yp, ypv = evaluate_prefix(engine, list(prefix), ["x1", "x2"], x, x_val)
    f = prediction_fvu(y, np.asarray(yp).reshape(-1))
    mdl = float(engine.complexity(list(prefix), certified=True, mode=Mode.f64, canon="default"))
    return {"expression_prefix": list(prefix), "skeleton_prefix": list(prefix), "expression_infix": " ".join(prefix),
            "constants": [], "fvu": f, "mdl": mdl, "n_nodes": len(prefix), "log_prob": -1.0, "pareto_rank": -1,
            "score": score_row({"fvu": f, "expression": prefix, "constant_count": count_constants(prefix), "log_prob": -1.0, "mdl": mdl}, weights),
            "y_pred": np.asarray(yp).reshape(-1), "y_pred_val": np.asarray(ypv).reshape(-1)}


class TestPySRAddsCandidates:
    def test_the_hall_of_fame_law_beats_a_worse_flash_answer(self, engine):
        x, y, x_val, y_val = _problem()
        weights = {"mdl": 1e-2}
        flash = _flash(engine, ["*", "2.1", "pow", "x1", "2"], x, y, x_val, weights)   # close, not exact
        hof = [{"complexity": 1, "loss": 9.0, "score": 0.0, "equation": "v1"},
               {"complexity": 7, "loss": 0.0, "score": 1.0, "equation": "(2.0 * (v1 ^ 2)) + 1.0"},
               {"complexity": 9, "loss": 0.0, "score": 0.1, "equation": "((2.0 * (v1 ^ 2)) + 1.0) + (0.0 * v2)"}]
        values = {"predicted_expression": "v1", "predicted_expression_prefix": ["x1"], "prediction_success": True, "equations": hof}
        winner = pick_prediction(values, flash=[flash], equations=hof, engine=engine, weights=weights,
                                 x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"])
        assert winner is not None and values["predicted_source"] == "pysr" and values["predicted_hof_index"] == 1
        assert values["pysr_expression"] == "v1"                                  # PySR's own pick is kept
        assert "**" not in values["predicted_expression_prefix"] and "pow" in values["predicted_expression_prefix"]
        np.testing.assert_allclose(values["y_pred"].reshape(-1), y, rtol=1e-12)
        np.testing.assert_allclose(values["y_pred_val"].reshape(-1), y_val, rtol=1e-12)
        assert values["predicted_mdl"] is not None and values["prediction_success"] is True
        scores = [e["score"] for e in values["hybrid_candidates"]]
        assert scores == sorted(scores) and len(scores) == 4                         # 1 Flash-ANSR + 3 PySR candidates, ranked
        # the exact law with a free 0 * v2 tail pays the MDL penalty and loses to the plain law
        assert [e["hof_index"] for e in values["hybrid_candidates"]][:2] == [1, 2]

    def test_flash_keeps_the_answer_when_it_scores_better(self, engine):
        x, y, x_val, y_val = _problem()
        weights = {"mdl": 1e-2}
        flash = _flash(engine, ["+", "*", "2", "pow", "x1", "2", "1"], x, y, x_val, weights)   # exact and short
        hof = [{"equation": "v1"}, {"equation": "((2.0 * (v1 ^ 2)) + 1.0) + (0.0 * v2)"}, {"equation": "not an expression ("}]
        values = {"predicted_expression": "v1", "predicted_expression_prefix": ["x1"], "prediction_success": True}
        winner = pick_prediction(values, flash=[flash], equations=hof, engine=engine, weights=weights,
                                 x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"])
        assert winner is not None and values["predicted_source"] == "flash-ansr"
        assert values["predicted_expression_prefix"] == ["+", "*", "2", "pow", "x1", "2", "1"]
        assert len(values["hybrid_candidates"]) == 3                                # the unreadable entry is dropped

    def test_a_failed_gp_stage_falls_back_on_flash(self, engine):
        x, y, x_val, y_val = _problem()
        weights = {"mdl": 1e-2}
        flash = _flash(engine, ["+", "*", "2", "pow", "x1", "2", "1"], x, y, x_val, weights)
        values = {"prediction_success": False, "error": "RuntimeError: Julia died"}
        pick_prediction(values, flash=[flash], equations=None, engine=engine, weights=weights,
                        x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"])
        assert values["predicted_source"] == "flash-ansr" and values["prediction_success"] is True
        assert values["pysr_error"].startswith("RuntimeError") and values["error"] is None

    def test_nothing_priceable_keeps_pysr_answer(self, engine):
        x, y, x_val, y_val = _problem()
        values = {"predicted_expression": "v1", "predicted_expression_prefix": ["x1"], "prediction_success": True}
        assert pick_prediction(values, flash=[], equations=[{"equation": "("}], engine=engine, weights={"mdl": 1e-2},
                               x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"]) is None
        assert values["predicted_source"] == "pysr" and values["predicted_expression"] == "v1"
        assert values["hybrid_ranking_error"] and values["hybrid_candidates"] == []

    def test_pysr_candidates_go_through_the_ladder(self, engine):
        """A PySR entry with almost-round constants is re-fitted and re-spelled like a Flash-ANSR
        candidate: the pool entry carries integer constants, a lower MDL and a spelling record."""
        x, y, x_val, y_val = _problem()
        weights = {"mdl": 1e-2}
        rows = pysr_candidates([{"equation": "(2.0000001 * (v1 ^ 2)) + 0.9999999"}], engine=engine, weights=weights,
                               x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"])
        assert rows, "the entry must enter the pool"
        best = min(rows, key=lambda r: r["score"])
        assert best["spelling"], "the ladder must have re-spelled the almost-round constants"
        assert all(float(c).is_integer() for c in best["constants"]) and best["constants"]
        assert "**" not in best["expression_prefix"]
        plain = pysr_candidates([{"equation": "(2.0000001 * (v1 ^ 2)) + 0.9999999"}], engine=engine, weights=weights,
                                x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"], refine={"constant_ladder": None})
        assert len(plain) == 1 and plain[0]["spelling"] is None and best["mdl"] < plain[0]["mdl"]

    def test_rank_order_is_score_then_tokens(self):
        a = {"expression_prefix": ["x1"], "skeleton_prefix": ["x1"], "score": 1.0}
        b = {"expression_prefix": ["x2"], "skeleton_prefix": ["x2"], "score": 1.0}
        c = {"source": "pysr", "hof_index": 0, "expression_prefix": ["x1"], "score": float("nan")}
        ranked = rank_candidates([b, a], [c])
        assert [e["expression_prefix"] for e in ranked] == [["x1"], ["x2"], ["x1"]]   # ties on the tokens; NaN last
        assert ranked[0]["source"] == "flash-ansr" and ranked[-1]["source"] == "pysr"


class TestPredictionFVU:
    def test_definition(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert prediction_fvu(y, y) == 0.0
        assert prediction_fvu(y, np.full(4, y.mean())) == pytest.approx(1.0)
        assert prediction_fvu(y, np.array([1.0, np.nan, 3.0, 4.0])) == float("inf")
        assert prediction_fvu(y, np.ones(3)) == float("inf")                      # a curve of the wrong length


class TestCanonicalEmission:
    def test_a_factor_the_fit_cancels_is_dropped(self, engine):
        """PySR's `exp(-v1^2) * tanh(v1)^2 / tanh(v1)^2` joins the pool as `exp(-x1^2)`, the form it is priced as."""
        x, y, x_val, y_val = _problem()
        y = np.exp(-x[:, 0] ** 2)
        rows = pysr_candidates([{"equation": "exp(-(v1^2)) * tanh(v1)^2 / tanh(v1)^2"}], engine=engine, weights={"mdl": 1e-2},
                               x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"], refine={"constant_ladder": None})
        assert rows and rows[0]["expression_prefix"] == ["exp", "neg", "pow", "x1", "2"]
        assert rows[0]["fvu"] < 1e-20 and rows[0]["n_nodes"] == 5
