"""HybridRegressor.fit: the estimator's contract -- one problem in, a HybridFitResult (a FitResult) out."""
from types import SimpleNamespace

import numpy as np
import pytest

import flash_ansr.hybrid.regressor as regressor_module
from flash_ansr.hybrid import HybridConfig, HybridFitResult, HybridRegressor, PySRSettings
from flash_ansr.inference import Candidate, CandidateLedger, FitResult
from flash_ansr.scoring import resolve_ranking


def _flash_candidate(engine, value, score, rank):
    return Candidate(
        raw_beam=[rank], expression=["*", "<constant>", "x1"], slots=[1], expression_prefix=["*", repr(value), "x1"],
        expression_infix=f"{value} * x1", skeleton_prefix=["*", "<constant>", "x1"], constants=[value], constants_emitted=None,
        log_prob=-1.0, score=score, fvu=0.5, n_nodes=3, mu=None, mdl=1000.0, constant_count=1, pruned_variant=False,
        pareto_rank=-1, rank=rank)


def _fake_flash(engine, calls):
    """A FlashANSR stand-in whose fit returns a real FitResult of two fitted candidates, c * x1."""
    ranking = resolve_ranking("mdl", mdl_strength=1e-2)

    def fit(X, y, variable_names="auto", *, draws=None, complexity=None, seed=None):
        calls.append({"draws": draws, "complexity": complexity, "seed": seed})
        cands = [_flash_candidate(engine, 1.5, 100.0, 0), _flash_candidate(engine, 2.5, 200.0, 1)]
        ledger = CandidateLedger(token_lists=[[0], [1], [2]], fvu=[0.5, 0.5, float("nan")], log_prob=[-1.0, -1.0, -2.0],
                                 valid=[1, 1, 0], fit_status=[0, 0, 2], constants=[[1.5], [2.5], []], rank=[0, 1, -1],
                                 result_index=[0, 1, -1])
        return FitResult(candidates=cands, ledger=ledger, generation_time=0.1, refinement_time=0.2, ranking=ranking,
                         n_variables=2, variable_mapping={}, draws=draws, n_points=int(np.asarray(y).shape[0]), engine=engine)

    return SimpleNamespace(fit=fit, simplipy_engine=engine, generation_config=SimpleNamespace(draws=1024, emission="fittable"),
                           ranking=SimpleNamespace(as_dict=lambda: ranking.as_dict()))


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2.0, size=(40, 2))
    return X, X[:, 0] * X[:, 1]


@pytest.fixture
def pysr_calls(monkeypatch):
    """PySR replaced by a fixed hall of fame: the true law, x1 * x2."""
    calls = []

    def fake_run_pysr(X, y, variables, *, timeout_in_seconds, niterations, guesses=None, X_val=None, settings=None):
        calls.append({"niterations": niterations, "guesses": list(guesses or []), "variables": list(variables)})
        return {"expression": "x1 * x2", "equations": [{"complexity": 3, "loss": 0.0, "score": 1.0, "equation": "x1 * x2"}],
                "y_pred": None, "y_pred_val": None, "wall_s": 0.01, "error": None, "n_guesses": len(guesses or []),
                "niterations_used": niterations}

    monkeypatch.setattr(regressor_module, "run_pysr", fake_run_pysr)
    return calls


def _hybrid(engine, flash_calls, **cfg):
    return HybridRegressor(_fake_flash(engine, flash_calls), HybridConfig(pysr=PySRSettings(warmup=False), **cfg))


def test_fit_returns_the_extended_pool_in_flash_ansrs_ranking_order(engine, problem, pysr_calls):
    X, y = problem
    flash_calls = []
    hybrid = _hybrid(engine, flash_calls)
    result = hybrid.fit(X, y)
    assert isinstance(result, HybridFitResult) and isinstance(result, FitResult) and hybrid.result_ is result
    assert flash_calls[0]["draws"] == 1024 and pysr_calls[0]["niterations"] == 64            # the r* = 0.5 default
    assert pysr_calls[0]["guesses"] and pysr_calls[0]["variables"] == ["x1", "x2"]            # Flash-ANSR seeds PySR
    assert [c.source for c in result.candidates] == ["pysr", "flash-ansr", "flash-ansr"]      # the exact law wins
    assert [c.rank for c in result.candidates] == [0, 1, 2]
    assert result.best.hof_index == 0 and result.niterations == 64 and result.n_seeds == len(pysr_calls[0]["guesses"])
    assert result.ledger.result_index == [1, 2, -1] and result.ledger.rank == [1, 2, -1]    # the ledger follows the new order
    np.testing.assert_allclose(hybrid.predict(X).ravel(), y, rtol=1e-12)
    assert "x1" in hybrid.get_expression() and "x2" in hybrid.get_expression()
    assert result.get_expression(1, precision=2) == hybrid.get_expression(1, precision=2)


def test_draws_alone_takes_its_r_star_pair_and_zero_iterations_is_flash_ansr_alone(engine, problem, pysr_calls):
    X, y = problem
    flash_calls = []
    hybrid = _hybrid(engine, flash_calls)
    hybrid.fit(X, y, draws=4096)
    assert flash_calls[-1]["draws"] == 4096 and pysr_calls[-1]["niterations"] == 512
    result = hybrid.fit(X, y, draws=512, niterations=0)
    assert len(pysr_calls) == 1                                                               # PySR did not run
    assert [c.source for c in result.candidates] == ["flash-ansr", "flash-ansr"] and result.niterations == 0


def test_a_pysr_constant_is_a_slot_that_precision_rounds(engine, problem, monkeypatch):
    X, y = problem
    y = 1.2345678 * X[:, 0]
    monkeypatch.setattr(regressor_module, "run_pysr", lambda *a, **k: {
        "expression": "1.2345678 * x1", "equations": [{"complexity": 3, "loss": 0.0, "score": 1.0, "equation": "1.2345678 * x1"}],
        "wall_s": 0.0, "error": None})
    result = _hybrid(engine, []).fit(X, y)
    pysr = [c for c in result.candidates if c.source == "pysr"][0]
    assert pysr.slots and len(pysr.constants) == len(pysr.slots)
    assert "1.23" in "".join(result.get_expression(pysr.rank, precision=2, return_prefix=True))


def test_a_saved_hybrid_result_loads_with_its_sources(engine, problem, pysr_calls, tmp_path):
    X, y = problem
    result = _hybrid(engine, []).fit(X, y)
    path = tmp_path / "result.pkl"
    result.save(path)
    loaded = HybridFitResult.load(path, engine=engine)
    assert [c.source for c in loaded.candidates] == [c.source for c in result.candidates]
    assert loaded.pysr_equations == result.pysr_equations and loaded.niterations == result.niterations
    np.testing.assert_allclose(loaded.predict(X), result.predict(X))
    with pytest.raises(ValueError, match="not a hybrid FitResult"):
        FitResult.save(result, tmp_path / "plain.pkl") or HybridFitResult.load(tmp_path / "plain.pkl")
