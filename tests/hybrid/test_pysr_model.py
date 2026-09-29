"""The GP stage's regressor: built over the 23-operator vocabulary with the seeds forwarded (a fake
PySRRegressor records the kwargs); the real Julia search only under FLASH_ANSR_HYBRID_JULIA=1."""
import os

import numpy as np
import pytest

import flash_ansr.hybrid.pysr_model as pm


class _FakeRegressor:
    last = None

    def __init__(self, **kwargs):
        _FakeRegressor.last = kwargs


def test_create_model_forwards_vocabulary_and_guesses(monkeypatch):
    no_export = object()
    monkeypatch.setattr(pm, "_require_pysr", lambda: _FakeRegressor)
    monkeypatch.setattr(pm, "_no_export_spec", lambda: no_export)
    settings = pm.PySRSettings.from_mapping({"maxsize": 20, "populations": 4})
    pm.create_model(timeout_in_seconds=7.4, niterations=10, settings=settings, guesses=["(v1 + 1.0)"])
    kw = _FakeRegressor.last
    assert kw["timeout_in_seconds"] == 7 and kw["niterations"] == 10 and kw["maxsize"] == 20 and kw["populations"] == 4
    assert kw["unary_operators"] == pm.UNARY_OPERATORS and kw["binary_operators"] == pm.BINARY_OPERATORS
    assert kw["guesses"] == ["(v1 + 1.0)"] and kw["model_selection"] == "best"
    assert kw["expression_spec"] is no_export                            # no sympy/numpy exports of the hall of fame
    pm.create_model(timeout_in_seconds=0.2, niterations=1)
    assert _FakeRegressor.last["timeout_in_seconds"] == 1 and "guesses" not in _FakeRegressor.last
    own = object()                                                       # an explicit spec in the settings wins
    pm.create_model(timeout_in_seconds=1, niterations=1, settings=pm.PySRSettings.from_mapping({"expression_spec": own}))
    assert _FakeRegressor.last["expression_spec"] is own


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_real_search_returns_a_hall_of_fame():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(64, 2))
    y = 1.5 * x[:, 0] + 2.0
    out = pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=5, niterations=1_000_000, guesses=["((1.5 * v1) + 2.0)"],
                      settings=pm.PySRSettings(warmup=False))
    assert out["error"] is None and out["expression"] and out["equations"] and out["wall_s"] > 0
    assert {"complexity", "loss", "score", "equation"} <= set(out["equations"][0])


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_real_search_never_builds_sympy(monkeypatch):
    """PySR turns every hall-of-fame entry into sympy at the end of fit; on 2026-09-29 a 25-deep tanh/sinh chain held
    one fit there for more than 8 hours. The hybrid's search must finish without calling that conversion at all."""
    import pysr.export

    def refuse(*args, **kwargs):
        raise AssertionError("the hybrid's PySR must not convert its hall of fame to sympy")

    monkeypatch.setattr(pysr.export, "pysr2sympy", refuse)
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, size=(48, 1))
    y = np.tanh(np.sinh(np.tanh(np.sinh(0.3 * x[:, 0]))))
    out = pm.run_pysr(x, y, ["v1"], timeout_in_seconds=10, niterations=5, settings=pm.PySRSettings(warmup=False))
    assert out["error"] is None and out["expression"] and len(out["equations"]) > 1


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_dropping_the_exports_leaves_the_search_unchanged():
    """A deterministic search (serial, fixed seed) with and without the exports: the same hall of fame and pick."""
    from pysr.expression_specs import ExpressionSpec

    rng = np.random.default_rng(2)
    x = rng.uniform(-2, 2, size=(40, 2))
    y = 1.5 * x[:, 0] * x[:, 1] - np.sin(x[:, 1])
    runs = []
    for spec in (None, ExpressionSpec()):
        extra = {"parallelism": "serial", "deterministic": True, "random_state": 0, **({"expression_spec": spec} if spec else {})}
        runs.append(pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=60, niterations=3,
                                settings=pm.PySRSettings(warmup=False, extra=extra)))
    assert runs[0]["error"] is None and runs[1]["error"] is None
    assert runs[0]["expression"] == runs[1]["expression"] and runs[0]["equations"] == runs[1]["equations"]
