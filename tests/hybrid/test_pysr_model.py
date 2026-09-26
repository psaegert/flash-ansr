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
    monkeypatch.setattr(pm, "_require_pysr", lambda: _FakeRegressor)
    settings = pm.PySRSettings.from_mapping({"maxsize": 20, "populations": 4})
    pm.create_model(timeout_in_seconds=7.4, niterations=10, settings=settings, guesses=["(v1 + 1.0)"])
    kw = _FakeRegressor.last
    assert kw["timeout_in_seconds"] == 7 and kw["niterations"] == 10 and kw["maxsize"] == 20 and kw["populations"] == 4
    assert kw["unary_operators"] == pm.UNARY_OPERATORS and kw["binary_operators"] == pm.BINARY_OPERATORS
    assert kw["guesses"] == ["(v1 + 1.0)"] and kw["model_selection"] == "best"
    pm.create_model(timeout_in_seconds=0.2, niterations=1)
    assert _FakeRegressor.last["timeout_in_seconds"] == 1 and "guesses" not in _FakeRegressor.last


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_real_search_returns_a_hall_of_fame():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(64, 2))
    y = 1.5 * x[:, 0] + 2.0
    out = pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=5, niterations=1_000_000, guesses=["((1.5 * v1) + 2.0)"],
                      X_val=x[:4], settings=pm.PySRSettings(warmup=False))
    assert out["error"] is None and out["expression"] and out["equations"]
    assert {"complexity", "loss", "score", "equation"} <= set(out["equations"][0])
    assert out["y_pred"].shape == (64,) and out["y_pred_val"].shape == (4,) and out["wall_s"] > 0
