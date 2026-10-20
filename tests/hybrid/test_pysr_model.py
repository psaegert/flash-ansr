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
    julia_operator_definitions = []                                     # no Julia here, in either test mode
    monkeypatch.setattr(pm, "_seval", julia_operator_definitions.append)
    monkeypatch.setattr(pm, "_OPERATORS_DEFINED", False)
    no_export = object()
    monkeypatch.setattr(pm, "_require_pysr", lambda: _FakeRegressor)
    monkeypatch.setattr(pm, "_no_export_spec", lambda: no_export)
    settings = pm.PySRSettings.from_mapping({"maxsize": 20, "populations": 4})
    pm.create_model(timeout_in_seconds=7.4, niterations=10, settings=settings, guesses=["(v1 + 1.0)"])
    kw = _FakeRegressor.last
    assert kw["timeout_in_seconds"] == 7 and kw["niterations"] == 10 and kw["maxsize"] == 20 and kw["populations"] == 4
    assert kw["unary_operators"] == pm.UNARY_OPERATORS and kw["binary_operators"] == pm.BINARY_OPERATORS
    assert "rootn" in kw["binary_operators"] and pm.ROOTN_JULIA not in kw["binary_operators"]   # rootn by name ...
    assert julia_operator_definitions == [pm.ROOTN_JULIA]                                         # ... defined once
    assert kw["guesses"] == ["(v1 + 1.0)"] and kw["model_selection"] == "best"
    assert kw["expression_spec"] is no_export                            # no sympy/numpy exports of the hall of fame
    pm.create_model(timeout_in_seconds=0.2, niterations=1)
    assert _FakeRegressor.last["timeout_in_seconds"] == 1 and "guesses" not in _FakeRegressor.last
    assert julia_operator_definitions == [pm.ROOTN_JULIA]                                         # not again
    own = object()                                                       # an explicit spec in the settings wins
    pm.create_model(timeout_in_seconds=1, niterations=1, settings=pm.PySRSettings.from_mapping({"expression_spec": own}))
    assert _FakeRegressor.last["expression_spec"] is own


class _FakePySRModule:
    """``pysr.sr`` as far as the seed hook needs it: the function PySR's fit reads its guesses through."""

    def __init__(self):
        self.received = []

    def _prepare_guesses_for_julia(self, guesses, nout, **kw):
        self.received.append(guesses)
        return guesses


def _fake_pysr(monkeypatch, *, reads_guesses=True, setup_s=0.0):
    """A stand-in PySR whose fit sets up for ``setup_s``, then reads its guesses through ``pysr.sr`` (or never does)."""
    import sys
    import time
    import types
    sr = _FakePySRModule()
    pkg = types.ModuleType("pysr")
    pkg.sr = sr
    monkeypatch.setitem(sys.modules, "pysr", pkg)
    monkeypatch.setitem(sys.modules, "pysr.sr", sr)

    class Regressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, variable_names=None):
            time.sleep(setup_s)
            if reads_guesses:
                self.guesses_seen = sys.modules["pysr.sr"]._prepare_guesses_for_julia(self.kwargs.get("guesses"), 1)

        @property
        def equations_(self):
            import pandas as pd
            return pd.DataFrame([{"complexity": 1, "loss": 0.0, "score": 0.0, "equation": "v1"}])

        def get_best(self):
            return {"equation": "v1"}

    monkeypatch.setattr(pm, "_require_pysr", lambda: Regressor)
    monkeypatch.setattr(pm, "_no_export_spec", lambda: object())
    return sr


def test_late_seeds_reach_the_point_where_pysr_reads_its_guesses(monkeypatch):
    import threading
    import time
    sr = _fake_pysr(monkeypatch, setup_s=0.05)
    future = pm.SeedFuture()
    threading.Timer(0.25, future.set, args=[["(v1 + 1.0)", "(v1 * v2)"]]).start()   # the seeds come later
    out = pm.run_pysr(np.zeros((4, 2)), np.zeros(4), ["v1", "v2"], timeout_in_seconds=5, niterations=1, guesses=future)
    assert out["error"] is None and sr.received == [["(v1 + 1.0)", "(v1 * v2)"]] and out["n_guesses"] == 2
    assert out["setup_s"] == pytest.approx(0.05, abs=0.04) and out["wait_s"] == pytest.approx(0.2, abs=0.06)
    assert out["setup_s"] + out["wait_s"] + out["search_s"] == pytest.approx(out["wall_s"], abs=1e-6)
    assert pm._PENDING is None                                                       # nothing left pending
    no_seeds = pm.SeedFuture()
    no_seeds.set([])
    out = pm.run_pysr(np.zeros((4, 2)), np.zeros(4), ["v1", "v2"], timeout_in_seconds=5, niterations=1, guesses=no_seeds)
    assert sr.received[-1] is None and out["n_guesses"] == 0                         # no seeds: PySR's own start
    assert time.time()                                                                # (keeps the import used)


def test_a_pysr_that_never_reads_its_guesses_is_an_error(monkeypatch):
    _fake_pysr(monkeypatch, reads_guesses=False)
    future = pm.SeedFuture()
    future.set(["(v1 + 1.0)"])
    out = pm.run_pysr(np.zeros((4, 2)), np.zeros(4), ["v1", "v2"], timeout_in_seconds=5, niterations=1, guesses=future)
    assert out["expression"] is None and "never read its guesses" in out["error"]


def test_a_failed_seed_stage_ends_the_search(monkeypatch):
    _fake_pysr(monkeypatch)
    future = pm.SeedFuture()
    future.fail(RuntimeError("no candidates"))
    out = pm.run_pysr(np.zeros((4, 2)), np.zeros(4), ["v1", "v2"], timeout_in_seconds=5, niterations=1, guesses=future)
    assert out["expression"] is None and out["error"].startswith("SeedsUnavailable") and "no candidates" in out["error"]
    assert pm._PENDING is None


def test_the_seed_hook_installs_once(monkeypatch):
    sr = _fake_pysr(monkeypatch)
    pm._install_seed_hook()
    first = sr._prepare_guesses_for_julia
    pm._install_seed_hook()
    assert sr._prepare_guesses_for_julia is first and first._flash_ansr_seed_hook
    assert first(["a"], 1) == ["a"]                                                  # no pending future: unchanged


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


def _deterministic(**extra):
    return pm.PySRSettings(warmup=False, extra={"parallelism": "serial", "deterministic": True, "random_state": 0, **extra})


def _root_problem():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.5, 3.0, size=(40, 2))
    return x, np.sqrt(x[:, 0]) * x[:, 1] + 1.0


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_rootn_defined_once_leaves_the_search_unchanged(monkeypatch):
    """The same deterministic search with rootn as an inline definition (evaluated again in every fit, as before
    2026-10-20) and by name (defined once): the same hall of fame and pick."""
    x, y = _root_problem()
    seeds = ["(rootn(v1, 2.0) * v2)"]
    by_name = pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=120, niterations=3, guesses=seeds, settings=_deterministic())
    monkeypatch.setattr(pm, "BINARY_OPERATORS", ["+", "-", "*", "/", "^", pm.ROOTN_JULIA])
    inline = pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=120, niterations=3, guesses=seeds, settings=_deterministic())
    assert by_name["error"] is None and inline["error"] is None
    assert by_name["expression"] == inline["expression"] and by_name["equations"] == inline["equations"]


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_seeds_handed_over_late_leave_the_search_unchanged():
    """The same deterministic search with the seeds given up front and handed over from another thread while PySR
    waits: the same hall of fame and pick, and PySR really waited for them."""
    import threading
    x, y = _root_problem()
    seeds = ["(rootn(v1, 2.0) * v2)", "(v2 + 1.0)"]
    early = pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=120, niterations=3, guesses=seeds, settings=_deterministic())
    future = pm.SeedFuture()
    threading.Timer(2.0, future.set, args=[seeds]).start()
    late = pm.run_pysr(x, y, ["v1", "v2"], timeout_in_seconds=120, niterations=3, guesses=future, settings=_deterministic())
    assert early["error"] is None and late["error"] is None and late["n_guesses"] == 2 and late["wait_s"] > 0.5
    assert early["expression"] == late["expression"] and early["equations"] == late["equations"]


@pytest.mark.skipif(os.environ.get("FLASH_ANSR_HYBRID_JULIA") != "1", reason="set FLASH_ANSR_HYBRID_JULIA=1 to run the Julia search")
def test_the_operator_check_is_compiled_once_and_answers_the_same():
    import pysr.sr as sr
    from pysr import jl
    pm.define_operators()
    check = sr.jl_is_function
    assert getattr(check, "_flash_ansr_compiled_once", False)
    assert check(jl.seval("rootn")) is True and check(jl.seval("sin")) is True and check(jl.seval("1.5")) is False
    pm.define_operators()
    assert sr.jl_is_function is check                                    # installed once
