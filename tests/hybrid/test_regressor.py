"""The regressor: the clocked generation pass and its snapshot cache, the r = 0 answer, the PySR branch
(the GP stage replaced by a canned hall of fame) and the clock's bookkeeping."""
import pickle
import warnings

import numpy as np
import pytest

import flash_ansr.hybrid.regressor as regressor_module
from flash_ansr.hybrid import HybridConfig, HybridRegressor

from hybrid_fakes import fake_model, toy_problem


class TestSnapshotCache:
    def test_cache_hits_on_the_same_data_and_regenerates_on_other_data(self, tmp_path):
        reg = HybridRegressor(object(), HybridConfig(budget_s=20.0, ratio=0.0, ratios=[0.0, 0.5, 1.0]), snapshot_dir=tmp_path)
        calls = []

        def fake_generate(X, y, X_val, *, variables, complexity):
            calls.append(X.shape)
            return {t: {"seconds": t, "choices": 10, "cum_wall": t, "cum_generation": 0.5, "cum_refinement": 0.5,
                        "pool_size": 1, "best": None, "seeds": [], "calls": 1, "candidates": []} for t in reg.time_targets()}

        reg._generate_snapshots = fake_generate
        x, y, x_val, _ = toy_problem(0)
        first = reg._snapshot(x, y, x_val, problem_id=0, variables=["v1", "v2"], complexity=None)
        second = reg._snapshot(x.copy(), y.copy(), x_val.copy(), problem_id=0, variables=["v1", "v2"], complexity=None)
        assert first == second and len(calls) == 1
        assert (tmp_path / "problem_000000.pkl").exists()
        xo, yo, xvo, _ = toy_problem(1)                                    # a re-drawn instance of "the same problem"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reg._snapshot(xo, yo, xvo, problem_id=0, variables=["v1", "v2"], complexity=None)
        assert len(calls) == 2 and any("different data" in str(w.message) for w in caught)
        # a snapshot with fewer targets than the sweep asks for is regenerated
        with (tmp_path / "problem_000000.pkl").open("wb") as fh:
            pickle.dump({"fingerprint": "x", "targets": {}}, fh)
        reg._snapshot(x, y, x_val, problem_id=0, variables=["v1", "v2"], complexity=None)
        assert len(calls) == 3

    def test_no_cache_without_a_directory_or_id(self, tmp_path):
        reg = HybridRegressor(object(), HybridConfig(budget_s=20.0, ratio=0.5))
        calls = []
        reg._generate_snapshots = lambda X, y, X_val, *, variables, complexity: calls.append(1) or {10.0: {}}
        x, y, x_val, _ = toy_problem(0)
        reg._snapshot(x, y, x_val, problem_id=0, variables=["v1", "v2"], complexity=None)
        reg._snapshot(x, y, x_val, problem_id=0, variables=["v1", "v2"], complexity=None)
        assert len(calls) == 2


class TestClockedGeneration:
    """The generation pass runs by the clock: chunks until the wall time reaches each share, landing
    within the tolerance; the candidate count is what the clock allowed."""

    def test_lands_each_share_within_the_tolerance(self, tmp_path):
        model = fake_model(seconds_per_candidate=0.0005, call_overhead=0.01)
        budget = 1.2
        reg = HybridRegressor(model, HybridConfig(budget_s=budget, ratio=0.5, ratios=[0.0, 0.5, 1.0], k_seeds=5,
                                                  landing_tolerance=0.02, first_chunk=32, min_chunk=8), snapshot_dir=tmp_path)
        x, y, x_val, _ = toy_problem(0)
        targets = reg._generate_snapshots(x, y, x_val, variables=["v1", "v2"], complexity=None)
        assert sorted(targets) == [0.6, 1.2]
        tol = 0.02 * budget
        for t, snap in targets.items():
            assert snap["seconds"] == t
            assert t - tol <= snap["cum_wall"] <= t + 0.15, (t, snap["cum_wall"])   # short by at most the tolerance, over by at most a chunk's slip
            assert snap["choices"] <= model.fit.counter
            assert snap["best"] is not None and len(snap["seeds"]) == 5 and len(snap["candidates"]) == 5
            assert "y_pred" in snap["candidates"][0] and "y_pred" not in snap["candidates"][1]   # rank 0 carries its curves
        assert targets[1.2]["choices"] > targets[0.6]["choices"] > 0
        assert targets[1.2]["calls"] <= 8                                   # a big chunk and a landing chunk per share, not a dribble
        assert model.generation_config.draws == 1024                         # the model's own setting is untouched (draws= is per call)
        assert all(s.startswith("(") and "v1" in s for s in targets[0.6]["seeds"])   # Julia syntax in the fit's names

    def test_fit_at_ratio_zero_is_flash_alone(self, tmp_path):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005)
        reg = HybridRegressor(model, HybridConfig(budget_s=0.5, ratio=0.0, ratios=[0.0, 0.5], k_seeds=3, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4), snapshot_dir=tmp_path)
        reg.prepare()                                                        # no PySR share: no warmup
        x, y, x_val, _ = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=7)
        assert out["prediction_success"] and out["predicted_source"] == "flash-ansr" and out["error"] is None
        assert out["hybrid_target_generation_s"] == 0.5 and out["hybrid_target_gp_s"] == 0.0
        assert out["fit_time"] == out["hybrid_generation_s"] and out["hybrid_choices"] > 0
        assert out["y_pred"].shape == (48, 1) and out["y_pred_val"].shape == (12, 1)
        assert out["predicted_expression_prefix"][0] == "*" and out["predicted_constants"]
        assert (tmp_path / "problem_000007.pkl").exists()
        with pytest.raises(ValueError, match="not one of the configured ratios"):
            reg.solve(x, y, ratio=0.3)
        with pytest.raises(ValueError, match="not one of the configured budgets"):
            reg.solve(x, y, budget=0.25)

    def test_fit_along_a_budget_ladder_reuses_one_pass(self, tmp_path):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005)
        reg = HybridRegressor(model, HybridConfig(budget_s=0.4, ratio=0.0, budgets=[0.4, 0.8], k_seeds=3, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4), snapshot_dir=tmp_path)
        x, y, x_val, _ = toy_problem(0)
        small = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=1, budget=0.4)
        counter = model.fit.counter
        large = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=1, budget=0.8)
        assert model.fit.counter == counter                                # the second budget came from the cached pass
        assert small["hybrid_budget_s"] == 0.4 and large["hybrid_budget_s"] == 0.8
        assert large["hybrid_choices"] > small["hybrid_choices"] and large["fit_time"] > small["fit_time"]


class TestPySRBranch:
    """The GP stage replaced by a canned hall of fame: the seeds and the clock reach PySR, its candidates
    are priced with the real engine and Flash-ANSR's sorting picks; the running costs are observed."""

    def test_seeded_search_and_pick(self, tmp_path, monkeypatch, engine):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine)
        seen = {}

        def canned_run_pysr(X, y, variables, *, timeout_in_seconds, niterations, guesses=None, X_val=None, settings=None):
            seen.update(timeout=timeout_in_seconds, niterations=niterations, guesses=list(guesses or []), variables=list(variables))
            return {"expression": "v1", "equations": [{"complexity": 1, "loss": 1.0, "score": 0.0, "equation": "v1"},
                                                      {"complexity": 7, "loss": 0.0, "score": 1.0, "equation": "((1.5 * v1) - (0.5 * v2)) + 2.0"}],
                    "y_pred": X[:, 0], "y_pred_val": None if X_val is None else X_val[:, 0], "n_guesses": len(guesses or []),
                    "niterations_used": niterations, "error": None, "wall_s": timeout_in_seconds + 0.3}

        monkeypatch.setattr(regressor_module, "run_pysr", canned_run_pysr)
        reg = HybridRegressor(model, HybridConfig(budget_s=4.0, ratio=0.9, ratios=[0.9], k_seeds=3, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4, pysr_overhead_s=1.0, pricing_reserve_s=0.1,
                                                  pysr={"warmup": False}), snapshot_dir=tmp_path / "snapshots" / "cat")
        x, y, x_val, y_val = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert seen["timeout"] == pytest.approx(3.6 - 1.0 - 0.1) and seen["niterations"] == 1_000_000
        assert len(seen["guesses"]) == 3 and seen["variables"] == ["v1", "v2"]
        assert out["predicted_source"] == "pysr" and out["predicted_hof_index"] == 1 and out["prediction_success"]
        np.testing.assert_allclose(out["y_pred"].reshape(-1), y, atol=1e-9)
        np.testing.assert_allclose(out["y_pred_val"].reshape(-1), y_val, atol=1e-9)
        assert out["pysr_expression"] == "v1" and out["pysr_expression_prefix"] == ["x1"]
        assert out["hybrid_gp_s"] == pytest.approx(seen["timeout"] + 0.3)
        assert out["fit_time"] == pytest.approx(out["hybrid_generation_s"] + out["hybrid_gp_s"] + out["hybrid_ranking_s"])
        assert len(out["hybrid_candidates"]) >= 4                            # 3 Flash-ANSR + PySR's rows
        # the clock observed PySR's fixed cost (0.3 s over its timeout) and the pricing, and persisted them
        assert reg.clock.pysr_overhead.count == 2 and reg.clock.pysr_overhead.value == pytest.approx((1.0 + 0.3) / 2)
        assert reg.clock.pricing_reserve.count == 2
        assert (tmp_path / "snapshots" / "clock_state.json").exists()

    def test_the_pick_prices_under_the_two_part_code(self, tmp_path, monkeypatch, engine):
        """flash-ansr 0.18 ranks with the two-part code (mode mdl, no mdl_strength), whose per-bit weight
        depends on the support size. The pick asked for effective_weights, which raises under it, so on
        2026-09-19 every PySR-alone row on solomon searched its whole budget and then recorded a
        RankingError. The fake above ranks with a fixed weight and never reached that path."""
        from flash_ansr.scoring import RankingConfig, two_part_strength

        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine,
                           ranking={"mode": "mdl", "mdl_strength": None})
        assert RankingConfig.from_dict(model.ranking.as_dict()).two_part
        seen = {}

        def canned_run_pysr(X, y, variables, *, timeout_in_seconds, niterations, guesses=None, X_val=None, settings=None):
            return {"expression": "v1", "equations": [{"complexity": 1, "loss": 1.0, "score": 0.0, "equation": "v1"},
                                                      {"complexity": 7, "loss": 0.0, "score": 1.0, "equation": "((1.5 * v1) - (0.5 * v2)) + 2.0"}],
                    "y_pred": X[:, 0], "y_pred_val": None if X_val is None else X_val[:, 0], "n_guesses": len(guesses or []),
                    "niterations_used": niterations, "error": None, "wall_s": timeout_in_seconds + 0.3}

        real_pick = regressor_module.pick_prediction

        def spy(values, **kw):
            seen["weights"] = dict(kw["weights"])
            return real_pick(values, **kw)

        monkeypatch.setattr(regressor_module, "run_pysr", canned_run_pysr)
        monkeypatch.setattr(regressor_module, "pick_prediction", spy)
        reg = HybridRegressor(model, HybridConfig(budget_s=4.0, ratio=0.9, ratios=[0.9], k_seeds=3, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4, pysr_overhead_s=1.0, pricing_reserve_s=0.1,
                                                  pysr={"warmup": False}), snapshot_dir=tmp_path / "snapshots" / "cat")
        x, y, x_val, y_val = toy_problem(0)
        y = y.copy()
        y[0] = np.nan                                                        # one non-finite target: n counts finite rows only
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert not out.get("hybrid_ranking_error") and not out.get("error"), out.get("error")
        assert out["prediction_success"] and out["predicted_source"] == "pysr"
        assert seen["weights"] == {"mdl": two_part_strength(len(y) - 1)}

    def test_a_failed_search_does_not_feed_the_fixed_cost_mean(self, tmp_path, monkeypatch, engine):
        """A stalled worker returned 7,208 s over its clock on 2026-09-12 and, as ONE observation in the
        sweep-wide running mean (2.8 s -> 21.8 s), zeroed the search clock of every later problem at
        r <= 0.2. A failed search measures nothing about PySR's dispatch and return: the row records the
        stall, the clock does not."""
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine)

        def stalled_run_pysr(X, y, variables, *, timeout_in_seconds, niterations, guesses=None, X_val=None, settings=None):
            return {"expression": None, "equations": [], "y_pred": None, "y_pred_val": None, "n_guesses": len(guesses or []),
                    "niterations_used": 0, "error": "WorkerTimeout: no reply within 7200 s", "wall_s": timeout_in_seconds + 7200.0}

        monkeypatch.setattr(regressor_module, "run_pysr", stalled_run_pysr)
        reg = HybridRegressor(model, HybridConfig(budget_s=4.0, ratio=0.9, ratios=[0.9], k_seeds=3, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4, pysr_overhead_s=1.0, pricing_reserve_s=0.1,
                                                  pysr={"warmup": False}), snapshot_dir=tmp_path / "snapshots" / "cat")
        x, y, x_val, y_val = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert out["hybrid_gp_s"] > 7200 and out["pysr_error"]                       # the stall is on the row ...
        assert reg.clock.pysr_overhead.count == 1 and reg.clock.pysr_overhead.value == 1.0   # ... and not in the clock
        assert reg.pysr_search_seconds(0.9, 4.0) == pytest.approx(3.6 - 1.0 - reg.clock.pricing_reserve.value)
        assert out["predicted_source"] == "flash-ansr" and out["prediction_success"]

    def test_a_share_the_fixed_costs_eat_skips_the_search(self, tmp_path, monkeypatch, engine):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine)
        monkeypatch.setattr(regressor_module, "run_pysr", lambda *a, **k: pytest.fail("PySR must not run"))
        reg = HybridRegressor(model, HybridConfig(budget_s=2.0, ratio=0.5, ratios=[0.5], k_seeds=2, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4, pysr_overhead_s=4.0, pysr={"warmup": False}), snapshot_dir=tmp_path)
        x, y, x_val, _ = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert out["hybrid_gp_s"] == 0.0 and "no search time" in out["pysr_error"] and out["error"] is None
        assert out["predicted_source"] == "flash-ansr" and out["prediction_success"]   # the Flash-ANSR candidates rank alone

    def test_a_failed_search_falls_back_on_flash(self, tmp_path, monkeypatch, engine):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine)
        monkeypatch.setattr(regressor_module, "run_pysr", lambda X, y, v, **k: {
            "expression": None, "equations": [], "y_pred": None, "y_pred_val": None, "n_guesses": 0, "niterations_used": 0,
            "error": "RuntimeError: Julia died", "wall_s": 0.5})
        reg = HybridRegressor(model, HybridConfig(budget_s=3.0, ratio=0.5, ratios=[0.5], k_seeds=2, landing_tolerance=0.05,
                                                  first_chunk=16, min_chunk=4, pysr_overhead_s=0.5, pysr={"warmup": False}), snapshot_dir=tmp_path)
        x, y, x_val, _ = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert out["predicted_source"] == "flash-ansr" and out["prediction_success"] and out["error"] is None
        assert out["pysr_error"].startswith("RuntimeError") and out["pysr_expression"] is None


class TestKnobMode:
    """The knob mode: the method is its work. Flash-ANSR draws exactly D candidates (one increment per rung
    of the ladder, from one cached pass), PySR runs exactly I iterations under a safety cap, no clock is
    read or updated."""

    def test_config(self):
        cfg = HybridConfig(draws=1024, niterations=64, rungs=[[512, 16], [1024, 64]])
        assert cfg.mode == "knobs" and cfg.rungs == [(512, 16), (1024, 64)] and cfg.ratios == [] and cfg.budgets == []
        assert HybridConfig(draws=64, niterations=0).rungs == [(64, 0)]         # the pair itself is a rung
        # r* = 0.5: draws alone takes its paired iterations; no knob at all is the default rung of the whole ladder
        assert (HybridConfig(draws=4096).draws, HybridConfig(draws=4096).niterations) == (4096, 512)
        default = HybridConfig()
        assert (default.mode, default.draws, default.niterations) == ("knobs", 1024, 64)
        assert default.rungs == [(512, 16), (1024, 64), (2048, 256), (4096, 512), (8192, 1024)]
        with pytest.raises(ValueError, match="knob mode sets draws"):
            HybridConfig(niterations=5)
        assert HybridConfig(budget_s=1.0, ratio=0.5).mode == "clock"
        assert HybridConfig.from_mapping({"draws": 8, "niterations": 2, "pysr": {"warmup": False}}).pysr.warmup is False

    def test_one_pass_serves_the_ladder_with_one_increment_per_rung(self, tmp_path):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005)
        reg = HybridRegressor(model, HybridConfig(draws=64, niterations=0, rungs=[[16, 0], [64, 0]], k_seeds=3), snapshot_dir=tmp_path)
        reg.prepare()
        x, y, x_val, _ = toy_problem(0)
        small = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=3, draws=16)
        assert model.fit.counter == 64 and small["hybrid_generation_calls"] == 1              # one pass serves the ladder: 16, then +48
        large = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=3)              # the configured rung (64, 0)
        assert model.fit.counter == 64 and large["hybrid_generation_calls"] == 2                # from the cache, no re-generation
        for out, d in ((small, 16), (large, 64)):
            assert out["hybrid_mode"] == "knobs" and out["hybrid_draws"] == d and out["hybrid_niterations"] == 0
            assert out["hybrid_choices"] == d and out["prediction_success"] and out["predicted_source"] == "flash-ansr"
            assert out["fit_time"] == out["hybrid_generation_s"] and "hybrid_target_generation_s" not in out
        assert large["fit_time"] > small["fit_time"]
        assert (tmp_path / "problem_000003.draws.pkl").exists()
        with pytest.raises(ValueError, match="not one of the configured rungs"):
            reg.solve(x, y, draws=32)

    def test_a_single_rung_is_one_generation_call(self, tmp_path):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005)
        reg = HybridRegressor(model, HybridConfig(draws=40, niterations=0, k_seeds=3), snapshot_dir=tmp_path)
        x, y, x_val, _ = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert model.fit.counter == 40 and out["hybrid_generation_calls"] == 1 and out["hybrid_choices"] == 40

    def test_pysr_runs_its_iterations_under_the_cap_and_the_clock_is_untouched(self, tmp_path, monkeypatch, engine):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine)
        seen = {}

        def canned_run_pysr(X, y, variables, *, timeout_in_seconds, niterations, guesses=None, X_val=None, settings=None):
            seen.update(timeout=timeout_in_seconds, niterations=niterations, guesses=list(guesses or []), variables=list(variables))
            return {"expression": "v1", "equations": [{"complexity": 1, "loss": 1.0, "score": 0.0, "equation": "v1"},
                                                      {"complexity": 7, "loss": 0.0, "score": 1.0, "equation": "((1.5 * v1) - (0.5 * v2)) + 2.0"}],
                    "y_pred": X[:, 0], "y_pred_val": None if X_val is None else X_val[:, 0], "n_guesses": len(guesses or []),
                    "niterations_used": niterations, "error": None, "wall_s": 0.7}

        monkeypatch.setattr(regressor_module, "run_pysr", canned_run_pysr)
        reg = HybridRegressor(model, HybridConfig(draws=32, niterations=16, k_seeds=3, pysr_timeout_cap_s=900.0,
                                                  pysr={"warmup": False}), snapshot_dir=tmp_path / "snapshots" / "cat")
        x, y, x_val, y_val = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], problem_id=0)
        assert seen["niterations"] == 16 and seen["timeout"] == 900.0 and len(seen["guesses"]) == 3
        assert out["hybrid_niterations"] == 16 and out["niterations_used"] == 16 and out["hybrid_gp_timeout_s"] == 900.0
        assert out["predicted_source"] == "pysr" and out["prediction_success"]
        np.testing.assert_allclose(out["y_pred_val"].reshape(-1), y_val, atol=1e-9)
        assert out["fit_time"] == pytest.approx(out["hybrid_generation_s"] + 0.7 + out["hybrid_ranking_s"])
        assert reg.clock.pysr_overhead.count == 1 and reg.clock.pricing_reserve.count == 1   # nothing observed
        assert not (tmp_path / "snapshots" / "clock_state.json").exists()                     # nothing persisted

    def test_zero_draws_is_pysr_cold(self, tmp_path, monkeypatch, engine):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005, engine=engine)
        monkeypatch.setattr(regressor_module, "run_pysr", lambda X, y, v, **k: {
            "expression": "v1", "equations": [{"complexity": 1, "loss": 1.0, "score": 0.0, "equation": "v1"}], "y_pred": X[:, 0],
            "y_pred_val": None, "n_guesses": len(k.get("guesses") or []), "niterations_used": k["niterations"], "error": None, "wall_s": 0.2})
        reg = HybridRegressor(model, HybridConfig(draws=0, niterations=8, pysr={"warmup": False}))
        x, y, x_val, _ = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"])
        assert model.fit.counter == 0 and out["n_guesses"] == 0 and out["prediction_success"] and out["hybrid_choices"] == 0


    def test_without_a_cache_a_rung_draws_only_its_own_count(self):
        model = fake_model(seconds_per_candidate=0.0002, call_overhead=0.005)
        reg = HybridRegressor(model, HybridConfig(draws=1024, niterations=0, rungs=[[16, 0], [64, 0], [1024, 0]], k_seeds=3))
        x, y, x_val, _ = toy_problem(0)
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], draws=64)
        assert model.fit.counter == 64 and out["hybrid_generation_calls"] == 2 and out["hybrid_choices"] == 64   # 16 + 48, never 1024
        out = reg.solve(x, y, X_val=x_val, variables=["v1", "v2"], draws=16)
        assert model.fit.counter == 80 and out["hybrid_generation_calls"] == 1                                    # a fresh pass of 16
