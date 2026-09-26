"""The budget split by the clock: shares, snapshot targets, PySR's own clock minus the running costs,
their persistence across cells, and the config's validation."""
import pytest

from flash_ansr.hybrid import ClockState, HybridConfig, HybridRegressor, PySRSettings, RunningMean


def _regressor(ratio, ratios, **kw):
    snapshot_dir = kw.pop("snapshot_dir", None)
    return HybridRegressor(object(), HybridConfig(budget_s=100.0, ratio=ratio, ratios=ratios, **kw), snapshot_dir=snapshot_dir)


class TestSplit:
    def test_shares_and_targets(self):
        ratios = [i / 10 for i in range(11)]
        a = _regressor(0.0, ratios)
        assert a.generation_seconds(0.0) == 100.0 and a.pysr_seconds(0.0) == 0.0
        assert a.generation_seconds(1.0) == 0.0 and a.pysr_seconds(1.0) == 100.0
        assert a.time_targets() == [10.0 * k for k in range(1, 11)]        # one snapshot per share, 0 excluded
        for r in ratios:
            assert a.generation_seconds(r) + a.pysr_seconds(r) == pytest.approx(100.0)

    def test_pysr_runs_on_its_own_clock_minus_the_running_costs(self):
        a = _regressor(0.1, [0.1], pysr_overhead_s=2.5, pricing_reserve_s=0.5)
        assert a.pysr_search_seconds(0.1) == pytest.approx(7.0)             # 10 s share - 2.5 - 0.5
        a.clock.pysr_overhead.observe(3.5)                                 # the configured value counts as one observation
        a.clock.pricing_reserve.observe(1.5)
        assert a.pysr_search_seconds(0.1) == pytest.approx(10.0 - 3.0 - 1.0)
        a.clock.pysr_overhead.observe(float("nan"))                       # ignored
        assert a.pysr_search_seconds(0.1) == pytest.approx(6.0)
        assert a.pysr_search_seconds(0.01) == 0.0                          # a share the fixed costs eat: no search

    def test_running_costs_persist_across_cells(self, tmp_path):
        first = _regressor(0.5, [0.5], snapshot_dir=tmp_path / "snapshots" / "fastsrb")
        first.clock.pysr_overhead.observe(6.0); first.clock.pricing_reserve.observe(0.6); first.clock.save()
        assert (tmp_path / "snapshots" / "clock_state.json").exists()      # beside the snapshot directories
        second = _regressor(0.3, [0.3], snapshot_dir=tmp_path / "snapshots" / "feynman")
        assert second.clock.pysr_overhead.value == pytest.approx(5.0) and second.clock.pysr_overhead.count == 2
        assert second.pysr_search_seconds(0.3) == pytest.approx(30.0 - 5.0 - 0.4)

    def test_running_mean_and_unreadable_state(self, tmp_path):
        m = RunningMean(4.0)
        m.observe(6.0); m.observe(None); m.observe(float("inf"))
        assert m.value == 5.0 and m.count == 2
        path = tmp_path / "clock_state.json"
        path.write_text("not json")
        state = ClockState(path, 4.0, 0.2)                                 # falls back on the seeds
        assert state.pysr_overhead.value == 4.0 and state.pricing_reserve.value == 0.2
        state.save()
        assert ClockState(path, 1.0, 1.0).pysr_overhead.value == 4.0
        ClockState(None, 1.0, 1.0).save()                                  # nothing to persist: a no-op


class TestConfig:
    def test_validation(self):
        with pytest.raises(ValueError, match="landing_tolerance"):
            HybridConfig(budget_s=100.0, ratio=0.5, landing_tolerance=0.0)
        with pytest.raises(ValueError, match="ratio"):
            HybridConfig(budget_s=100.0, ratio=1.5)
        with pytest.raises(ValueError, match="budget_s"):
            HybridConfig(budget_s=0.0, ratio=0.5)
        with pytest.raises(ValueError, match="unknown hybrid option"):
            HybridConfig.from_mapping({"budget_s": 100.0, "ratio": 0.5, "snapshot_dir": "x"})

    def test_budget_ladder(self):
        """A T-curve at a fixed r: one generation pass snapshots at every (1 - r) T of the ladder."""
        reg = HybridRegressor(object(), HybridConfig(budget_s=30.0, ratio=0.3, budgets=[15.0, 30.0, 60.0, 120.0]))
        assert reg.config.budgets == [15.0, 30.0, 60.0, 120.0]
        assert reg.time_targets() == [pytest.approx(0.7 * t) for t in (15.0, 30.0, 60.0, 120.0)]
        assert reg.pysr_seconds(0.3, 120.0) == pytest.approx(36.0) and reg.generation_seconds(0.3) == pytest.approx(21.0)
        with pytest.raises(ValueError, match="budget"):
            HybridConfig(budget_s=30.0, ratio=0.3, budgets=[0.0])
        both = HybridConfig(budget_s=100.0, ratio=0.5, ratios=[0.0, 0.5], budgets=[50.0, 100.0])
        assert HybridRegressor(object(), both).time_targets() == [25.0, 50.0, 100.0]     # r = 0 at 50 and 100, r = 0.5 at 50 and 100

    def test_from_mapping(self):
        cfg = HybridConfig.from_mapping({"budget_s": 100, "ratio": "0.5", "ratios": [0.0, 1.0], "k_seeds": "10",
                                         "pysr": {"maxsize": 25, "warmup": False, "populations": 8}})
        assert cfg.ratios == [0.0, 0.5, 1.0] and cfg.k_seeds == 10 and cfg.budget_s == 100.0
        assert isinstance(cfg.pysr, PySRSettings) and cfg.pysr.maxsize == 25 and not cfg.pysr.warmup
        assert cfg.pysr.extra == {"populations": 8}                        # any other key is a PySRRegressor kwarg
