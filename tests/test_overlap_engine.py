"""Unit tests for OverlappedEvaluationEngine (GPU-free, fake adapter/model/source).

These exercise the engine's control flow deterministically: the support gate + serial fallback, that
overlap commits in sample order and matches the serial engine exactly on a deterministic adapter,
placeholder handling, and bidirectional shutdown (a producer-side failure surfaces; a consumer-side
failure does not leave the producer thread hung)."""
from __future__ import annotations

import time
import types
from concurrent.futures.process import BrokenProcessPool

import pytest

from flash_ansr.eval.core import EvaluationResult
from flash_ansr.eval.engine import EvaluationEngine, OverlappedEvaluationEngine
from flash_ansr.eval.result_store import ResultStore


class FakeSample:
    def __init__(self, sid: int, *, is_placeholder: bool = False):
        self.sid = sid
        self.is_placeholder = is_placeholder
        self.placeholder_reason = "fake" if is_placeholder else None

    def clone_metadata(self):
        return {"sid": self.sid}


class FakeSource:
    def __init__(self, samples, *, raise_after: int | None = None):
        self._samples = samples
        self._raise_after = raise_after

    def __iter__(self):
        for i, s in enumerate(self._samples):
            if self._raise_after is not None and i == self._raise_after:
                raise RuntimeError("data source exploded")
            yield s

    def size_hint(self):
        return len(self._samples)

    def prepare(self, *, adapter=None):
        return None


def _fake_model(*, refine_pool=True, prune=0.0, method="softmax_sampling", choices=1024):
    return types.SimpleNamespace(
        _refine_pool=object() if refine_pool else None,
        _overlap_mode=False,
        prune_constant_budget=prune,
        generation_config=types.SimpleNamespace(method=method, choices=choices),
        refiner_workers=8,
        parallel_simplify=True,
    )


class FakeAdapter:
    """Deterministic two-phase adapter: refine doubles the sample id; order is verifiable via `sid`."""

    def __init__(self, model, *, gen_raises_on=None, refine_raises_on=None, pool_breaks_on=None,
                 gen_pool_breaks_on=None):
        self.model = model
        self._gen_raises_on = set(gen_raises_on or [])
        self._refine_raises_on = set(refine_raises_on or [])
        self._pool_breaks_on = set(pool_breaks_on or [])
        self._gen_pool_breaks_on = set(gen_pool_breaks_on or [])

    def prepare(self, *, data_source=None):
        return None

    def generate_phase(self, sample):
        record = sample.clone_metadata()
        if sample.sid in self._gen_pool_breaks_on:
            # The producer-side BrokenProcessPool (the overlap simplify pass dies) -- the dominant
            # high-c abort timing Step 4b enables.
            raise BrokenProcessPool(f"gen-side pool died on {sample.sid}")
        if sample.sid in self._gen_raises_on:
            raise RuntimeError(f"gen boom on {sample.sid}")
        # GenState-faithful: carries `memory_for_scoring` (the engine drops it to bound GPU residency).
        gen_state = types.SimpleNamespace(sid=sample.sid, memory_for_scoring="gpu-tensor")
        return record, gen_state, 0.0

    def refine_extract_phase(self, sample, record, gen_state, fit_t0, *, wall_clock=False, refine_seed=None):
        if gen_state is None:
            return EvaluationResult(record)
        if sample.sid in self._pool_breaks_on:
            raise BrokenProcessPool(f"pool died on {sample.sid}")
        if sample.sid in self._refine_raises_on:
            raise RuntimeError(f"refine boom on {sample.sid}")
        record["value"] = gen_state.sid * 2
        record["prediction_success"] = True
        return EvaluationResult(record)

    def evaluate_sample(self, sample):
        record, gs, t0 = self.generate_phase(sample)
        return self.refine_extract_phase(sample, record, gs, t0, wall_clock=True)


def _run(engine_cls, samples, **src_kw):
    adapter = FakeAdapter(_fake_model())
    src = FakeSource(samples, **src_kw)
    eng = engine_cls(src, adapter, result_store=ResultStore())
    return eng.run(limit=len(samples), verbose=False, progress=False,
                   log_placeholders=False, summary_interval=None)


def test_overlap_matches_serial_and_preserves_order():
    samples = [FakeSample(i) for i in range(20)]
    serial = _run(EvaluationEngine, samples)
    overlap = _run(OverlappedEvaluationEngine, samples)
    assert serial["sid"] == list(range(20))            # order preserved
    assert overlap["sid"] == serial["sid"]             # same order
    assert overlap["value"] == serial["value"]         # same (deterministic) results
    assert overlap["value"] == [i * 2 for i in range(20)]


@pytest.mark.parametrize("model_kw,reason", [
    (dict(refine_pool=False), "no persistent refine pool"),
    (dict(prune=16.0), "prune"),
    (dict(method="mcts"), "softmax"),
])
def test_falls_back_to_serial_when_unsupported(model_kw, reason):
    samples = [FakeSample(i) for i in range(8)]
    adapter = FakeAdapter(_fake_model(**model_kw))
    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=ResultStore())
    ok, why = eng._overlap_supported()
    assert ok is False and reason in why
    with pytest.warns(RuntimeWarning):
        snap = eng.run(limit=len(samples), verbose=True, progress=False,
                       log_placeholders=False, summary_interval=None)
    assert snap["value"] == [i * 2 for i in range(8)]   # serial fallback still correct


@pytest.mark.parametrize("choices", [4096, 8192, 262144])
def test_overlap_engages_at_high_c(choices):
    """Step 4b: choices >= the simplify-parallel threshold (4096) is NO LONGER gated. The producer's
    post-generation simplify and the consumer's refine share the pool safely (generation dominates at
    high c, so they do not contend), so overlap engages and matches serial exactly."""
    samples = [FakeSample(i) for i in range(12)]
    adapter = FakeAdapter(_fake_model(choices=choices))
    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=ResultStore())
    ok, why = eng._overlap_supported()
    assert ok is True, f"overlap should engage at high c after Step 4b (got: {why!r})"
    snap = eng.run(limit=len(samples), verbose=False, progress=False,
                   log_placeholders=False, summary_interval=None)
    assert snap["sid"] == list(range(12))               # order preserved
    assert snap["value"] == [i * 2 for i in range(12)]  # results match the deterministic adapter


def test_adapter_without_phase_split_falls_back():
    eng = OverlappedEvaluationEngine(FakeSource([]), object(), result_store=ResultStore())
    ok, why = eng._overlap_supported()
    assert ok is False and "phase split" in why


def test_placeholder_handling():
    samples = [FakeSample(0), FakeSample(1, is_placeholder=True), FakeSample(2)]
    snap = _run(OverlappedEvaluationEngine, samples)
    assert snap["sid"] == [0, 1, 2]
    assert snap["placeholder"][1] is True
    assert snap["placeholder"][0] in (False, None)


def test_generate_phase_error_recorded_not_fatal():
    # An unexpected generate_phase exception is handled per-problem (placeholder), not a fatal crash.
    samples = [FakeSample(i) for i in range(5)]
    adapter = FakeAdapter(_fake_model(), gen_raises_on=[2])
    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=ResultStore())
    snap = eng.run(limit=5, verbose=False, progress=False, log_placeholders=False, summary_interval=None)
    assert snap["sid"] == [0, 1, 2, 3, 4]
    assert snap["placeholder"][2] is True               # the failed problem is a placeholder
    assert snap["value"][0] == 0 and snap["value"][4] == 8


def test_producer_data_source_failure_surfaces():
    # A failure in the data-source iteration (outside the per-sample guard) must surface, after the
    # already-produced results are committed.
    samples = [FakeSample(i) for i in range(5)]
    adapter = FakeAdapter(_fake_model())
    store = ResultStore()
    eng = OverlappedEvaluationEngine(FakeSource(samples, raise_after=3), adapter, result_store=store)
    with pytest.raises(RuntimeError, match="data source exploded"):
        eng.run(limit=5, verbose=False, progress=False, log_placeholders=False, summary_interval=None)
    assert store.size == 3                               # the 3 produced before the explosion committed


def test_pool_break_tears_down_with_clear_error_and_commits_prior():
    # A BrokenProcessPool during refine must NOT be swallowed as a per-problem error: the engine stops,
    # quiesces the producer, commits the contiguous prior results, and raises an actionable error.
    samples = [FakeSample(i) for i in range(8)]
    adapter = FakeAdapter(_fake_model(), pool_breaks_on=[3])
    store = ResultStore()
    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=store)
    with pytest.raises(RuntimeError, match="persistent refine pool broke"):
        eng.run(limit=8, verbose=False, progress=False, log_placeholders=False, summary_interval=None)
    assert store.size == 3                                # problems 0,1,2 committed before the break on 3
    # the engine reset the model's overlap flag and left no producer thread alive
    assert adapter.model._overlap_mode is False
    alive = [t for t in __import__("threading").enumerate() if t.name == "overlap-gpu-producer" and t.is_alive()]
    assert not alive


class _SaveSpyStore(ResultStore):
    """Counts save() calls (without writing to disk) so a test can assert a checkpoint was attempted."""

    def __init__(self):
        super().__init__()
        self.save_calls = 0

    def save(self, *args, **kwargs):
        self.save_calls += 1  # record only; no file IO in the unit test


def test_producer_pool_break_checkpoints_then_aborts():
    # Step-4b review regression: a BrokenProcessPool from the PRODUCER's generate_phase (the overlap
    # simplify pass dying -- the DOMINANT high-c abort timing 4b enables) routes to exc_box, which must
    # checkpoint the contiguous committed results BEFORE surfacing -- symmetric with the consumer/
    # pool_broke path. Without the fix, save_every=None + output_path would discard the whole run.
    samples = [FakeSample(i) for i in range(8)]
    adapter = FakeAdapter(_fake_model(), gen_pool_breaks_on=[5])
    store = _SaveSpyStore()
    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=store)
    with pytest.raises(BrokenProcessPool):
        eng.run(limit=8, output_path="results/_test_producer_break.pkl",
                verbose=False, progress=False, log_placeholders=False, summary_interval=None)
    assert store.save_calls >= 1                          # THE FIX: exc_box abort path checkpoints first
    assert store.size >= 1                                # contiguous prior results were committed
    assert adapter.model._overlap_mode is False           # overlap flag reset on teardown
    alive = [t for t in __import__("threading").enumerate() if t.name == "overlap-gpu-producer" and t.is_alive()]
    assert not alive                                      # no producer thread left hung


def test_overlap_mode_flag_set_during_run_and_reset_after():
    samples = [FakeSample(i) for i in range(5)]
    adapter = FakeAdapter(_fake_model())
    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=ResultStore())
    assert adapter.model._overlap_mode is False
    eng.run(limit=5, verbose=False, progress=False, log_placeholders=False, summary_interval=None)
    assert adapter.model._overlap_mode is False           # reset after a normal run


def test_consumer_failure_does_not_hang_producer():
    # If the consumer raises (here: result_store.append blows up), the producer must not be left hung;
    # the bounded drain+join in the finally retires it.
    samples = [FakeSample(i) for i in range(30)]
    adapter = FakeAdapter(_fake_model())

    class ExplodingStore(ResultStore):
        def __init__(self):
            super().__init__()
            self._n = 0

        def append(self, record):
            self._n += 1
            if self._n == 3:
                raise RuntimeError("consumer boom")
            return super().append(record)

    eng = OverlappedEvaluationEngine(FakeSource(samples), adapter, result_store=ExplodingStore())
    t0 = time.time()
    with pytest.raises(RuntimeError, match="consumer boom"):
        eng.run(limit=30, verbose=False, progress=False, log_placeholders=False, summary_interval=None)
    assert time.time() - t0 < 15.0                       # bounded shutdown (no deadlock)
    # the producer thread must not be left alive
    alive = [t for t in __import__("threading").enumerate() if t.name == "overlap-gpu-producer" and t.is_alive()]
    assert not alive


# --------------------------------------------------------------------------- run_config wiring
def test_select_engine_cls_overlap_when_pool_present():
    """run_config picks the overlap engine iff a persistent refine pool was actually created."""
    from flash_ansr.eval.run_config import _select_engine_cls
    assert _select_engine_cls(FakeAdapter(_fake_model(refine_pool=True))) is OverlappedEvaluationEngine


def test_select_engine_cls_serial_when_no_pool():
    """Default (no persistent pool) -> the serial engine, byte-identical to the historical default."""
    from flash_ansr.eval.run_config import _select_engine_cls
    assert _select_engine_cls(FakeAdapter(_fake_model(refine_pool=False))) is EvaluationEngine


def test_select_engine_cls_serial_when_no_model():
    """A non-FlashANSR adapter (no .model, e.g. PySR) -> the serial engine."""
    from flash_ansr.eval.run_config import _select_engine_cls
    assert _select_engine_cls(types.SimpleNamespace(model=None)) is EvaluationEngine
