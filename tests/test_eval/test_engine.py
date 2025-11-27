import pickle

import numpy as np

from flash_ansr.eval.core import EvaluationSample, EvaluationResult
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.result_store import ResultStore


class _ListDataSource:
    def __init__(self, samples, *, skip: int = 0):
        self._samples = samples
        self._skip = max(0, skip)

    def __iter__(self):
        yield from self._samples[self._skip:]

    def size_hint(self):
        return max(0, len(self._samples) - self._skip)


class _FlakyAdapter:
    def __init__(self):
        self.calls = 0

    def evaluate_sample(self, sample):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        record = sample.clone_metadata()
        record["prediction_success"] = True
        return EvaluationResult(record)


def _dummy_sample(idx: int) -> EvaluationSample:
    data = np.zeros((2, 1), dtype=np.float32)
    metadata = {"sample_id": idx}
    return EvaluationSample(
        x_support=data,
        y_support=data,
        x_validation=data,
        y_validation=data,
        metadata=metadata,
    )


def test_engine_records_placeholder_on_adapter_exception(capsys):
    samples = [_dummy_sample(0), _dummy_sample(1)]
    source = _ListDataSource(samples)
    adapter = _FlakyAdapter()
    store = ResultStore()

    engine = EvaluationEngine(data_source=source, model_adapter=adapter, result_store=store)
    snapshot = engine.run(progress=False, verbose=False, summary_interval=1)
    output = capsys.readouterr().out

    assert len(snapshot["placeholder"]) == 2
    assert snapshot["placeholder"][0] is True
    assert snapshot["prediction_success"][0] is False
    assert snapshot["placeholder_reason"][0] == "adapter_exception"
    assert snapshot["sample_id"] == [0, 1]
    assert snapshot["prediction_success"][1] is True
    assert "Placeholder #1 recorded" in output
    assert "Final evaluation summary" in output


def test_engine_can_resume_from_saved_results(tmp_path):
    samples = [_dummy_sample(idx) for idx in range(3)]
    output_path = tmp_path / "results.pkl"

    first_store = ResultStore()
    flaky_adapter = _FlakyAdapter()
    first_engine = EvaluationEngine(
        data_source=_ListDataSource(samples),
        model_adapter=flaky_adapter,
        result_store=first_store,
    )

    snapshot = first_engine.run(
        limit=2,
        save_every=1,
        output_path=str(output_path),
        progress=False,
        verbose=False,
        summary_interval=1,
    )

    assert output_path.exists()
    assert snapshot["sample_id"] == [0, 1]
    assert snapshot["placeholder"] == [True, False]
    assert snapshot["placeholder_reason"][0] == "adapter_exception"

    with output_path.open("rb") as handle:
        payload = pickle.load(handle)

    resumed_store = ResultStore(payload)
    assert resumed_store.size == 2

    resumed_adapter = _FlakyAdapter()
    resumed_adapter.calls = 1  # Skip the initial failure on resume

    resumed_engine = EvaluationEngine(
        data_source=_ListDataSource(samples, skip=resumed_store.size),
        model_adapter=resumed_adapter,
        result_store=resumed_store,
    )

    final_snapshot = resumed_engine.run(
        progress=False,
        verbose=False,
        summary_interval=1,
    )

    assert final_snapshot["sample_id"] == [0, 1, 2]
    assert final_snapshot["placeholder"] == [True, False, False]
    assert final_snapshot["prediction_success"] == [False, True, True]
    assert final_snapshot["placeholder_reason"][0] == "adapter_exception"
