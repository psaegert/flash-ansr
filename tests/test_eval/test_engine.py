import numpy as np

from flash_ansr.eval.core import EvaluationSample, EvaluationResult
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.result_store import ResultStore


class _ListDataSource:
    def __init__(self, samples):
        self._samples = samples

    def __iter__(self):
        yield from self._samples

    def size_hint(self):
        return len(self._samples)


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
