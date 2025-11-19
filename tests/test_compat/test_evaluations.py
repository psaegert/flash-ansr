from pathlib import Path

import pytest

from flash_ansr.compat.evaluation_pysr import PySREvaluation
from flash_ansr.compat.evaluation_nesymres import NeSymReSEvaluation


class DummyDataset:
    def __init__(self, pool_size: int, simplipy_engine: str = "engine") -> None:
        self.skeleton_pool = [object() for _ in range(pool_size)]
        self.simplipy_engine = simplipy_engine


def test_pysr_from_config_extracts_nested_block() -> None:
    config = {
        "evaluation": {
            "n_support": 5,
            "noise_level": 0.125,
            "timeout_in_seconds": 90,
            "niterations": 77,
            "padding": False,
            "use_mult_div_operators": True,
        }
    }

    evaluation = PySREvaluation.from_config(config)

    assert evaluation.n_support == 5
    assert evaluation.noise_level == pytest.approx(0.125)
    assert evaluation.timeout_in_seconds == 90
    assert evaluation.niterations == 77
    assert evaluation.padding is False
    assert evaluation.use_mult_div_operators is True


def test_pysr_evaluate_short_circuits_when_results_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluation = PySREvaluation(n_support=3)
    dataset = DummyDataset(pool_size=2)

    precomputed = {
        "expression": ["a", "b"],
        "log_prob": [0.1, 0.2],
    }

    def _fail(*_args: object, **_kwargs: object) -> None:  # pragma: no cover - guard
        pytest.fail("evaluation should not build sources or adapters when already complete")

    monkeypatch.setattr("flash_ansr.compat.evaluation_pysr.SkeletonDatasetSource", _fail)
    monkeypatch.setattr("flash_ansr.compat.evaluation_pysr.PySRAdapter", _fail)
    monkeypatch.setattr("flash_ansr.compat.evaluation_pysr.EvaluationEngine", _fail)

    results = evaluation.evaluate(dataset=dataset, results_dict=precomputed)

    assert results["expression"] == ["a", "b"]
    assert results["log_prob"] == [0.1, 0.2]
    assert results["placeholder"] == [False, False]
    assert results["placeholder_reason"] == [None, None]


def test_pysr_evaluate_constructs_engine_and_returns_sorted_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluation = PySREvaluation(
        n_support=5,
        noise_level=0.05,
        timeout_in_seconds=15,
        niterations=30,
        padding=False,
        use_mult_div_operators=True,
    )
    dataset = DummyDataset(pool_size=5, simplipy_engine="simplipy-test")

    call_log: dict[str, dict[str, object]] = {}

    class DummySource:
        def __init__(self, dataset: DummyDataset, **kwargs: object) -> None:
            call_log["source"] = {"dataset": dataset, **kwargs}

    class DummyAdapter:
        def __init__(self, **kwargs: object) -> None:
            call_log["adapter"] = kwargs

    class DummyEngine:
        def __init__(self, data_source: DummySource, model_adapter: DummyAdapter, result_store) -> None:
            call_log["engine_init"] = {
                "data_source": data_source,
                "model_adapter": model_adapter,
                "result_store_size": result_store.size,
            }

        def run(self, **kwargs: object) -> dict[str, list[int]]:
            call_log["engine_run"] = kwargs
            return {"b": [2], "a": [1]}

    monkeypatch.setattr("flash_ansr.compat.evaluation_pysr.SkeletonDatasetSource", DummySource)
    monkeypatch.setattr("flash_ansr.compat.evaluation_pysr.PySRAdapter", DummyAdapter)
    monkeypatch.setattr("flash_ansr.compat.evaluation_pysr.EvaluationEngine", DummyEngine)

    output_path = tmp_path / "results.pkl"

    results = evaluation.evaluate(
        dataset=dataset,
        results_dict={"expression": ["seed"], "log_prob": [0.0]},
        size=3,
        save_every=1,
        output_file=str(output_path),
        verbose=False,
    )

    assert list(results.keys()) == ["a", "b"]
    assert call_log["source"]["target_size"] == 2
    assert call_log["source"]["n_support"] == 5
    assert call_log["source"]["noise_level"] == pytest.approx(0.05)
    assert call_log["adapter"]["simplipy_engine"] == "simplipy-test"
    assert call_log["adapter"]["timeout_in_seconds"] == 15
    assert call_log["adapter"]["niterations"] == 30
    assert call_log["adapter"]["use_mult_div_operators"] is True
    assert call_log["adapter"]["padding"] is False
    assert call_log["engine_init"]["result_store_size"] == 1
    assert call_log["engine_run"]["limit"] == 2
    assert call_log["engine_run"]["save_every"] == 1
    assert call_log["engine_run"]["output_path"] == str(output_path)
    assert call_log["engine_run"]["verbose"] is False
    assert call_log["engine_run"]["progress"] is False


def test_pysr_evaluate_requires_output_file_when_save_every_set() -> None:
    evaluation = PySREvaluation()
    dataset = DummyDataset(pool_size=1)

    with pytest.raises(ValueError, match="output_file"):
        evaluation.evaluate(dataset=dataset, save_every=2)


def test_nesymres_from_config_handles_optional_fields() -> None:
    config = {
        "evaluation": {
            "n_support": 8,
            "noise_level": 0.2,
            "beam_width": 11,
            "device": "cuda",
        }
    }

    evaluation = NeSymReSEvaluation.from_config(config)

    assert evaluation.n_support == 8
    assert evaluation.noise_level == pytest.approx(0.2)
    assert evaluation.beam_width == 11
    assert evaluation.device == "cuda"


def test_nesymres_evaluate_defaults_to_dataset_length(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluation = NeSymReSEvaluation(n_support=2, noise_level=0.0, beam_width=6, device="cpu")
    dataset = DummyDataset(pool_size=4)

    call_log: dict[str, dict[str, object]] = {}

    class DummySource:
        def __init__(self, dataset: DummyDataset, **kwargs: object) -> None:
            call_log["source"] = {"dataset": dataset, **kwargs}

    class DummyAdapter:
        def __init__(self, **kwargs: object) -> None:
            call_log["adapter"] = kwargs

    class DummyEngine:
        def __init__(self, data_source: DummySource, model_adapter: DummyAdapter, result_store) -> None:
            call_log["engine_init"] = {
                "data_source": data_source,
                "model_adapter": model_adapter,
                "result_store_size": result_store.size,
            }

        def run(self, **kwargs: object) -> dict[str, list[int]]:
            call_log["engine_run"] = kwargs
            return {"y": [1], "x": [2]}

    monkeypatch.setattr("flash_ansr.compat.evaluation_nesymres.SkeletonDatasetSource", DummySource)
    monkeypatch.setattr("flash_ansr.compat.evaluation_nesymres.NeSymReSAdapter", DummyAdapter)
    monkeypatch.setattr("flash_ansr.compat.evaluation_nesymres.EvaluationEngine", DummyEngine)

    results = evaluation.evaluate(
        model="model",
        fitfunc=lambda *_args, **_kwargs: None,
        simplipy_engine="simplipy",
        dataset=dataset,
        verbose=False,
    )

    assert list(results.keys()) == ["x", "y"]
    assert call_log["source"]["target_size"] == len(dataset.skeleton_pool)
    assert call_log["source"]["device"] == "cpu"
    assert call_log["adapter"]["model"] == "model"
    assert call_log["adapter"]["fitfunc"] is not None
    assert call_log["adapter"]["simplipy_engine"] == "simplipy"
    assert call_log["adapter"]["beam_width"] == 6
    assert call_log["engine_run"]["limit"] == len(dataset.skeleton_pool)
    assert call_log["engine_run"]["verbose"] is False
    assert call_log["engine_run"]["progress"] is False


def test_nesymres_evaluate_honors_explicit_size(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluation = NeSymReSEvaluation(n_support=1, noise_level=0.05, beam_width=None, device="cuda")
    dataset = DummyDataset(pool_size=10)

    class DummySource:
        def __init__(self, dataset: DummyDataset, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.dataset = dataset

    class DummyAdapter:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class DummyEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> dict[str, list[int]]:
            self.kwargs = kwargs
            return {"result": [1]}

    source_holder: dict[str, DummySource] = {}
    engine_holder: dict[str, DummyEngine] = {}

    def _source_factory(*args: object, **kwargs: object) -> DummySource:
        instance = DummySource(*args, **kwargs)
        source_holder["instance"] = instance
        return instance

    def _engine_factory(*args: object, **kwargs: object) -> DummyEngine:
        instance = DummyEngine(*args, **kwargs)
        engine_holder["instance"] = instance
        return instance

    monkeypatch.setattr("flash_ansr.compat.evaluation_nesymres.SkeletonDatasetSource", _source_factory)
    monkeypatch.setattr("flash_ansr.compat.evaluation_nesymres.NeSymReSAdapter", DummyAdapter)
    monkeypatch.setattr("flash_ansr.compat.evaluation_nesymres.EvaluationEngine", _engine_factory)

    eval_results = evaluation.evaluate(
        model="model",
        fitfunc=lambda *_args, **_kwargs: None,
        simplipy_engine="simplipy",
        dataset=dataset,
        size=3,
    )

    assert eval_results == {"result": [1]}
    assert source_holder["instance"].kwargs["target_size"] == 3
    assert engine_holder["instance"].kwargs["limit"] == 3
