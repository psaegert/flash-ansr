import importlib
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import yaml


run_config = importlib.import_module("flash_ansr.eval.run_config")


def _write_results(path, length: int) -> None:
    payload = {"expression": list(range(length)), "y_pred": list(range(length))}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_build_evaluation_run_short_circuits_when_limit_completed(tmp_path, monkeypatch):
    results_path = tmp_path / "existing.pkl"
    _write_results(results_path, length=3)

    flags = {"data": False, "adapter": False}

    def fake_build_data_source(*args, **kwargs):  # pragma: no cover - should not run
        flags["data"] = True
        return object(), {}

    def fake_build_model_adapter(*args, **kwargs):  # pragma: no cover - should not run
        flags["adapter"] = True
        return object()

    monkeypatch.setattr(run_config, "_build_data_source", fake_build_data_source)
    monkeypatch.setattr(run_config, "_build_model_adapter", fake_build_model_adapter)

    plan = run_config.build_evaluation_run(
        {
            "run": {
                "data_source": {"type": "skeleton_dataset", "dataset": "ignored"},
                "model_adapter": {"type": "flash_ansr"},
                "runner": {
                    "limit": 3,
                    "output": str(results_path),
                    "resume": True,
                },
            }
        }
    )

    assert plan.completed is True
    assert plan.engine is None
    assert plan.existing_results == 3
    assert flags["data"] is False
    assert flags["adapter"] is False


def test_build_evaluation_run_constructs_engine_with_remaining(tmp_path, monkeypatch):
    results_path = tmp_path / "partial.pkl"
    _write_results(results_path, length=2)

    fake_source = object()
    fake_adapter = object()
    captured = {}

    def fake_build_data_source(config, *, target_size_override, skip, preloaded_assets=None):
        captured["data_cfg"] = config
        captured["target_size"] = target_size_override
        captured["skip"] = skip
        return fake_source, {"dataset": SimpleNamespace()}

    def fake_build_model_adapter(config, *, context):
        captured["adapter_cfg"] = config
        captured["context"] = context
        return fake_adapter

    class DummyEngine:
        def __init__(self, *, data_source, model_adapter, result_store):
            self.data_source = data_source
            self.model_adapter = model_adapter
            self.result_store = result_store

    monkeypatch.setattr(run_config, "_build_data_source", fake_build_data_source)
    monkeypatch.setattr(run_config, "_build_model_adapter", fake_build_model_adapter)
    monkeypatch.setattr(run_config, "EvaluationEngine", DummyEngine)

    plan = run_config.build_evaluation_run(
        {
            "run": {
                "data_source": {"type": "skeleton_dataset", "dataset": "ignored", "target_size": 99},
                "model_adapter": {"type": "flash_ansr", "extra": True},
                "runner": {
                    "limit": 5,
                    "save_every": 2,
                    "output": str(results_path),
                    "resume": True,
                },
            }
        }
    )

    assert plan.completed is False
    assert plan.remaining == 3
    assert plan.save_every == 2
    assert plan.total_limit == 5
    assert isinstance(plan.engine, DummyEngine)
    assert plan.engine.data_source is fake_source
    assert plan.engine.model_adapter is fake_adapter
    assert plan.engine.result_store.size == 2
    assert captured["target_size"] == 3
    assert captured["skip"] == 2
    assert captured["context"] == {"dataset": SimpleNamespace()}


def test_build_evaluation_run_fastsrb_short_circuits_when_completed(tmp_path, monkeypatch):
    results_path = tmp_path / "existing.pkl"
    _write_results(results_path, length=6)

    class FakeBenchmark:
        def __init__(self, path, random_state=None):
            self.path = path
            self.random_state = random_state

        def equation_ids(self):
            return ["eq1", "eq2", "eq3"]

    monkeypatch.setattr(run_config, "FastSRBBenchmark", FakeBenchmark)
    monkeypatch.setattr(run_config, "_build_data_source", _raise_runtime_error)

    plan = run_config.build_evaluation_run(
        {
            "run": {
                "data_source": {
                    "type": "fastsrb",
                    "benchmark_path": "bench.yaml",
                    "datasets_per_expression": 2,
                },
                "model_adapter": {"type": "flash_ansr"},
                "runner": {
                    "output": str(results_path),
                    "resume": True,
                },
            }
        }
    )

    assert plan.completed is True
    assert plan.engine is None
    assert plan.total_limit == 6


def _raise_runtime_error(*_args, **_kwargs):  # pragma: no cover - helper to ensure builders are not invoked
    raise RuntimeError("data source should not be constructed when evaluation is complete")


def test_build_evaluation_run_fastsrb_infers_remaining(tmp_path, monkeypatch):
    results_path = tmp_path / "partial.pkl"
    _write_results(results_path, length=4)

    class FakeBenchmark:
        def __init__(self, path, random_state=None):
            self.path = path
            self.random_state = random_state

        def equation_ids(self):
            return ["eq1", "eq2", "eq3"]

    class DummySource:
        def __init__(self, pending):
            self._pending = pending or 0

        def size_hint(self):
            return self._pending

    class DummyEngine:
        def __init__(self, *, data_source, model_adapter, result_store):
            self.data_source = data_source
            self.model_adapter = model_adapter
            self.result_store = result_store

    monkeypatch.setattr(run_config, "FastSRBBenchmark", FakeBenchmark)

    captured: dict[str, Any] = {}

    def fake_build_data_source(config, *, target_size_override, skip, preloaded_assets=None):
        captured["target_size"] = target_size_override
        captured["skip"] = skip
        return DummySource(target_size_override), {}

    def fake_build_model_adapter(config, *, context):
        captured["adapter_cfg"] = config
        captured["context"] = context
        return object()

    monkeypatch.setattr(run_config, "_build_data_source", fake_build_data_source)
    monkeypatch.setattr(run_config, "_build_model_adapter", fake_build_model_adapter)
    monkeypatch.setattr(run_config, "EvaluationEngine", DummyEngine)

    plan = run_config.build_evaluation_run(
        {
            "run": {
                "data_source": {
                    "type": "fastsrb",
                    "benchmark_path": "bench.yaml",
                    "datasets_per_expression": 2,
                },
                "model_adapter": {"type": "flash_ansr"},
                "runner": {
                    "output": str(results_path),
                    "resume": True,
                },
            }
        }
    )

    assert plan.remaining == 2
    assert plan.total_limit == 6
    assert captured["target_size"] == 2
    assert plan.engine is not None
    assert isinstance(plan.engine, DummyEngine)


def test_build_evaluation_run_selects_named_experiment(tmp_path, monkeypatch):
    fake_source = object()
    fake_adapter = object()
    captured: dict[str, Any] = {}

    def fake_build_data_source(config, *, target_size_override, skip, preloaded_assets=None):
        captured["data_cfg"] = config
        captured["target"] = target_size_override
        captured["skip"] = skip
        return fake_source, {"dataset": SimpleNamespace()}

    def fake_build_model_adapter(config, *, context):
        captured["adapter_cfg"] = config
        captured["context"] = context
        return fake_adapter

    class DummyEngine:
        def __init__(self, *, data_source, model_adapter, result_store):
            self.data_source = data_source
            self.model_adapter = model_adapter
            self.result_store = result_store

    monkeypatch.setattr(run_config, "_build_data_source", fake_build_data_source)
    monkeypatch.setattr(run_config, "_build_model_adapter", fake_build_model_adapter)
    monkeypatch.setattr(run_config, "EvaluationEngine", DummyEngine)

    plan = run_config.build_evaluation_run(
        {
            "default_experiment": "fast",
            "experiments": {
                "fast": {
                    "run": {
                        "data_source": {"type": "skeleton_dataset", "dataset": "dataset_a", "target_size": 5},
                        "model_adapter": {"type": "flash_ansr", "alpha": 1},
                        "runner": {"limit": 2},
                    }
                },
                "slow": {"run": {"data_source": {"type": "skeleton_dataset", "dataset": "dataset_b"}, "model_adapter": {"type": "flash_ansr"}}},
            },
        },
        experiment="fast",
    )

    assert plan.engine is not None
    assert captured["data_cfg"]["dataset"] == "dataset_a"
    assert captured["adapter_cfg"]["alpha"] == 1
    assert captured["context"] == {"dataset": SimpleNamespace()}


def test_build_evaluation_run_requires_experiment_name(tmp_path):
    with pytest.raises(ValueError):
        run_config.build_evaluation_run(
            {
                "experiments": {
                    "only": {
                        "run": {
                            "data_source": {"type": "skeleton_dataset", "dataset": "dataset_a"},
                            "model_adapter": {"type": "flash_ansr"},
                        }
                    }
                }
            }
        )


def test_build_data_source_fastsrb_uses_available_pool(monkeypatch):
    class FakeBenchmark:
        def __init__(self, path, random_state=None):
            self.path = path
            self.random_state = random_state
            self._ids = ["eq1", "eq2", "eq3"]

        def equation_ids(self):
            return list(self._ids)

    created_kwargs = {}

    class FakeSource:
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(run_config, "FastSRBBenchmark", FakeBenchmark)
    monkeypatch.setattr(run_config, "FastSRBSource", FakeSource)

    config = {
        "type": "fastsrb",
        "benchmark_path": "/tmp/bench.yaml",
        "count": 2,
        "support_points": 128,
        "method": "random",
        "eq_ids": "eq1, eq2",
        "noise_level": 0.1,
    }

    source, context = run_config._build_data_source(config, target_size_override=None, skip=1, preloaded_assets=None)

    assert isinstance(context["benchmark"], FakeBenchmark)
    assert created_kwargs["benchmark"] is context["benchmark"]
    assert created_kwargs["datasets_per_expression"] == 2
    assert created_kwargs["support_points"] == 128
    assert created_kwargs["eq_ids"] == ["eq1", "eq2"]
    # total available = len(eq_ids)*count = 4, target_size = total - skip
    assert created_kwargs["target_size"] == 3
    assert isinstance(source, FakeSource)


def test_parse_equation_ids_accepts_sequences():
    assert run_config._parse_equation_ids([1, "x"]) == ["1", "x"]
    assert run_config._parse_equation_ids("a, b c") == ["a", "b", "c"]
    assert run_config._parse_equation_ids(None) is None


def test_flash_ansr_generation_overrides(tmp_path, monkeypatch):
    eval_cfg = {
        "evaluation": {
            "n_restarts": 1,
            "refiner_method": "curve_fit_lm",
            "refiner_p0_noise": "normal",
            "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 1.0},
            "parsimony": 0.1,
            "device": "cuda",
            "refiner_workers": None,
            "generation_config": {
                "method": "softmax_sampling",
                "kwargs": {"choices": 8, "max_len": 16},
            },
        }
    }
    eval_path = tmp_path / "evaluation.yaml"
    with eval_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(eval_cfg, handle)

    captured: dict[str, Any] = {}

    def fake_create_generation_config(method, **kwargs):
        captured["method"] = method
        captured["kwargs"] = kwargs
        return {"method": method, "kwargs": kwargs}

    class FakeFlashANSR:
        @staticmethod
        def load(*, directory, generation_config, **kwargs):
            captured["flash_ansr_gen"] = generation_config
            captured["flash_ansr_dir"] = directory
            return SimpleNamespace()

    class DummyAdapter:
        def __init__(self, model, device, complexity, refiner_workers):
            self.model = model
            self.device = device
            self.complexity = complexity
            self.refiner_workers = refiner_workers

    monkeypatch.setattr(run_config, "create_generation_config", fake_create_generation_config)
    monkeypatch.setattr(run_config, "FlashANSR", FakeFlashANSR)
    monkeypatch.setattr(run_config, "FlashANSRAdapter", DummyAdapter)

    adapter = run_config._build_flash_ansr_adapter(
        {
            "model_path": str(tmp_path),
            "evaluation_config": str(eval_path),
            "generation_overrides": {"kwargs": {"choices": 2}},
        },
        context={},
    )

    assert isinstance(adapter, DummyAdapter)
    assert captured["method"] == "softmax_sampling"
    assert captured["kwargs"]["choices"] == 2
    assert captured["flash_ansr_gen"]["kwargs"]["choices"] == 2


def test_skeleton_dataset_max_trials(monkeypatch):
    dataset = SimpleNamespace()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(run_config, "_load_dataset", lambda spec: dataset)

    class DummySource:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run_config, "SkeletonDatasetSource", DummySource)

    source, context = run_config._build_data_source(
        {
            "type": "skeleton_dataset",
            "dataset": "ignored",
            "max_trials": 250,
        },
        target_size_override=10,
        skip=0,
        preloaded_assets=None,
    )

    assert isinstance(source, DummySource)
    assert context["dataset"] is dataset
    assert captured["max_trials"] == 250


def test_flash_ansr_inline_evaluation_config(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_create_generation_config(method, **kwargs):
        captured["method"] = method
        captured["kwargs"] = kwargs
        return {"method": method, "kwargs": kwargs}

    class FakeFlashANSR:
        @staticmethod
        def load(*, directory, generation_config, **kwargs):
            captured["flash_ansr_gen"] = generation_config
            captured["flash_ansr_dir"] = directory
            captured["flash_ansr_kwargs"] = kwargs
            return SimpleNamespace()

    class DummyAdapter:
        def __init__(self, model, device, complexity, refiner_workers):
            self.model = model
            self.device = device
            self.complexity = complexity
            self.refiner_workers = refiner_workers

    monkeypatch.setattr(run_config, "create_generation_config", fake_create_generation_config)
    monkeypatch.setattr(run_config, "FlashANSR", FakeFlashANSR)
    monkeypatch.setattr(run_config, "FlashANSRAdapter", DummyAdapter)

    inline_cfg = {
        "n_restarts": 2,
        "refiner_method": "curve_fit_lm",
        "refiner_p0_noise": "normal",
        "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 1.0},
        "parsimony": 0.1,
        "device": "cuda",
        "generation_config": {
            "method": "softmax_sampling",
            "kwargs": {"choices": 4, "max_len": 16},
        },
    }

    adapter = run_config._build_flash_ansr_adapter(
        {
            "model_path": "./models/v23",
            "evaluation_config": inline_cfg,
            "complexity": "none",
            "device": "cuda",
        },
        context={},
    )

    assert isinstance(adapter, DummyAdapter)
    assert captured["method"] == "softmax_sampling"
    assert captured["kwargs"]["choices"] == 4
    assert captured["flash_ansr_gen"]["kwargs"]["choices"] == 4
    assert captured["flash_ansr_kwargs"]["n_restarts"] == 2


def test_pysr_adapter_accepts_non_dataset_with_explicit_engine(monkeypatch):
    loaded = {}

    class DummyEngineLoader:
        @staticmethod
        def load(path, install=True):
            loaded["path"] = path
            loaded["install"] = install
            return SimpleNamespace(name="engine")

    class DummyAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(run_config, "SimpliPyEngine", DummyEngineLoader)
    monkeypatch.setattr(run_config, "PySRAdapter", DummyAdapter)

    adapter = run_config._build_pysr_adapter(
        {"niterations": 2, "simplipy_engine": "dev_7-3"},
        context={},
    )

    assert isinstance(adapter, DummyAdapter)
    assert loaded["path"] == "dev_7-3"


def test_pysr_adapter_without_engine_raises(monkeypatch):
    class DummyAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(run_config, "PySRAdapter", DummyAdapter)

    with pytest.raises(ValueError):
        run_config._build_pysr_adapter({"niterations": 1}, context={})


def test_completed_skeleton_dataset_skips_run(tmp_path, monkeypatch):
    results_path = tmp_path / "existing.pkl"
    _write_results(results_path, length=6)

    class DummyPool:
        def __len__(self):
            return 3

    dataset = SimpleNamespace(skeleton_pool=DummyPool())

    monkeypatch.setattr(run_config, "_load_dataset", lambda spec: dataset)
    monkeypatch.setattr(run_config, "_build_data_source", _raise_runtime_error)

    plan = run_config.build_evaluation_run(
        {
            "run": {
                "data_source": {
                    "type": "skeleton_dataset",
                    "dataset": {"path": "ignored"},
                    "datasets_per_expression": 2,
                },
                "model_adapter": {"type": "flash_ansr"},
                "runner": {
                    "output": str(results_path),
                    "resume": True,
                },
            }
        }
    )

    assert plan.completed is True
    assert plan.engine is None
    assert plan.total_limit == 6


def test_partial_skeleton_dataset_infers_remaining(tmp_path, monkeypatch):
    results_path = tmp_path / "partial.pkl"
    _write_results(results_path, length=4)

    class DummyPool:
        def __len__(self):
            return 3

    dataset = SimpleNamespace(skeleton_pool=DummyPool())

    class DummySource:
        def __init__(self, *, target_size):
            self._target = target_size

        def size_hint(self):
            return self._target

    class DummyEngine:
        def __init__(self, *, data_source, model_adapter, result_store):
            self.data_source = data_source
            self.model_adapter = model_adapter
            self.result_store = result_store

    monkeypatch.setattr(run_config, "_load_dataset", lambda spec: dataset)

    captured: dict[str, Any] = {}

    def fake_build_data_source(config, *, target_size_override, skip, preloaded_assets=None):
        captured["target_size"] = target_size_override
        captured["skip"] = skip
        assert preloaded_assets and preloaded_assets.get("dataset") is dataset
        return DummySource(target_size=target_size_override), {"dataset": dataset}

    def fake_build_model_adapter(config, *, context):
        captured["adapter_cfg"] = config
        captured["context"] = context
        return object()

    monkeypatch.setattr(run_config, "_build_data_source", fake_build_data_source)
    monkeypatch.setattr(run_config, "_build_model_adapter", fake_build_model_adapter)
    monkeypatch.setattr(run_config, "EvaluationEngine", DummyEngine)

    plan = run_config.build_evaluation_run(
        {
            "run": {
                "data_source": {
                    "type": "skeleton_dataset",
                    "dataset": {"path": "ignored"},
                    "datasets_per_expression": 2,
                },
                "model_adapter": {"type": "flash_ansr"},
                "runner": {
                    "output": str(results_path),
                    "resume": True,
                },
            }
        }
    )

    assert plan.remaining == 2
    assert plan.total_limit == 6
    assert captured["target_size"] == 2
    assert plan.engine is not None
