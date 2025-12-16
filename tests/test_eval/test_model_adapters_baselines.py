import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import SkeletonPool
from flash_ansr.baselines import BruteForceModel, SkeletonPoolModel
from flash_ansr.eval.core import EvaluationSample
from flash_ansr.eval.model_adapters import BruteForceAdapter, SkeletonPoolAdapter


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("dev_7-3", install=True)


def _build_toy_pool(engine: SimpliPyEngine) -> SkeletonPool:
    sample_strategy = {
        "n_operator_distribution": "equiprobable_lengths",
        "min_operators": 0,
        "max_operators": 0,
        "power": 1,
        "max_length": 4,
        "max_tries": 1,
        "independent_dimensions": True,
    }

    support_sampler_config = {
        "support_prior": {
            "name": "uniform",
            "kwargs": {"low": -1, "high": 1, "min_value": -1, "max_value": 1},
        },
        "n_support_prior": {
            "name": "uniform",
            "kwargs": {"low": 6, "high": 6, "min_value": 6, "max_value": 6},
        },
    }

    pool = SkeletonPool.from_dict(
        skeletons={("x1",)},
        simplipy_engine=engine,
        sample_strategy=sample_strategy,
        literal_prior={"name": "normal", "kwargs": {"loc": 0, "scale": 1}},
        variables=["x1"],
        support_sampler_config=support_sampler_config,
    )

    pool.skeletons = {("x1",)}
    return pool


def _build_sample() -> EvaluationSample:
    x_support = np.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_support = x_support.copy()
    x_validation = np.linspace(-0.5, 0.5, 2).reshape(-1, 1)
    y_validation = x_validation.copy()

    metadata = {
        "skeleton": ["x1"],
        "variables": ["x1"],
        "variable_names": ["x1"],
        "complexity": 1,
    }

    return EvaluationSample(
        x_support=x_support,
        y_support=y_support,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata=metadata,
    )


def test_skeleton_pool_adapter_identity(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=1,
        unique=True,
        ignore_holdouts=True,
        seed=0,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    adapter = SkeletonPoolAdapter(model)
    adapter.prepare()
    sample = _build_sample()

    result = adapter.evaluate_sample(sample)
    values = result.to_mapping()

    assert values["prediction_success"] is True
    np.testing.assert_allclose(values["y_pred"].squeeze(), sample.y_support.squeeze(), atol=1e-3)
    np.testing.assert_allclose(values["y_pred_val"].squeeze(), sample.y_validation.squeeze(), atol=1e-3)
    assert values["predicted_expression"]
    assert values["predicted_skeleton_prefix"] is not None


def test_brute_force_adapter_identity(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        max_expressions=16,
        max_length=2,
        include_constant_token=False,
        ignore_holdouts=True,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    adapter = BruteForceAdapter(model)
    adapter.prepare()
    sample = _build_sample()

    result = adapter.evaluate_sample(sample)
    values = result.to_mapping()

    assert values["prediction_success"] is True
    np.testing.assert_allclose(values["y_pred"].squeeze(), sample.y_support.squeeze(), atol=1e-3)
    np.testing.assert_allclose(values["y_pred_val"].squeeze(), sample.y_validation.squeeze(), atol=1e-3)
    assert values["predicted_expression"]
    assert values["predicted_skeleton_prefix"] is not None
