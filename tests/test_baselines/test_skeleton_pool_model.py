import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import SkeletonPool
from flash_ansr.baselines.skeleton_pool_model import SkeletonPoolModel


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("dev_7-3", install=True)


def _build_toy_pool(engine: SimpliPyEngine) -> SkeletonPool:
    # Deterministic pool with a single variable skeleton to keep the test fast and stable.
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
            "kwargs": {"low": 4, "high": 4, "min_value": 4, "max_value": 4},
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

    # Pre-populate the cached skeletons so sampling is deterministic.
    pool.skeletons = {("x1",)}
    return pool


def _build_multivar_pool(engine: SimpliPyEngine) -> SkeletonPool:
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
            "kwargs": {"low": 4, "high": 4, "min_value": 4, "max_value": 4},
        },
    }

    pool = SkeletonPool.from_dict(
        skeletons={("x1",)},
        simplipy_engine=engine,
        sample_strategy=sample_strategy,
        literal_prior={"name": "normal", "kwargs": {"loc": 0, "scale": 1}},
        variables=["x1", "x2", "x3"],
        support_sampler_config=support_sampler_config,
    )

    pool.skeletons = {("x1",)}
    return pool


def test_fit_and_predict_identity(simplipy_engine: SimpliPyEngine) -> None:
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

    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = X.copy()

    model.fit(X, y)

    assert len(model._results) == 1
    preds = model.predict(X)
    np.testing.assert_allclose(preds.squeeze(), y.squeeze(), atol=1e-3)
    # Ensure we can recover the expression string for user display.
    expr = model.get_expression()
    assert expr in (["x1"], "x1")


def test_truncates_extra_columns(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=1,
        unique=True,
        ignore_holdouts=True,
        seed=1,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    x_primary = np.linspace(-1.0, 1.0, 6)
    noise = np.random.RandomState(1).normal(scale=0.25, size=x_primary.shape)
    X = np.stack([x_primary, noise], axis=1)
    y = x_primary.reshape(-1, 1)

    model.fit(X, y)

    preds = model.predict(X)
    np.testing.assert_allclose(preds.squeeze(), y.squeeze(), atol=1e-3)


def test_pads_missing_columns(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_multivar_pool(simplipy_engine)
    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=1,
        unique=True,
        ignore_holdouts=True,
        seed=2,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    X = np.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y = X.copy()

    model.fit(X, y)

    preds = model.predict(X)
    np.testing.assert_allclose(preds.squeeze(), y.squeeze(), atol=1e-3)


def test_zero_samples_returns_empty(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=0,
        unique=True,
        ignore_holdouts=True,
        seed=0,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    X = np.zeros((3, 1))
    y = np.zeros((3, 1))

    model.fit(X, y)

    assert model._results == []
    with pytest.raises(ValueError):
        model.predict(X)


def test_unique_sampling_uses_cached_pool(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    pool.skeletons = {("x1",), ("<constant>",)}

    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=2,
        unique=True,
        ignore_holdouts=True,
        seed=42,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    X = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    y = X.copy()

    model.fit(X, y)

    used = {tuple(result["expression"]) for result in model._results}
    assert used == {("x1",), ("<constant>",)}
    with pytest.raises(IndexError):
        model.predict(X, nth_best=5)
