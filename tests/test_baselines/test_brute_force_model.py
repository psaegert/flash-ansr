import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import SkeletonPool
from flash_ansr.baselines import BruteForceModel


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

    pool.skeletons = {("x1",)}
    return pool


def test_fit_and_predict_identity(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        max_expressions=10,
        max_length=2,
        include_constant_token=False,
        ignore_holdouts=True,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = X.copy()

    model.fit(X, y)

    assert len(model._results) >= 1
    preds = model.predict(X)
    np.testing.assert_allclose(preds.squeeze(), y.squeeze(), atol=1e-3)
    expr = model.get_expression()
    assert expr in (["x1"], "x1")


def test_respects_limits(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        max_expressions=3,
        max_length=1,
        include_constant_token=False,
        ignore_holdouts=True,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    X = np.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    y = X.copy()

    model.fit(X, y)

    assert len(model._results) <= 3
    assert all(len(result["expression"]) <= 1 for result in model._results)


def test_truncates_extra_columns(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        max_expressions=8,
        max_length=2,
        include_constant_token=False,
        ignore_holdouts=True,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    x_primary = np.linspace(-1.0, 1.0, 6)
    noise = np.random.RandomState(0).normal(scale=0.5, size=x_primary.shape)
    X = np.stack([x_primary, noise], axis=1)
    y = x_primary.reshape(-1, 1)

    model.fit(X, y)

    preds = model.predict(X)
    np.testing.assert_allclose(preds.squeeze(), y.squeeze(), atol=1e-3)


def test_constant_expression_prediction(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        max_expressions=5,
        max_length=1,
        include_constant_token=True,
        ignore_holdouts=True,
        n_restarts=2,
        refiner_p0_noise=None,
    )

    X = np.linspace(-2.0, 2.0, 5).reshape(-1, 1)
    y = np.full_like(X, fill_value=2.0)

    model.fit(X, y)

    preds = model.predict(X)
    np.testing.assert_allclose(preds.squeeze(), y.squeeze(), atol=1e-2)
    _ = model.get_expression()


def test_predict_requires_fit(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        max_expressions=2,
        max_length=1,
        include_constant_token=False,
        ignore_holdouts=True,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    X = np.zeros((3, 1))

    with pytest.raises(ValueError):
        model.predict(X)

    model.fit(X, X)
    with pytest.raises(IndexError):
        model.predict(X, nth_best=10)
