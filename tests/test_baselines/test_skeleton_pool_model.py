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
