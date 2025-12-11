import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import FlashANSR, SoftmaxSamplingConfig, install_model, get_path
from flash_ansr.expressions import SkeletonPool
from flash_ansr.baselines.skeleton_pool_model import SkeletonPoolModel
from flash_ansr.results import (
    RESULTS_FORMAT_VERSION,
    deserialize_results_payload,
    load_results_payload,
    save_results_payload,
    serialize_results_payload,
)


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    # Use the small dev config that tests already rely on.
    return SimpliPyEngine.load("dev_7-3", install=True)


def _make_result_entry(expr: list[str]) -> dict:
    fits = [(np.array([3.0], dtype=float), None, 0.0)]
    return {
        "log_prob": -1.0,
        "fvu": 0.01,
        "score": 0.02,
        "expression": expr,
        "complexity": len(expr),
        "requested_complexity": None,
        "raw_beam": expr,
        "beam": expr,
        "raw_beam_decoded": " ".join(expr),
        "function": None,
        "refiner": None,
        "fits": fits,
        "prompt_metadata": None,
    }


def test_serialize_and_deserialize_rebuilds_refiner(tmp_path, simplipy_engine: SimpliPyEngine) -> None:
    expr = ["+", "x1", "<constant>"]
    results = [_make_result_entry(expr)]

    metadata = {
        "format_version": RESULTS_FORMAT_VERSION,
        "parsimony": 0.1,
        "n_variables": 1,
        "input_dim": 1,
        "variable_mapping": {"x1": "x"},
    }

    payload = serialize_results_payload(results, metadata=metadata)

    assert "refiner" not in payload["results"][0]
    assert "function" not in payload["results"][0]

    path = tmp_path / "results.pkl"
    save_results_payload(payload, path)

    loaded = load_results_payload(path)
    restored = deserialize_results_payload(
        loaded,
        simplipy_engine=simplipy_engine,
        n_variables=1,
        input_dim=1,
        rebuild_refiners=True,
    )

    assert len(restored) == 1
    entry = restored[0]
    assert entry["refiner"] is not None
    assert entry["function"] is not None
    np.testing.assert_allclose(entry["fits"][0][0], np.array([3.0]))

    X = np.array([[1.0], [2.0]], dtype=float)
    preds = entry["refiner"].predict(X)
    np.testing.assert_allclose(preds.flatten(), np.array([4.0, 5.0]), rtol=1e-6)


def test_deserialize_without_rebuild_preserves_fits_only(tmp_path, simplipy_engine: SimpliPyEngine) -> None:
    expr = ["+", "x1", "<constant>"]
    results = [_make_result_entry(expr)]

    payload = serialize_results_payload(results, metadata={"parsimony": 0.1})

    path = tmp_path / "results.pkl"
    save_results_payload(payload, path)

    loaded = load_results_payload(path)
    restored = deserialize_results_payload(
        loaded,
        simplipy_engine=simplipy_engine,
        n_variables=1,
        input_dim=1,
        rebuild_refiners=False,
    )

    entry = restored[0]
    assert entry["refiner"] is None
    assert entry["function"] is None
    assert entry["fits"] == [(np.array([3.0]), None, 0.0)]


def test_skeleton_pool_model_save_load_roundtrip(tmp_path, simplipy_engine: SimpliPyEngine) -> None:
    # Build a minimal deterministic pool with one linear skeleton a*x + b.
    skeletons = {('+', '*', '<constant>', 'x1', '<constant>')}
    sample_strategy = {"max_operators": 2, "independent_dimensions": True}
    literal_prior = {"name": "normal", "kwargs": {"loc": 0, "scale": 1}}
    variables = ["x1"]
    support_sampler_config = {
        "support_prior": {"name": "uniform", "kwargs": {"min_value": -2, "max_value": 2}},
        "n_support_prior": {"name": "uniform", "kwargs": {"low": 4, "high": 4, "min_value": 4, "max_value": 4}},
    }

    pool = SkeletonPool.from_dict(
        skeletons=skeletons,
        simplipy_engine=simplipy_engine,
        sample_strategy=sample_strategy,
        literal_prior=literal_prior,
        variables=variables,
        support_sampler_config=support_sampler_config,
        operator_weights={"+": 1.0, "*": 1.0},
    )

    model = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=1,
        seed=0,
        n_restarts=4,
        parsimony=0.0,
    )

    x = np.linspace(-2.0, 2.0, 20, dtype=float).reshape(-1, 1)
    y = 2.0 * x + 1.0

    model.fit(x, y)

    val = np.array([[-1.5], [0.0], [1.5]], dtype=float)
    preds_before = model.predict(val)

    path = tmp_path / "roundtrip.pkl"
    model.save_results(path)

    reloaded = SkeletonPoolModel(
        simplipy_engine=simplipy_engine,
        skeleton_pool=pool,
        samples=0,
    )
    reloaded.load_results(path)

    preds_after = reloaded.predict(val)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6, atol=1e-8)


def test_flash_ansr_save_load_roundtrip_softmax_sampling(tmp_path, simplipy_engine: SimpliPyEngine) -> None:
    model_repo = "psaegert/flash-ansr-v23.0-3M"
    install_model(model_repo)
    model_dir = get_path("models", model_repo)

    generation_config = SoftmaxSamplingConfig(
        choices=16,
        top_k=8,
        top_p=0.95,
        max_len=24,
        batch_size=32,
        temperature=0.8,
        simplify=True,
        unique=True,
    )

    regressor = FlashANSR.load(
        directory=model_dir,
        generation_config=generation_config,
        n_restarts=4,
        parsimony=0.0,
    )

    x = np.linspace(-2.0, 2.0, 24, dtype=float).reshape(-1, 1)
    y = 2.0 * x + 1.0

    regressor.fit(x, y)

    val = np.array([[-1.5], [0.0], [1.5]], dtype=float)
    preds_before = regressor.predict(val)

    save_path = tmp_path / "flash_roundtrip.pkl"
    regressor.save_results(save_path)

    reloaded = FlashANSR.load(
        directory=model_dir,
        generation_config=generation_config,
        n_restarts=4,
        parsimony=0.0,
    )
    reloaded.load_results(save_path)

    preds_after = reloaded.predict(val)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6, atol=1e-8)
