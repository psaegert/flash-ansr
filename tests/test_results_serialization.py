import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import FlashANSR, get_path
from flash_ansr.results import (
    RESULTS_FORMAT_VERSION,
    deserialize_results_payload,
    load_results_payload,
    save_results_payload,
    serialize_results_payload,
)


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    # The generation-2 engine released simplipy (>= 0.12) can actually load.
    return SimpliPyEngine.load("acj-4-3", install=True)


def _make_result_entry(expr: list[str]) -> dict:
    fits = [(np.array([3.0], dtype=float), None, 0.0)]
    return {
        "log_prob": -1.0,
        "fvu": 0.01,
        "score": 0.02,
        "expression": expr,
        "constant_count": 1,
        "complexity": len(expr),
        "requested_complexity": None,
        "raw_beam": expr,
        "beam": expr,
        "raw_beam_decoded": " ".join(expr),
        "function": None,
        "refiner": None,
        "fits": fits,
    }


def test_serialize_and_deserialize_rebuilds_refiner(tmp_path, simplipy_engine: SimpliPyEngine) -> None:
    expr = ["+", "x1", "<constant>"]
    results = [_make_result_entry(expr)]

    metadata = {
        "format_version": RESULTS_FORMAT_VERSION,
        "ranking": {"mode": "weighted", "weights": {"n_nodes": 0.1}},
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

    payload = serialize_results_payload(results, metadata={
        "ranking": {"mode": "weighted", "weights": {"n_nodes": 0.1}},
    })

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


def test_fit_result_roundtrip_on_the_test_model(tmp_path, simplipy_engine: SimpliPyEngine) -> None:
    """A FitResult is plain data: it saves, loads with an engine, predicts the same numbers, keeps its
    ranking, and re-ranks offline exactly as the live run ordered it."""
    import torch
    from test_prior_sampling import _catalog_config
    from flash_ansr import FitResult
    from flash_ansr.model.flash_ansr_model import FlashANSRModel
    from flash_ansr.model.tokenizer import Tokenizer
    from flash_ansr.utils.generation import create_generation_config

    tokenizer = Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))
    torch.manual_seed(0)
    model = FlashANSRModel.from_config(get_path("configs", "test", "model.yaml"))
    nsr = FlashANSR(simplipy_engine=simplipy_engine, flash_ansr_model=model, tokenizer=tokenizer,
                    generation_config=create_generation_config(method="prior_sampling", catalog=_catalog_config(), draws=48),
                    refine={"n_restarts": 2}, compute={"workers": 0}, model_directory=None)
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(64, 2))
    y = (2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0).reshape(-1, 1)
    result = nsr.fit(X, y, seed=0)
    assert result.best is not None and nsr.result_ is result

    val = rng.uniform(-3, 3, size=(5, 2))
    before = result.predict(val)
    path = tmp_path / "result.pkl"
    result.save(path)
    loaded = FitResult.load(path, engine=simplipy_engine)
    assert [c.raw_beam for c in loaded.candidates] == [c.raw_beam for c in result.candidates]
    assert loaded.ranking == result.ranking and loaded.draws == result.draws
    np.testing.assert_allclose(loaded.predict(val), before, rtol=1e-12, atol=0.0)
    assert loaded.get_expression() == result.get_expression()
    assert loaded.get_expression(return_prefix=True) == result.get_expression(return_prefix=True)
    assert loaded.get_expression(precision=2) == result.get_expression(precision=2)

    # loading without an engine keeps the data but refuses to evaluate
    bare = FitResult.load(path)
    assert bare.best is not None
    with pytest.raises(ValueError, match="engine"):
        bare.predict(val)

    # the offline re-rank under the SAME ranking reproduces the live order exactly
    again = loaded.rerank()
    assert [c.raw_beam for c in again.candidates] == [c.raw_beam for c in result.candidates]
    assert [c.score for c in again.candidates] == [c.score for c in result.candidates]
    # under another ranking the set is the same, the order may differ, and the ledger follows
    other = loaded.rerank("weighted", weights={"n_nodes": 0.05})
    assert other.ranking.mode == "weighted" and loaded.ranking.mode == "mdl"
    assert sorted(map(tuple, (c.raw_beam for c in other.candidates))) == sorted(map(tuple, (c.raw_beam for c in loaded.candidates)))
    assert [c.rank for c in other.candidates] == list(range(len(other.candidates)))
    fitted_rows = [i for i, st in enumerate(other.ledger.fit_status) if st == 0]
    assert sorted(other.ledger.rank[i] for i in fitted_rows) == list(range(len(other.candidates)))

    # a foreign format version is refused
    import pickle
    payload = pickle.load(open(path, "rb"))
    payload["format_version"] = 1
    stale = tmp_path / "stale.pkl"
    pickle.dump(payload, open(stale, "wb"))
    with pytest.raises(ValueError, match="format"):
        FitResult.load(stale)
