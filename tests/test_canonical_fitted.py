"""The fitted candidate is emitted in canonical form: the expression the certified price describes.
Generation simplifies the skeleton, where two free constants cannot cancel; once fitted they can."""
import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr.flash_ansr import _refine_candidate_worker, canonicalize_fitted, price_realized
from flash_ansr.refine import Refiner


@pytest.fixture(scope="module")
def engine():
    return SimpliPyEngine.load("acj-4-3", install=True)


def _fit(engine, expression, X, y, p0):
    r = Refiner(simplipy_engine=engine, n_variables=X.shape[1])
    r.fit(expression, X, y, p0=np.asarray(p0, dtype=float), p0_noise=None, p0_noise_kwargs=None, n_restarts=1,
          method="curve_fit_lm", converge_error="ignore", refine_scope="fittable")
    assert r.valid_fit
    return r


def test_a_factor_that_cancels_once_fitted_is_dropped(engine):
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (64, 1))
    y = np.exp(-X[:, 0] ** 2)
    expression = "/ * exp neg pow x1 <constant> pow tanh x1 <constant> pow tanh x1 <constant>".split()
    fitted = _fit(engine, expression, X, y, [2.0, 2.0, 2.0])
    carried, canonical, changed, same_slots = canonicalize_fitted(engine, fitted, expression, X, n_variables=1)
    assert changed and not same_slots
    assert canonical == ["exp", "neg", "pow", "x1", "2"]                     # the fitted exponent folded to the typed literal 2
    assert carried.valid_fit and carried.slot_indices == []
    np.testing.assert_allclose(carried.predict(X).reshape(-1), fitted.predict(X).reshape(-1), rtol=1e-12)
    assert price_realized(engine, carried, canonical) == price_realized(engine, fitted, expression)   # the class price, unchanged


def test_constants_fold_into_one_slot(engine):
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, (64, 1))
    y = 6.0 * X[:, 0]
    expression = ["*", "<constant>", "*", "<constant>", "x1"]
    fitted = _fit(engine, expression, X, y, [2.0, 3.0])
    carried, canonical, changed, same_slots = canonicalize_fitted(engine, fitted, expression, X, n_variables=1)
    assert changed and not same_slots
    assert canonical == ["*", "<constant>", "x1"] and carried.slot_indices == [1]
    np.testing.assert_allclose(carried.all_constants_values[0][0], [6.0], rtol=1e-9)
    np.testing.assert_allclose(carried.predict(X).reshape(-1), y, rtol=1e-9)


def test_a_respelling_is_not_a_collapse(engine):
    """The canon writes an exact 1.5 as 3/2 (`/ * 3 sin x1 2`): longer, a different slot structure, the
    ladder's business -- the fitted spelling stays."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 3, (64, 1))
    y = 1.5 * np.sin(X[:, 0])
    expression = ["*", "<constant>", "sin", "x1"]
    fitted = _fit(engine, expression, X, y, [1.5])
    assert list(engine.simplify(["*", "1.5", "sin", "x1"])) == ["/", "*", "3", "sin", "x1", "2"]
    carried, canonical, changed, same_slots = canonicalize_fitted(engine, fitted, expression, X, n_variables=1)
    assert not changed and same_slots and carried is fitted and canonical == expression


def test_an_equal_length_respelling_is_left_to_the_ladder(engine):
    rng = np.random.default_rng(4)
    X = rng.uniform(0.5, 3, (64, 1))
    y = np.sqrt(X[:, 0])
    expression = ["pow", "x1", "<constant>"]
    fitted = _fit(engine, expression, X, y, [0.5])
    _, canonical, changed, _ = canonicalize_fitted(engine, fitted, expression, X, n_variables=1)
    assert not changed and canonical == expression                          # `rootn x1 2` is not shorter than `pow x1 0.5`


def test_the_worker_emits_the_canonical_member(engine):
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, (64, 1))
    y = np.exp(-X[:, 0] ** 2)
    payload = {
        "simplipy_engine": engine,
        "expression": "/ * exp neg pow x1 <constant> pow tanh x1 <constant> pow tanh x1 <constant>".split(),
        "n_variables": 1, "X": X, "y": y.reshape(-1, 1), "n_restarts": 1, "method": "curve_fit_lm", "p0": [2.0, 2.0, 2.0],
        "p0_noise": "normal", "p0_noise_kwargs": {"loc": 0, "scale": 1}, "converge_error": "ignore", "numpy_errors": "ignore",
        "y_variance": float(np.var(y)), "ranking_weights": {"mdl": 1e-2}, "log_prob": -1.0, "constant_count": 0,
        "complexity": None, "metadata_snapshot": None, "raw_beam": [1], "beam": [1], "raw_beam_decoded": "",
        "refine_scope": "fittable", "constant_ladder": None, "seed": 0,
    }
    result, warning = _refine_candidate_worker(payload)
    assert warning is None and result is not None
    assert result["expression"] == ["exp", "neg", "pow", "x1", "2"]
    assert result["expression_as_fitted"] == payload_expression_as_fitted()
    assert result["constants_emitted"] is None                                # the slots changed: the model's p0 no longer aligns
    assert result["fvu"] < 1e-20 and result["mdl"] == price_realized(engine, Refiner.from_serialized(
        simplipy_engine=engine, n_variables=1, expression=result["expression"], n_inputs=1, fits=[(np.array([]), None, 0.0)],
        refine_scope="placeholders"), result["expression"])


def payload_expression_as_fitted():
    return "/ * exp neg pow x1 <constant> pow tanh x1 <constant> pow tanh x1 <constant>".split()
