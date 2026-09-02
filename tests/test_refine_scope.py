"""Predict-vs-refine doctrine (owner ruling 2026-09-02): which literals the refiner may move.

The model PREDICTS the typed literals (pow exponents, rootn indices) and the refiner fits the
rest. ``refinement_slots`` is the one slot definition the refiner and the p0 seeding share.
"""
import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr.refine import Refiner, refinement_slots, literal_value, DEFAULT_REFINE_SCOPE


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-4-3', install=True)


def test_default_scope_is_fittable() -> None:
    assert DEFAULT_REFINE_SCOPE == 'fittable'


def test_slots_placeholders_only_the_slots(engine: SimpliPyEngine) -> None:
    expr = ['*', '<constant>', 'pow', 'x1', '2']
    assert refinement_slots(expr, engine, 'placeholders') == [1]
    expr = ['*', '2.5', 'pow', 'x1', '2']
    assert refinement_slots(expr, engine, 'placeholders') == []


def test_slots_fittable_frees_coefficients_keeps_typed(engine: SimpliPyEngine) -> None:
    # coefficient 2.5 is fittable; the exponent 2 controls the domain and stays.
    expr = ['*', '2.5', 'pow', 'x1', '2']
    assert refinement_slots(expr, engine, 'fittable') == [1]
    # a placeholder AND a spelled coefficient: both are slots, in order of appearance
    expr = ['+', '<constant>', '*', '3', 'rootn', 'x1', '3']
    assert refinement_slots(expr, engine, 'fittable') == [1, 3]


def test_slots_all_frees_typed_literals_too(engine: SimpliPyEngine) -> None:
    expr = ['*', '2.5', 'pow', 'x1', '2']
    assert refinement_slots(expr, engine, 'all') == [1, 4]
    # ... including the spellings the old digit-only rule skipped
    expr = ['pow', 'x1', '-2']
    assert refinement_slots(expr, engine, 'all') == [2]
    assert refinement_slots(expr, engine, 'fittable') == []


def test_slots_reject_unknown_scope(engine: SimpliPyEngine) -> None:
    with pytest.raises(ValueError, match="refine_scope"):
        refinement_slots(['x1'], engine, 'typed')  # type: ignore[arg-type]


def test_literal_value_parses_rationals_and_specials() -> None:
    assert literal_value('3/2') == 1.5
    assert literal_value('-7/30') == pytest.approx(-7 / 30)
    assert literal_value('2.5') == 2.5
    assert literal_value('np.pi') == pytest.approx(np.pi)


def _data(fn, n: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = fn(X[:, 0]).reshape(-1, 1)
    return X, y


def test_fit_fittable_keeps_the_exponent_verbatim(engine: SimpliPyEngine) -> None:
    # y = 3 x^2 spelled as <constant> * pow(x, 2): ONE fitted constant, exponent 2 compiled in.
    X, y = _data(lambda x: 3.0 * x ** 2)
    refiner = Refiner(simplipy_engine=engine, n_variables=1).fit(
        expression=['*', '<constant>', 'pow', 'x1', '2'], X=X, y=y, n_restarts=4, refine_scope='fittable')
    assert refiner.constants_symbols == ['C_0']
    assert refiner.slot_indices == [1]
    assert refiner.valid_fit
    assert refiner.all_constants_values[0][0][0] == pytest.approx(3.0, rel=1e-6)
    assert refiner.loss == pytest.approx(0.0, abs=1e-12)


def test_fit_all_frees_the_exponent(engine: SimpliPyEngine) -> None:
    X, y = _data(lambda x: 3.0 * x ** 2)
    refiner = Refiner(simplipy_engine=engine, n_variables=1).fit(
        expression=['*', '<constant>', 'pow', 'x1', '2'], X=X, y=y, n_restarts=4, refine_scope='all')
    assert refiner.constants_symbols == ['C_0', 'C_1']
    assert refiner.slot_indices == [1, 4]


def test_fit_placeholders_compiles_spelled_literals_in(engine: SimpliPyEngine) -> None:
    # y = 2.5 x^2 with the coefficient SPELLED: under 'placeholders' nothing is fitted and the
    # verbatim expression already explains the data.
    X, y = _data(lambda x: 2.5 * x ** 2)
    refiner = Refiner(simplipy_engine=engine, n_variables=1).fit(
        expression=['*', '2.5', 'pow', 'x1', '2'], X=X, y=y, n_restarts=2, refine_scope='placeholders')
    assert refiner.constants_symbols == []
    y_pred = refiner.expression_lambda(X[:, 0])
    assert np.allclose(y_pred, y[:, 0])


def test_fit_fittable_seeds_a_spelled_coefficient_from_p0(engine: SimpliPyEngine) -> None:
    # A spelled but fittable coefficient is a slot: p0 (one value per slot) seeds it and the
    # optimizer moves it to the data. Typed literal count stays zero.
    X, y = _data(lambda x: 3.0 * x ** 2)
    refiner = Refiner(simplipy_engine=engine, n_variables=1).fit(
        expression=['*', '2.5', 'pow', 'x1', '2'], X=X, y=y, n_restarts=1, p0=np.array([2.5]),
        p0_noise=None, refine_scope='fittable')
    assert refiner.constants_symbols == ['C_0']
    assert refiner.all_constants_values[0][0][0] == pytest.approx(3.0, rel=1e-6)


def test_from_serialized_round_trips_the_scope(engine: SimpliPyEngine) -> None:
    fits = [(np.array([3.0]), np.eye(1), 0.0)]
    r = Refiner.from_serialized(engine, 1, ['*', '<constant>', 'pow', 'x1', '2'], 1, fits, refine_scope='fittable')
    assert r.constants_symbols == ['C_0'] and r.valid_fit
    r_all = Refiner.from_serialized(engine, 1, ['*', '<constant>', 'pow', 'x1', '2'], 1, fits, refine_scope='all')
    assert r_all.constants_symbols == ['C_0', 'C_1']
    assert r_all.valid_fit is False   # one value for two slots is discarded, not misassigned
