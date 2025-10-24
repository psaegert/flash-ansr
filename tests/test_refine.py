import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr.refine import Refiner


def test_transform_gracefully_handles_missing_constants() -> None:
    engine = SimpliPyEngine.load('dev_7-3', install=True)
    refiner = Refiner(simplipy_engine=engine, n_variables=1)
    refiner.constants_symbols = ['C_0', 'C_1']
    refiner._all_constants_values = [(np.array([0.5, -0.25]), np.eye(2), 0.0)]

    expression = ['+', '<constant>', '*', '<constant>', '<constant>']

    transformed = refiner.transform(expression, nth_best_constants=0, return_prefix=True, precision=2)

    assert len(transformed) == len(expression)
    assert transformed[:3] == ['+', '0.5', '*']
    assert transformed[3] == '-0.25'
    assert transformed[4] == '<constant>'


def test_fit_discards_mismatched_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimpliPyEngine.load('dev_7-3', install=True)
    refiner = Refiner(simplipy_engine=engine, n_variables=1)

    expression = ['+', '<constant>', '<constant>']
    X = np.ones((4, 1), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)

    def fake_fit(self: Refiner, *args, **kwargs):
        return np.array([1.0]), np.eye(1)

    monkeypatch.setattr(Refiner, "_fit", fake_fit)

    refiner.fit(expression=expression, X=X, y=y)

    assert refiner._all_constants_values == []
    assert refiner.valid_fit is False
