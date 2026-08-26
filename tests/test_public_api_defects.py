"""Seven public-API defects from the inference audit, each pinned by its own repro.

The common shape: a surface that says one thing and does another, failing quietly enough that a
caller reads a plausible answer instead of an error.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from simplipy import SimpliPyEngine

from flash_ansr.utils.skeleton import (
    record_unencodable_drop,
    reset_unencodable_drops,
    unencodable_drops,
)
from flash_ansr.utils.tensor_ops import pad_input_set


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4", install=True)


class TestPadInputSetAcceptsWhatTheSignaturePromises:
    def test_dataframe_is_converted_not_passed_through(self) -> None:
        # It used to fall through both branches and return the FRAME, which then reached
        # Refiner.predict as `expression_lambda(*X.T)` -- unpacking 64 row labels as positional
        # args and surfacing as "takes 19 positional arguments but 66 were given" from infer().
        out = pad_input_set(pd.DataFrame(np.zeros((8, 2)), columns=['a', 'b']), 4)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8, 4)

    def test_ndarray_and_tensor_still_work(self) -> None:
        assert pad_input_set(np.zeros((4, 1)), 3).shape == (4, 3)
        assert tuple(pad_input_set(torch.zeros(4, 1), 3).shape) == (4, 3)

    def test_no_padding_needed_is_a_passthrough(self) -> None:
        arr = np.ones((4, 3))
        assert pad_input_set(arr, 3).shape == (4, 3)

    def test_unsupported_type_raises_instead_of_passing_through(self) -> None:
        with pytest.raises(TypeError, match="ndarray, a Tensor or a DataFrame"):
            pad_input_set([[1.0, 2.0]], 3)


class TestUnencodableDropIsCounted:
    def test_counter_starts_at_zero_after_reset(self) -> None:
        reset_unencodable_drops()
        assert unencodable_drops() == 0

    def test_drops_accumulate(self) -> None:
        reset_unencodable_drops()
        record_unencodable_drop()
        record_unencodable_drop()
        assert unencodable_drops() == 2
        reset_unencodable_drops()
        assert unencodable_drops() == 0


class TestRefinerTransformSlotAlignment:
    """A numeric LITERAL is a fitted degree of freedom; the substitution must know that."""

    def test_literal_site_receives_its_own_fitted_value(self, simplipy_engine) -> None:  # type: ignore[no-untyped-def]
        from flash_ansr.refine import Refiner
        refiner = Refiner(simplipy_engine=simplipy_engine, n_variables=1)
        rng = np.random.default_rng(0)
        X = rng.uniform(1.0, 4.0, size=(64, 1))
        y = (7.0 * X[:, 0] + 100.0).reshape(-1, 1)
        expression = ['+', '*', '2', 'x1', '<constant>']

        refiner.fit(expression=expression, X=X, y=y, n_restarts=8, p0_noise='normal',
                    p0_noise_kwargs={'loc': 0.0, 'scale': 5.0}, converge_error='ignore')

        # identify_constants(convert_numbers_to_constant=True) promoted '2' to a fitted slot, so the
        # fit has TWO constants. Keyed on the token test alone, transform skipped the '2', wrote
        # C_0's value into C_1's slot, and dropped the rest -- publishing 2*x1 + 7.0 for a function
        # that had fitted exactly as 7*x1 + 100.
        out = refiner.transform(expression, return_prefix=True, precision=6)
        assert out == ['+', '*', '7.0', 'x1', '100.0']

    def test_plain_placeholder_expression_is_unaffected(self, simplipy_engine) -> None:  # type: ignore[no-untyped-def]
        from flash_ansr.refine import Refiner
        refiner = Refiner(simplipy_engine=simplipy_engine, n_variables=1)
        rng = np.random.default_rng(1)
        X = rng.uniform(1.0, 4.0, size=(64, 1))
        y = (3.0 * X[:, 0]).reshape(-1, 1)
        expression = ['*', '<constant>', 'x1']
        refiner.fit(expression=expression, X=X, y=y, n_restarts=8, p0_noise='normal',
                    p0_noise_kwargs={'loc': 0.0, 'scale': 5.0}, converge_error='ignore')
        out = refiner.transform(expression, return_prefix=True, precision=4)
        assert out[0] == '*' and out[2] == 'x1'
        assert float(out[1]) == pytest.approx(3.0, rel=1e-6)
