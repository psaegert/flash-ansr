"""Every trained circumstance needs a public entry point (design principle 1).

T16 trains predict_y in two placements and every decoder task in two conditioning modes; before
this, only one placement and one mode were reachable. These tests pin the boundary behaviour of
`conditioned=` and `predict_y(expression=...)`: the refusals, the placement, and the fact that the
knob is threaded rather than accepted-and-ignored.
"""
import inspect

import numpy as np
import pytest
import torch

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.preprocessing import CapabilityUnavailable
from flash_ansr.tasks import _conditioning, predict_complexity, predict_constants, predict_y


class _Model:
    def __init__(self, optional_condition: bool) -> None:
        self.optional_condition = optional_condition
        self.device = torch.device("cpu")


class _Estimator:
    n_variables = 3

    def __init__(self, optional_condition: bool = True) -> None:
        self.flash_ansr_model = _Model(optional_condition)


class TestConditioningHelper:
    def test_conditioned_needs_data(self) -> None:
        with pytest.raises(ValueError, match="needs X and y"):
            _conditioning(_Estimator(), None, None, conditioned=True, n_rows=4, verb="predict_y")

    def test_conditioned_builds_no_mask(self) -> None:
        X, y = np.zeros((6, 3), dtype=np.float32), np.zeros(6, dtype=np.float32)
        _data, _mask, condition_mask = _conditioning(
            _Estimator(), X, y, conditioned=True, n_rows=4, verb="predict_y")
        assert condition_mask is None, "a conditioned call must not route through null_memory"

    def test_unconditioned_mask_is_all_false_one_row_per_draw(self) -> None:
        _data, _mask, condition_mask = _conditioning(
            _Estimator(), None, None, conditioned=False, n_rows=7, verb="predict_y")
        assert condition_mask is not None
        assert condition_mask.shape == (7,)
        assert not bool(condition_mask.any()), "unconditioned means every row unconditioned"

    def test_unconditioned_synthesizes_a_support_set_of_the_right_width(self) -> None:
        data, _mask, _cm = _conditioning(
            _Estimator(), None, None, conditioned=False, n_rows=2, verb="predict_y")
        # (batch, points, n_variables + 1) -- the y column rides alongside x.
        assert data.shape[-1] == _Estimator.n_variables + 1

    def test_model_without_null_memory_refuses(self) -> None:
        with pytest.raises(CapabilityUnavailable, match="optional_condition"):
            _conditioning(_Estimator(optional_condition=False), None, None,
                          conditioned=False, n_rows=4, verb="predict_complexity")


class TestSurfaceIsThreaded:
    """A parameter that is documented as having an effect must have one (principle 6)."""

    @pytest.mark.parametrize("verb", [predict_y, predict_constants, predict_complexity])
    def test_every_sampled_verb_takes_conditioned(self, verb) -> None:
        parameter = inspect.signature(verb).parameters.get("conditioned")
        assert parameter is not None, f"{verb.__name__} cannot reach the unconditioned mode"
        assert parameter.default is True, "conditioned must default to the ordinary path"

    @pytest.mark.parametrize("method", ["predict_y", "predict_constants", "predict_complexity",
                                        "fit", "infer"])
    def test_the_estimator_exposes_it_too(self, method) -> None:
        assert "conditioned" in inspect.signature(getattr(FlashANSR, method)).parameters

    def test_predict_y_takes_an_expression(self) -> None:
        # The trained SUFFIX placement: the block sits after </expression>.
        assert "expression" in inspect.signature(predict_y).parameters
        assert "expression" in inspect.signature(FlashANSR.predict_y).parameters

    def test_estimator_forwards_conditioned_rather_than_dropping_it(self) -> None:
        source = inspect.getsource(FlashANSR.predict_y)
        assert "conditioned=conditioned" in source
        assert "expression=expression" in source
