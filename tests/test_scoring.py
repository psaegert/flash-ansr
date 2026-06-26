"""Characterization tests for the canonical scoring primitives.

These pin the exact numeric behaviour the inference candidate-scoring path relied on BEFORE the
de-duplication (the values were taken from ``FlashANSR``'s original ``np.log10``-based
implementation), so a regression in :mod:`flash_ansr.scoring` is caught here without importing the
heavy product class. See ``REPO_SPLIT_PLAN.md`` §2.
"""
import math

import numpy as np
import pytest

from flash_ansr.scoring import (
    FLOAT64_EPS,
    compute_fvu,
    count_constants,
    is_constant_token,
    normalize_variance,
    score_from_fvu,
)


class TestNormalizeVariance:
    def test_passthrough_when_above_floor(self):
        assert normalize_variance(2.0) == 2.0

    def test_floors_zero(self):
        assert normalize_variance(0.0) == FLOAT64_EPS

    def test_floors_negative(self):
        assert normalize_variance(-1.0) == FLOAT64_EPS

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_non_finite_returns_floor(self, bad):
        assert normalize_variance(bad) == FLOAT64_EPS


class TestComputeFvu:
    def test_basic_ratio(self):
        assert compute_fvu(0.5, 10, 2.0) == pytest.approx(0.25)

    def test_single_sample_returns_raw_loss(self):
        # variance is undefined with <=1 sample, so the raw loss is returned unchanged
        assert compute_fvu(0.5, 1, 2.0) == 0.5
        assert compute_fvu(0.5, 0, 2.0) == 0.5

    def test_zero_variance_floored(self):
        assert compute_fvu(0.5, 10, 0.0) == pytest.approx(0.5 / FLOAT64_EPS)

    def test_perfect_fit_zero_loss(self):
        assert compute_fvu(0.0, 10, 2.0) == 0.0


class TestScoreFromFvu:
    def test_no_penalties_no_likelihood(self):
        # log10(0.01) == -2 exactly; no structural/likelihood terms
        assert score_from_fvu(0.01, 0, 0, None, 0.0, 0.0, 0.0) == pytest.approx(-2.0)

    def test_structural_penalties(self):
        # log10(0.01) + 0.1*5 + 0.2*2 = -2 + 0.5 + 0.4 = -1.1
        got = score_from_fvu(0.01, 5, 2, None, 0.1, 0.2, 0.0)
        assert got == pytest.approx(-1.1)

    def test_likelihood_term_added_when_finite(self):
        # likelihood term = likelihood_penalty * (-log_prob) = 0.5 * 3.0 = +1.5
        base = score_from_fvu(0.01, 0, 0, None, 0.0, 0.0, 0.5)
        with_lp = score_from_fvu(0.01, 0, 0, -3.0, 0.0, 0.0, 0.5)
        assert base == pytest.approx(-2.0)
        assert with_lp == pytest.approx(-0.5)

    @pytest.mark.parametrize("lp", [None, np.inf, -np.inf, np.nan])
    def test_non_finite_or_missing_log_prob_contributes_nothing(self, lp):
        assert score_from_fvu(0.01, 0, 0, lp, 0.0, 0.0, 0.5) == pytest.approx(-2.0)

    @pytest.mark.parametrize("bad_fvu", [0.0, -1.0, np.inf, -np.inf, np.nan])
    def test_non_positive_or_non_finite_fvu_uses_floor(self, bad_fvu):
        # falls back to log10(FLOAT64_EPS); must stay finite (no -inf perfect-fit collapse)
        expected = float(np.log10(FLOAT64_EPS))
        assert score_from_fvu(bad_fvu, 0, 0, None, 0.0, 0.0, 0.0) == pytest.approx(expected)
        assert math.isfinite(score_from_fvu(bad_fvu, 0, 0, None, 0.0, 0.0, 0.0))

    def test_tiny_positive_fvu_floored(self):
        # fvu below the floor is lifted to the floor
        assert score_from_fvu(FLOAT64_EPS / 10, 0, 0, None, 0.0, 0.0, 0.0) == pytest.approx(
            float(np.log10(FLOAT64_EPS))
        )

    def test_negative_constant_count_clamped(self):
        # max(int(constant_count), 0) guards against a negative count
        assert score_from_fvu(0.01, 0, -5, None, 0.0, 1.0, 0.0) == pytest.approx(-2.0)


class TestIsConstantToken:
    """Pins the exact constant-token recognition the (formerly 5-way triplicated) scoring helper
    relied on, now single-owned in flash_ansr.scoring. See REPO_SPLIT_PLAN.md §2."""

    @pytest.mark.parametrize(
        "token",
        ["<constant>", "C_0", "C_12", "0", "1", "(-1)", "np.pi", "np.e",
         'float("inf")', 'float("-inf")', 'float("nan")', "3.5", "-3", "1.0e3", "42"],
    )
    def test_recognised_as_constant(self, token):
        assert is_constant_token(token) is True

    @pytest.mark.parametrize("token", ["x0", "sin", "+", "C_", "C_x", "<num>"])
    def test_not_a_constant(self, token):
        assert is_constant_token(token) is False


class TestCountConstants:
    def test_none_returns_zero(self):
        assert count_constants(None) == 0

    def test_empty_returns_zero(self):
        assert count_constants([]) == 0

    def test_counts_mixed_expression(self):
        assert count_constants(["+", "<constant>", "*", "x0", "C_0"]) == 2

    def test_counts_no_constants(self):
        assert count_constants(["+", "x0", "x1"]) == 0

    def test_coerces_non_string_tokens(self):
        # the constant-counting path may receive non-str tokens; they are str()-ed first
        assert count_constants([0, 1, "x0"]) == 2
