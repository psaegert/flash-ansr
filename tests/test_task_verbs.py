"""The auxiliary task verbs: capabilities T16 was trained on that nothing could call.

Unit-level coverage of the boundary behaviour (alias resolution, refusals, batching); the decode
loops themselves are exercised against a real checkpoint in the capability scripts.
"""
import numpy as np
import pytest
import torch

from flash_ansr.preprocessing import CapabilityUnavailable
from flash_ansr.tasks import (
    DEFAULT_SAMPLES,
    NIBBLES_PER_SPAN,
    ComplexityDistribution,
    ValueDistribution,
    _encoder_batch,
    _normalize_expression,
    _require,
)


class _Tok:
    def __init__(self, names): self._m = {n: i for i, n in enumerate(names)}
    def __contains__(self, k): return k in self._m
    def __getitem__(self, k): return self._m[k]


class _Engine:
    @staticmethod
    def parse_expression(text):
        return text.split()


class _Estimator:
    simplipy_engine = _Engine()


class TestRequire:
    def test_missing_token_refuses_before_compute(self) -> None:
        with pytest.raises(CapabilityUnavailable, match="predict_constants"):
            _require(_Tok(["<bos>"]), "<predict_constants>", "predict_constants")

    def test_present_token_resolves_to_its_id(self) -> None:
        assert _require(_Tok(["<bos>", "<eos>"]), "<eos>", "verb") == 1


class TestNormalizeExpression:
    def test_v_aliases_resolve_to_x(self) -> None:
        # FastSRB and several catalogs name variables v1..vn; the vocabulary is x1..xN. The
        # capability probes did this rename by hand at every call site.
        out = _normalize_expression(_Estimator(), ["<mul>", "v1", "v12", "</mul>"])
        assert out == ["<mul>", "x1", "x12", "</mul>"]

    def test_x_names_are_untouched(self) -> None:
        assert _normalize_expression(_Estimator(), ["<mul>", "x1", "</mul>"]) == ["<mul>", "x1", "</mul>"]

    def test_lookalikes_are_not_renamed(self) -> None:
        # 'var'/'v' are not the alias pattern; only a full v<digits> match is.
        assert _normalize_expression(_Estimator(), ["v", "var", "v1x"]) == ["v", "var", "v1x"]

    def test_infix_string_is_parsed(self) -> None:
        assert _normalize_expression(_Estimator(), "<add> v1 <constant>") == ["<add>", "x1", "<constant>"]


class TestEncoderBatch:
    def test_packs_x_and_y_into_one_batch(self) -> None:
        data, mask = _encoder_batch(np.zeros((16, 2), dtype=np.float32), np.ones(16), 3, "cpu")
        assert data.shape == (1, 16, 4)      # padded to n_variables + the y column
        assert mask.shape == (1, 16)
        assert bool(mask.all())

    def test_one_dimensional_x_is_promoted(self) -> None:
        data, _ = _encoder_batch(np.zeros(8, dtype=np.float32), np.ones(8), 1, "cpu")
        assert data.shape == (1, 8, 2)

    def test_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="agree on the number of points"):
            _encoder_batch(np.zeros((16, 1)), np.ones(8), 1, "cpu")

    def test_non_finite_target_is_refused(self) -> None:
        y = np.ones(8); y[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _encoder_batch(np.zeros((8, 1)), y, 1, "cpu")


class TestValueDistribution:
    """A predicted value is a DISTRIBUTION: every nibble is a softmax draw."""

    def test_default_sample_count_is_not_one(self) -> None:
        # The whole point: one decode is one sample, so no verb may default to a single draw.
        assert DEFAULT_SAMPLES > 1

    def test_summaries_over_finite_draws(self) -> None:
        d = ValueDistribution(draws=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert d.n == 5
        assert d.median == 3.0
        assert d.q05 < d.median < d.q95

    def test_non_finite_draws_are_counted_not_quantiled(self) -> None:
        # A nibble pattern can decode to nan/inf. Those are real outcomes of the distribution, but
        # they cannot enter a quantile without destroying it.
        d = ValueDistribution(draws=[1.0, float("nan"), 3.0, float("inf")])
        assert d.finite_draws == [1.0, 3.0]
        assert d.non_finite_fraction == pytest.approx(0.5)
        assert d.median == 2.0

    def test_mode_and_agreement_describe_concentration(self) -> None:
        d = ValueDistribution(draws=[2.0, 2.0, 2.0, 5.0])
        assert d.mode == 2.0
        assert d.agreement == pytest.approx(0.75)

    def test_scattered_draws_have_low_agreement(self) -> None:
        d = ValueDistribution(draws=[1.0, 2.0, 3.0, 4.0])
        assert d.agreement == pytest.approx(0.25)

    def test_all_non_finite_is_not_a_crash(self) -> None:
        d = ValueDistribution(draws=[float("nan"), float("inf")])
        assert np.isnan(d.median) and np.isnan(d.q05) and d.agreement == 0.0

    def test_empty_distribution(self) -> None:
        d = ValueDistribution()
        assert d.n == 0 and np.isnan(d.median) and d.non_finite_fraction == 0.0

    def test_complexity_carries_the_hypothesis_diagnostic(self) -> None:
        d = ComplexityDistribution(draws=[146000.0, 150000.0], self_initiated_fraction=1.0)
        assert d.median == pytest.approx(148000.0)
        assert d.self_initiated_fraction == 1.0

    def test_repr_summarises_rather_than_dumping_draws(self) -> None:
        assert "median" in repr(ValueDistribution(draws=[1.0, 2.0]))


class TestComplexityUnit:
    def test_mu_is_a_simplipy_unit_not_a_token_count(self) -> None:
        # The unit confusion the audit filed four findings against: mu runs 1e3-1e6, a token count
        # runs ~1e1. Draws shaped like a token count would mean the block is being misread.
        d = ComplexityDistribution(draws=[234500.0, 76000.0, 454500.0])
        assert d.median > 1e3


class TestCandidateCarriesBothConstantSets:
    def test_emitted_and_refined_are_separate_fields(self) -> None:
        # constants_emitted is what the MODEL said; constants is what the optimizer made of it.
        # Measured on FastSRB: refinement improves FVU on only 55% of comparable rows, so keeping
        # only the refined values discards a competitive answer on nearly half of them.
        from flash_ansr.inference import Candidate
        c = Candidate(
            raw_beam=[1, 2], expression=['+', '<constant>', 'x1'], expression_prefix=['+', '2.0', 'x1'],
            expression_infix='2.0 + x1', skeleton_prefix=['+', '<constant>', 'x1'],
            constants=[2.5], constants_emitted=[2.0], log_prob=-1.0, score=-3.0, fvu=1e-15,
            complexity=3, mu=146000.0, constant_count=1, pruned_variant=False)
        assert c.constants == [2.5]
        assert c.constants_emitted == [2.0]

    def test_v23_beams_carry_no_emitted_constants(self) -> None:
        from flash_ansr.inference import Candidate
        c = Candidate(
            raw_beam=[1], expression=['x1'], expression_prefix=['x1'], expression_infix='x1',
            skeleton_prefix=['x1'], constants=[], constants_emitted=None, log_prob=-1.0,
            score=-1.0, fvu=0.5, complexity=1, mu=None, constant_count=0, pruned_variant=False)
        assert c.constants_emitted is None
