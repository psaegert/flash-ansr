"""The auxiliary task verbs: capabilities T16 was trained on that nothing could call.

Unit-level coverage of the boundary behaviour (alias resolution, refusals, batching); the decode
loops themselves are exercised against a real checkpoint in the capability scripts.
"""
import numpy as np
import pytest
import torch

from flash_ansr.preprocessing import CapabilityUnavailable
from flash_ansr.tasks import (
    NIBBLES_PER_SPAN,
    ConstantPrediction,
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


class TestConstantPrediction:
    def test_defaults_are_empty_not_none(self) -> None:
        pred = ConstantPrediction()
        assert pred.values == [] and pred.nibble_logprobs == []
        assert pred.closed_cleanly == [] and pred.off_grammar_steps == 0

    def test_a_float32_is_exactly_eight_nibbles(self) -> None:
        assert NIBBLES_PER_SPAN == 8
