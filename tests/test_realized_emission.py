"""The candidate post-processing hands on is the one the model EMITTED (owner ruling 2026-09-12).

Post-processing used ONE representation for two incompatible jobs: a lossy dedup key and the
candidate payload. Because the payload had to round-trip through token ids, and a v24 vocabulary has
no numeral tokens, the key's `mask_all` was forced onto the candidate -- so a beam that mixed a bare
`<constant>` with a spelled `<ieee754>` span (exactly what `<mask_fittable>` trains the model to
produce) was rewritten into an all-masked skeleton and its prediction deleted. Measured on
v25.0-T8-20M: 71.7 % of valid candidates were mixed, 0 survived.

The key may be lossy; the payload may not. These tests pin that separation.
"""
import pytest

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.data.serialization import map_ieee754_spans, realize_slot_values
from flash_ansr.utils.ieee754 import (IEEE754_START_TOKEN, IEEE754_END_TOKEN, BYTE_TOKENS,
                                      wrap_float64)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("base", install=True)


class _Host:
    """The attributes `_postprocess_sampled` actually reads, plus the span readers."""

    _span_ids = FlashANSRModel._span_ids
    _map_ieee754_spans = FlashANSRModel._map_ieee754_spans
    _realize_ieee754_spans = FlashANSRModel._realize_ieee754_spans
    _postprocess_sampled = FlashANSRModel._postprocess_sampled
    extract_valid_raw_expressions = FlashANSRModel.extract_valid_raw_expressions

    def __init__(self, tokenizer, engine):  # type: ignore[no-untyped-def]
        self.tokenizer = tokenizer
        self.simplipy_engine = engine


def _ids(tokenizer: Tokenizer, *tokens: str) -> list[int]:
    return [int(tokenizer[token]) for token in tokens]


def _span(tokenizer: Tokenizer, value: float) -> list[int]:
    return [int(tokenizer[token]) for token in wrap_float64(value)]


def _seq(tokenizer: Tokenizer, body: list[int]) -> list[int]:
    return [int(tokenizer["<bos>"]), int(tokenizer["<expression>"]),
            *body, int(tokenizer["</expression>"]), int(tokenizer["<eos>"])]


def _shape(tokenizer: Tokenizer, seq: list[int]) -> tuple[int, int]:
    """``(spans, bare placeholders)`` in a sequence's expression body."""
    body, _before, _after = tokenizer.extract_expression_from_beam(seq)
    return (sum(1 for t in body if int(t) == int(tokenizer[IEEE754_START_TOKEN])),
            sum(1 for t in body if int(t) == int(tokenizer["<constant>"])))


class TestMapper:
    """`map_ieee754_spans` reports one value PER SLOT, so a mixed emission is describable."""

    def _call(self, tokenizer, ids):  # type: ignore[no-untyped-def]
        return map_ieee754_spans(
            ids, start_id=int(tokenizer[IEEE754_START_TOKEN]), end_id=int(tokenizer[IEEE754_END_TOKEN]),
            byte_ids=[int(tokenizer[t]) for t in BYTE_TOKENS], constant_id=int(tokenizer["<constant>"]))

    def test_a_mixed_emission_reports_none_for_the_masked_slot(self, tokenizer):  # type: ignore[no-untyped-def]
        ids = [*_ids(tokenizer, "*", "<constant>"), *_span(tokenizer, 2.0)]
        mapped, values = self._call(tokenizer, ids)
        assert tokenizer.decode_expression(mapped) == ["*", "<constant>", "<constant>"]
        assert values == [None, 2.0]          # the model masked the first and spelled the second

    def test_all_spelled_and_all_masked(self, tokenizer):  # type: ignore[no-untyped-def]
        _mapped, values = self._call(tokenizer, [*_ids(tokenizer, "*"), *_span(tokenizer, 3.5),
                                                 *_span(tokenizer, -2.0)])
        assert values == [3.5, -2.0]
        _mapped, values = self._call(tokenizer, _ids(tokenizer, "*", "<constant>", "<constant>"))
        assert values == [None, None]

    def test_a_malformed_carrier_is_refused(self, tokenizer):  # type: ignore[no-untyped-def]
        truncated = [*_ids(tokenizer, "*", "x1"), *_span(tokenizer, 2.0)[:-1]]
        assert self._call(tokenizer, truncated) is None
        stray = [*_ids(tokenizer, "*", "x1"), int(tokenizer[IEEE754_END_TOKEN])]
        assert self._call(tokenizer, stray) is None


class TestRealization:
    def test_values_are_spelled_back_into_their_slots(self) -> None:
        assert realize_slot_values(["pow", "+", "x1", "<constant>", "<constant>"], [None, 2.0]) == \
            ["pow", "+", "x1", "<constant>", "2"]
        assert realize_slot_values(["*", "<constant>", "x1"], [3.5]) == ["*", "3.5", "x1"]
        assert realize_slot_values(["*", "<constant>", "x1"], [None]) == ["*", "<constant>", "x1"]

    def test_a_slot_count_mismatch_raises_rather_than_misattributing(self) -> None:
        with pytest.raises(ValueError, match="slots"):
            realize_slot_values(["*", "<constant>", "<constant>"], [2.0])


class TestPostProcessingPreservesTheEmission:
    """The regression test for the defect: a mixed beam must come out as it went in."""

    def test_a_mixed_beam_survives_intact(self, tokenizer, engine):  # type: ignore[no-untyped-def]
        host = _Host(tokenizer, engine)
        seq = _seq(tokenizer, [*_ids(tokenizer, "*", "<constant>", "x1")])
        mixed = _seq(tokenizer, [*_ids(tokenizer, "+", "<constant>"), *_span(tokenizer, 2.0)])
        for label, beam in (("all masked", seq), ("mixed", mixed)):
            kept, _scores, _valid = host._postprocess_sampled(
                [list(beam)], [0.0], simplify=True, unique=True, valid_only=True)
            assert kept, label
            assert list(kept[0]) == list(beam), f"{label}: the payload was rewritten"
            assert _shape(tokenizer, list(kept[0])) == _shape(tokenizer, list(beam)), label

    def test_an_all_spelled_beam_still_survives(self, tokenizer, engine):  # type: ignore[no-untyped-def]
        host = _Host(tokenizer, engine)
        beam = _seq(tokenizer, [*_ids(tokenizer, "*", "x1"), *_span(tokenizer, 2.5)])
        kept, _scores, _valid = host._postprocess_sampled(
            [list(beam)], [0.0], simplify=True, unique=True, valid_only=True)
        assert kept and list(kept[0]) == list(beam)

    def test_the_dedup_key_is_the_emitted_expression_so_values_separate_candidates(self, tokenizer, engine):  # type: ignore[no-untyped-def]
        """Two beams differing only in a SPELLED value are two candidates; identical ones are one."""
        host = _Host(tokenizer, engine)
        two = _seq(tokenizer, [*_ids(tokenizer, "*", "x1"), *_span(tokenizer, 2.0)])
        three = _seq(tokenizer, [*_ids(tokenizer, "*", "x1"), *_span(tokenizer, 3.0)])
        kept, _s, _v = host._postprocess_sampled([list(two), list(three)], [0.0, -1.0],
                                                 simplify=True, unique=True, valid_only=True)
        assert len(kept) == 2
        kept, _s, _v = host._postprocess_sampled([list(two), list(two)], [0.0, -1.0],
                                                 simplify=True, unique=True, valid_only=True)
        assert len(kept) == 1

    def test_the_chunked_path_keys_on_the_same_realization(self, tokenizer, engine):  # type: ignore[no-untyped-def]
        """`extract_valid_raw_expressions` must produce the simplify_map KEYS `_postprocess_sampled`
        looks up -- otherwise the parallel simplify silently misses and the two paths diverge."""
        host = _Host(tokenizer, engine)
        beam = _seq(tokenizer, [*_ids(tokenizer, "*", "<constant>"), *_span(tokenizer, 2.0)])
        keys = host.extract_valid_raw_expressions([list(beam)])
        assert keys == [["*", "<constant>", "2"]]


class TestSugarSpellsItsFactor:
    """`mult4` asserts the coefficient IS 4; rewriting it to `* <constant>` discards a prediction."""

    def test_exact_factors(self, tokenizer):  # type: ignore[no-untyped-def]
        if "mult4" not in tokenizer.token2idx:
            pytest.skip("the test vocabulary has no integer-factor sugar")
        assert tokenizer.constantify_expression(["mult4", "x1"], exact=True) == ["*", "4", "x1"]
        assert tokenizer.constantify_expression(["mult4", "x1"]) == ["*", "<constant>", "x1"]
