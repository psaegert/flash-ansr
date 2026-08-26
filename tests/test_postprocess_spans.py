"""Post-processing must not destroy v24 ieee754 constant emissions (contract T11).

The sampling post-processor (`_postprocess_sampled`) and the chunked-path key
collector (`extract_valid_raw_expressions`) judged validity/simplify/dedup on the
RAW token stream: an expanded `<ieee754>` span made the expression simplipy-invalid,
so every constant-bearing sample was silently dropped (measured on v24.0-T16:
57/64 raw FastSRB samples carried spans, 0 survived post-processing) and the T11
verbatim-init handshake could never fire from the deployed path. These tests pin
the span-mapped behavior: skeleton-level validity/dedup, RAW sequence preserved.
"""
import pytest

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.data.serialization import replace_ieee754_spans_with_constants
from flash_ansr.utils.ieee754 import IEEE754_START_TOKEN, IEEE754_END_TOKEN, NIBBLE_TOKENS, wrap_float32


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("base", install=True)


class _Host:
    """The two attributes `_postprocess_sampled` actually reads, plus the helper."""

    _map_ieee754_spans = FlashANSRModel._map_ieee754_spans
    _postprocess_sampled = FlashANSRModel._postprocess_sampled
    extract_valid_raw_expressions = FlashANSRModel.extract_valid_raw_expressions

    def __init__(self, tokenizer, engine):  # type: ignore[no-untyped-def]
        self.tokenizer = tokenizer
        self.simplipy_engine = engine


def _span_ids(tokenizer: Tokenizer, value: float) -> list[int]:
    return [int(tokenizer[token]) for token in wrap_float32(value)]


def _seq(tokenizer: Tokenizer, body: list[int]) -> list[int]:
    return [int(tokenizer["<bos>"]), int(tokenizer["<expression>"]),
            *body, int(tokenizer["</expression>"]), int(tokenizer["<eos>"])]


def _carrier(tokenizer: Tokenizer, value: float) -> list[int]:
    return _seq(tokenizer, [int(tokenizer["*"]), int(tokenizer["x1"]), *_span_ids(tokenizer, value)])


def _recover_values(tokenizer: Tokenizer, seq: list[int]) -> list[float] | None:
    expr, _, _ = tokenizer.extract_expression_from_beam(seq)
    _, values = replace_ieee754_spans_with_constants(
        expr,
        start_id=int(tokenizer[IEEE754_START_TOKEN]),
        end_id=int(tokenizer[IEEE754_END_TOKEN]),
        nibble_ids=[int(tokenizer[t]) for t in NIBBLE_TOKENS],
        constant_id=int(tokenizer["<constant>"]),
    )
    return values


def test_span_carrier_survives_with_values_intact(tokenizer, engine):  # type: ignore[no-untyped-def]
    host = _Host(tokenizer, engine)
    seqs, scores, valid = host._postprocess_sampled([_carrier(tokenizer, 2.0)], [-1.0])
    assert len(seqs) == 1 and valid == [True]
    assert _recover_values(tokenizer, seqs[0]) == [2.0]


def test_span_free_path_unchanged(tokenizer, engine):  # type: ignore[no-untyped-def]
    host = _Host(tokenizer, engine)
    body = [int(tokenizer["*"]), int(tokenizer["x1"]), int(tokenizer["x2"])]
    seqs, scores, valid = host._postprocess_sampled([_seq(tokenizer, body)], [-1.0])
    assert len(seqs) == 1 and valid == [True]
    start_id = int(tokenizer[IEEE754_START_TOKEN])
    assert start_id not in seqs[0]


def test_malformed_span_still_dropped(tokenizer, engine):  # type: ignore[no-untyped-def]
    host = _Host(tokenizer, engine)
    truncated = _span_ids(tokenizer, 2.0)[:5] + [int(tokenizer[IEEE754_END_TOKEN])]
    seqs, _, _ = host._postprocess_sampled(
        [_seq(tokenizer, [int(tokenizer["*"]), int(tokenizer["x1"]), *truncated])], [-1.0])
    assert seqs == []


def test_dedup_is_by_skeleton_across_constant_values(tokenizer, engine):  # type: ignore[no-untyped-def]
    host = _Host(tokenizer, engine)
    both = [_carrier(tokenizer, 2.0), _carrier(tokenizer, 3.0)]
    seqs, _, _ = host._postprocess_sampled(both, [-1.0, -2.0])
    assert len(seqs) == 1  # same skeleton -> deduped, first (best-scored) survives
    assert _recover_values(tokenizer, seqs[0]) == [2.0]
    seqs, _, _ = host._postprocess_sampled(both, [-1.0, -2.0], unique=False)
    assert len(seqs) == 2


def test_chunked_key_collector_maps_spans(tokenizer, engine):  # type: ignore[no-untyped-def]
    host = _Host(tokenizer, engine)
    exprs = host.extract_valid_raw_expressions([_carrier(tokenizer, 2.0)])
    assert len(exprs) == 1
    assert "<constant>" in exprs[0]
    assert not any(t.startswith("<h") for t in exprs[0])
