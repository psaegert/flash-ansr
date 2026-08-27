"""In-decode span compaction on the SAMPLING loop (the static position-indexed path).

Compaction used to live on beam search, not because sampling could not host it but
because it was built on the dynamic cat-grow cache and beam extended what was already
there. On the static path it is a per-row position rewind and needs no cache surgery at
all -- see tests/test_models/test_static_position_rewind.py for the mechanism and its
golden equality. These tests cover the LOOP: that it schedules the rewind, that the
tokens it emits stay well-formed, and that the pairing gates fire.
"""
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.generation import SoftmaxSamplingConfig
from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
)
from flash_ansr.utils.numeric import NUMERIC_DTYPE


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def model(tokenizer):  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    torch.manual_seed(0x24C2)
    m = FlashANSRModel(simplipy_engine=SimpliPyEngine.load("base", install=True),
                       tokenizer=tokenizer, **kwargs).eval()
    with torch.no_grad():   # bias it to actually emit spans, or the tests are vacuous
        m.next_token_head[-1].bias[tokenizer[IEEE754_START_TOKEN]] += 8.0
    return m


def _sample(model, tokenizer, compact: bool, seed: int = 11):
    torch.manual_seed(seed)
    raw, _ = model.sample_top_kp(
        torch.rand(13, 11, dtype=NUMERIC_DTYPE), choices=8, max_len=40, batch_size=8,
        return_raw=True, initial_tokens=[tokenizer["<bos>"]], static_decode=True,
        constrain_ieee754=True, compact_ieee754=compact)
    return raw


def _count_wellformed_spans(sequences, tokenizer) -> int:
    """Every span is open + exactly 8 nibbles + close, and no tag strays outside one."""
    open_id, close_id = tokenizer[IEEE754_START_TOKEN], tokenizer[IEEE754_END_TOKEN]
    nibble_ids = {int(tokenizer[token]) for token in NIBBLE_TOKENS}
    found = 0
    for sequence in sequences:
        i = 0
        while i < len(sequence):
            if sequence[i] == open_id:
                assert i + IEEE754_SPAN_LENGTH <= len(sequence), f"unterminated span: {sequence}"
                assert sequence[i + IEEE754_SPAN_LENGTH - 1] == close_id, f"bad close: {sequence}"
                inner = sequence[i + 1:i + IEEE754_SPAN_LENGTH - 1]
                assert all(t in nibble_ids for t in inner), f"non-nibble inside a span: {sequence}"
                found += 1
                i += IEEE754_SPAN_LENGTH
            else:
                assert sequence[i] != close_id, f"stray close tag: {sequence}"
                i += 1
    return found


def test_compacted_decode_still_emits_wellformed_expanded_spans(model, tokenizer):
    """Compaction changes what the model CONDITIONS on, never what it returns: the token
    buffer stays expanded, which is the one carrier everything downstream reads."""
    without = _sample(model, tokenizer, compact=False)
    with_ = _sample(model, tokenizer, compact=True)
    assert _count_wellformed_spans(without, tokenizer) > 0, "vacuous: no spans were emitted"
    assert _count_wellformed_spans(with_, tokenizer) > 0


def _count_compact_forwards(model, tokenizer, compact: bool, monkeypatch) -> int:
    """How many times a <float> token is actually fed to the decoder -- i.e. how many spans
    were compacted. Observed at `forward_static`, not inferred from the output."""
    float_id = int(tokenizer["<float>"])
    seen = 0
    original = model.forward_static

    def counting(input_tokens, *args, **kwargs):
        nonlocal seen
        seen += int((input_tokens == float_id).sum())
        return original(input_tokens, *args, **kwargs)

    monkeypatch.setattr(model, "forward_static", counting)
    _sample(model, tokenizer, compact=compact)
    return seen


def test_compaction_actually_fires(model, tokenizer, monkeypatch):
    """Counted where it happens. Asserting on output DIVERGENCE instead would be flaky: a
    span that closes with nothing after it changes no later token, so identical outputs do
    not imply the rewind was skipped."""
    assert _count_compact_forwards(model, tokenizer, False, monkeypatch) == 0
    assert _count_compact_forwards(model, tokenizer, True, monkeypatch) > 0


def test_the_token_buffer_is_not_aliased_by_the_rewrite(model, tokenizer):
    """Regression: the per-step token tensor is a VIEW onto `sequences` (slice indexing), so
    writing <float> into it without a copy clobbers the </ieee754> the grammar is still
    waiting for -- and the decode then emits a second close tag."""
    close_id = tokenizer[IEEE754_END_TOKEN]
    float_id = tokenizer["<float>"]
    for sequence in _sample(model, tokenizer, compact=True):
        assert float_id not in sequence, "the compact token must never reach the output buffer"
        for i, token in enumerate(sequence):
            if token == close_id:
                start = i - IEEE754_SPAN_LENGTH + 1
                assert start >= 0 and sequence[start] == tokenizer[IEEE754_START_TOKEN]


@pytest.mark.parametrize("kwargs,expected", [
    ({"constrain_ieee754": False, "static_decode": True}, "constrain_ieee754=True"),
    ({"constrain_ieee754": True, "static_decode": False}, "static_decode=True"),
])
def test_the_decode_refuses_an_unsupported_pairing(model, tokenizer, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        model.sample_top_kp(
            torch.rand(13, 11, dtype=NUMERIC_DTYPE), choices=2, max_len=20, return_raw=True,
            initial_tokens=[tokenizer["<bos>"]], compact_ieee754=True, **kwargs)


def test_the_config_refuses_the_same_pairings_at_construction():
    """Validated at config time too, where the traceback names the knob to change."""
    assert SoftmaxSamplingConfig().to_kwargs()['compact_ieee754'] is False
    assert SoftmaxSamplingConfig(constrain_ieee754=True,
                                 compact_ieee754=True).to_kwargs()['compact_ieee754'] is True
    with pytest.raises(ValueError, match="constrain_ieee754=True"):
        SoftmaxSamplingConfig(compact_ieee754=True)
    with pytest.raises(ValueError, match="static decode"):
        SoftmaxSamplingConfig(constrain_ieee754=True, compact_ieee754=True, static_decode=False)
