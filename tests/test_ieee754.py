"""T1 — IEEE-754 binary32 token codec round-trip (v24 constants representation contract).

float32 -> 8 hex-nibble tokens -> float32 must be the identity for every finite float32,
including denormals and extremes. Non-finite values must raise at serialization time
(assert, don't assume).

Format ruling 2026-08-18: the 32 bit tokens are replaced by 8 hex-nibble tokens over a
16-symbol alphabet (same representative power, fewer tokens); a full span is
``<ieee754>`` + 8 nibbles + ``</ieee754>`` = 10 tokens (was 34).
"""
import math
import struct

import numpy as np
import pytest

from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_N_NIBBLES,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
    float32_to_nibble_tokens,
    nibble_tokens_to_float32,
    unwrap_ieee754_span,
    wrap_float32,
)


def _bits_from_uint32(pattern: int) -> float:
    return struct.unpack(">f", pattern.to_bytes(4, "big"))[0]


# Curated extremes: zeros, denormal boundaries, normal boundaries, common values.
EXTREME_VALUES = [
    0.0,
    -0.0,
    _bits_from_uint32(0x00000001),   # smallest positive denormal
    _bits_from_uint32(0x807FFFFF),   # largest negative denormal
    _bits_from_uint32(0x007FFFFF),   # largest positive denormal
    _bits_from_uint32(0x00800000),   # smallest positive normal
    float(np.finfo(np.float32).max),
    float(np.finfo(np.float32).min),
    float(np.finfo(np.float32).tiny),
    1.0,
    -1.0,
    float(np.float32(np.pi)),
    float(np.float32(1e-40)),        # denormal magnitude
    float(np.float32(3.14159e37)),
]


def test_t1_nibble_alphabet_is_sixteen_dedicated_tokens() -> None:
    # The 16-symbol alphabet, in nibble-value order <h0> .. <hf>; span = 10 tokens.
    assert len(NIBBLE_TOKENS) == 16
    assert len(set(NIBBLE_TOKENS)) == 16
    assert NIBBLE_TOKENS == tuple(f"<h{digit:x}>" for digit in range(16))
    assert IEEE754_N_NIBBLES == 8
    assert IEEE754_SPAN_LENGTH == 10


def test_t1_roundtrip_extremes_and_denormals() -> None:
    for value in EXTREME_VALUES:
        tokens = float32_to_nibble_tokens(value)
        assert len(tokens) == IEEE754_N_NIBBLES
        assert all(token in NIBBLE_TOKENS for token in tokens)
        recovered = nibble_tokens_to_float32(tokens)
        # Bit-exact identity, including the sign of zero.
        assert struct.pack(">f", recovered) == struct.pack(">f", np.float32(value))


def test_t1_roundtrip_large_random_sample() -> None:
    rng = np.random.default_rng(0xC1)
    patterns = rng.integers(0, 2**32, size=10_000, dtype=np.uint64).astype(np.uint32)
    checked = 0
    for pattern in patterns:
        value = _bits_from_uint32(int(pattern))
        if not math.isfinite(value):
            continue  # non-finite payloads are not producible constants
        tokens = float32_to_nibble_tokens(value)
        recovered = nibble_tokens_to_float32(tokens)
        assert struct.pack(">f", recovered) == int(pattern).to_bytes(4, "big")
        checked += 1
    assert checked > 9_000  # the sample must actually be large


def test_t1_nibble_order_is_big_endian_msb_first() -> None:
    # 0x01234567 exercises eight DISTINCT nibbles: the tokens must read off the
    # big-endian hex spelling of the bit pattern, most-significant nibble first.
    tokens = float32_to_nibble_tokens(_bits_from_uint32(0x01234567))
    assert tokens == [NIBBLE_TOKENS[digit] for digit in (0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7)]

    # -2.0 = 0xC0000000: sign+exponent land in the LEADING nibbles (an asymmetric witness).
    tokens = float32_to_nibble_tokens(-2.0)
    hex_string = "".join(token[2] for token in tokens)  # '<hX>' -> 'X'
    assert hex_string == "c0000000"


def test_t1_nonfinite_serialization_raises() -> None:
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            float32_to_nibble_tokens(value)
    # float64 value that overflows float32 must raise too, not silently become inf.
    with pytest.raises(ValueError):
        float32_to_nibble_tokens(1e39)


def test_t1_span_wrap_unwrap() -> None:
    span = wrap_float32(1.5)
    assert len(span) == IEEE754_SPAN_LENGTH == 10
    assert span[0] == IEEE754_START_TOKEN
    assert span[-1] == IEEE754_END_TOKEN
    assert unwrap_ieee754_span(span) == 1.5


def test_t1_unwrap_rejects_malformed_spans() -> None:
    good = wrap_float32(2.5)
    with pytest.raises(ValueError):
        unwrap_ieee754_span(good[1:])  # missing start tag
    with pytest.raises(ValueError):
        unwrap_ieee754_span(good[:-1])  # missing end tag
    with pytest.raises(ValueError):
        unwrap_ieee754_span([IEEE754_START_TOKEN, NIBBLE_TOKENS[0], IEEE754_END_TOKEN])  # wrong width
    with pytest.raises(ValueError):
        nibble_tokens_to_float32(["0"] * 8)  # literal '0' is not a nibble token
    with pytest.raises(ValueError):
        nibble_tokens_to_float32(["<b0>"] * 8)  # retired bit tokens are not nibble tokens
