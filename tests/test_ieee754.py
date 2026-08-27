"""T1 — IEEE-754 binary64 token codec round-trip (the v25 constants representation contract).

float64 -> 8 BYTE tokens -> float64 must be the identity for every finite float64, including
denormals and extremes. Non-finite values must raise at serialization time (assert, don't
assume).

Format rulings: 2026-08-18 replaced 32 bit tokens with 8 hex nibbles over a 16-symbol
alphabet; 2026-08-27 replaced those with 8 BYTES over a 256-symbol alphabet at binary64.
The span width is unchanged through both -- 8 nibbles of binary32 and 8 bytes of binary64
are both 8 content positions, so a span is still ``<ieee754>`` + 8 + ``</ieee754>`` = 10
tokens. What changed is the alphabet (16 -> 256) and the value semantics (f32 -> f64).
"""
import math
import struct

import numpy as np
import pytest

from flash_ansr.utils.ieee754 import (
    BYTE_TOKENS,
    IEEE754_END_TOKEN,
    IEEE754_N_BYTES,
    IEEE754_N_BYTE_SYMBOLS,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    byte_tokens_to_float64,
    byte_values_to_float64,
    float64_to_byte_tokens,
    float64_to_byte_values,
    unwrap_ieee754_span,
    wrap_float64,
)


def _bits_from_uint64(pattern: int) -> float:
    return struct.unpack(">d", pattern.to_bytes(8, "big"))[0]


# Curated extremes: zeros, denormal boundaries, normal boundaries, common values.
EXTREME_VALUES = [
    0.0,
    -0.0,
    _bits_from_uint64(0x0000000000000001),   # smallest positive denormal
    _bits_from_uint64(0x800FFFFFFFFFFFFF),   # largest negative denormal
    _bits_from_uint64(0x000FFFFFFFFFFFFF),   # largest positive denormal
    _bits_from_uint64(0x0010000000000000),   # smallest positive normal
    float(np.finfo(np.float64).max),
    float(np.finfo(np.float64).min),
    float(np.finfo(np.float64).tiny),
    1.0,
    -1.0,
    math.pi,
    1e-320,                                  # denormal magnitude in binary64
    1e308,
    1e39,                                    # overflowed float32; a plain normal at binary64
]


def test_t1_byte_alphabet_is_256_dedicated_tokens() -> None:
    assert len(BYTE_TOKENS) == IEEE754_N_BYTE_SYMBOLS == 256
    assert len(set(BYTE_TOKENS)) == 256
    # TWO hex digits, always (Q3). One digit would re-emit <b0>..<bf> for values 0..15 --
    # the RETIRED bit tokens -- and a stale vocabulary would then pass a presence check
    # while silently meaning something else.
    assert BYTE_TOKENS == tuple(f"<b{value:02x}>" for value in range(256))
    assert all(len(token) == len("<bff>") for token in BYTE_TOKENS)
    # The span is UNCHANGED by the migration: 8 content positions either way.
    assert IEEE754_N_BYTES == 8
    assert IEEE754_SPAN_LENGTH == 10


def test_t1_content_values_above_fifteen_round_trip() -> None:
    """THE migration test. Three sites spelled "4 bits per position" as bare literals
    (``& 0xF`` in both vectorized halves, ``<< 4`` in the token decoder). Two of them
    CORRUPT rather than raise if left behind, and the consumer that would surface it casts
    to uint8, which holds 0..255 happily. Nothing else in the suite exercises a content
    value above 15, so nothing else would notice."""
    # Every byte of this pattern exceeds 15, so a stray `& 0xF` truncates all eight and a
    # stray `<< 4` reassembles them at the wrong stride.
    pattern = 0x412C3D4E5F6A7B8C
    assert all(byte > 0xF for byte in pattern.to_bytes(8, "big"))

    value = _bits_from_uint64(pattern)
    tokens = float64_to_byte_tokens(value)
    assert [BYTE_TOKENS.index(token) for token in tokens] == list(pattern.to_bytes(8, "big"))
    assert struct.pack(">d", byte_tokens_to_float64(tokens)) == pattern.to_bytes(8, "big")

    # The vectorized pair must agree with the token pair, and must span the full alphabet.
    values = np.array([_bits_from_uint64(0x0102030405060708),
                       _bits_from_uint64(pattern),
                       _bits_from_uint64(0xFFEEDDCCBBAA9988)])
    byte_values = float64_to_byte_values(values)
    assert byte_values.shape == (3, IEEE754_N_BYTES)
    assert byte_values.max() > 0xF, "vacuous: no content value above 15"
    assert byte_values.max() < IEEE754_N_BYTE_SYMBOLS
    assert list(byte_values[1]) == list(pattern.to_bytes(8, "big"))
    recovered = byte_values_to_float64(byte_values)
    assert [struct.pack(">d", v) for v in recovered] == [struct.pack(">d", v) for v in values]


def test_t1_roundtrip_extremes_and_denormals() -> None:
    for value in EXTREME_VALUES:
        tokens = float64_to_byte_tokens(value)
        assert len(tokens) == IEEE754_N_BYTES
        assert all(token in BYTE_TOKENS for token in tokens)
        recovered = byte_tokens_to_float64(tokens)
        # Bit-exact identity, including the sign of zero.
        assert struct.pack(">d", recovered) == struct.pack(">d", value)


def test_t1_roundtrip_large_random_sample() -> None:
    rng = np.random.default_rng(0xC1)
    patterns = rng.integers(0, 2**64, size=10_000, dtype=np.uint64)
    checked = 0
    for pattern in patterns:
        value = _bits_from_uint64(int(pattern))
        if not math.isfinite(value):
            continue  # non-finite payloads are not producible constants
        tokens = float64_to_byte_tokens(value)
        recovered = byte_tokens_to_float64(tokens)
        assert struct.pack(">d", recovered) == int(pattern).to_bytes(8, "big")
        checked += 1
    assert checked > 9_000  # the sample must actually be large


def test_t1_byte_order_is_big_endian_msb_first() -> None:
    # Eight DISTINCT bytes: the tokens read off the big-endian spelling, MSB first.
    tokens = float64_to_byte_tokens(_bits_from_uint64(0x0123456789ABCDEF))
    assert tokens == [BYTE_TOKENS[b] for b in (0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF)]

    # -2.0 = 0xc000000000000000: sign+exponent land in the LEADING byte (asymmetric witness).
    tokens = float64_to_byte_tokens(-2.0)
    assert tokens == [BYTE_TOKENS[0xC0], *[BYTE_TOKENS[0x00]] * 7]
    assert "".join(token[2:-1] for token in tokens) == "c000000000000000"


def test_t1_nonfinite_serialization_raises() -> None:
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            float64_to_byte_tokens(value)


def test_t1_float32_overflow_is_no_longer_an_error() -> None:
    """The inversion this migration exists for. 1e39 overflowed binary32 and the codec
    refused it; at binary64 it is an ordinary normal and must serialize exactly."""
    assert float(np.float32(1e39)) == float("inf")  # the old refusal was correct THEN
    assert byte_tokens_to_float64(float64_to_byte_tokens(1e39)) == 1e39
    assert unwrap_ieee754_span(wrap_float64(1e-320)) == 1e-320


def test_t1_precision_is_not_silently_narrowed() -> None:
    """A value distinguishable only beyond the binary32 mantissa must survive. Under the
    old codec both of these collapsed onto the same float32."""
    a = 1.0 + 2 ** -40
    b = 1.0 + 2 ** -41
    assert float(np.float32(a)) == float(np.float32(b)) == 1.0
    assert byte_tokens_to_float64(float64_to_byte_tokens(a)) == a
    assert float64_to_byte_tokens(a) != float64_to_byte_tokens(b)


def test_t1_span_wrap_unwrap() -> None:
    span = wrap_float64(1.5)
    assert len(span) == IEEE754_SPAN_LENGTH == 10
    assert span[0] == IEEE754_START_TOKEN
    assert span[-1] == IEEE754_END_TOKEN
    assert unwrap_ieee754_span(span) == 1.5


def test_t1_unwrap_rejects_malformed_spans() -> None:
    good = wrap_float64(2.5)
    with pytest.raises(ValueError):
        unwrap_ieee754_span(good[1:])   # missing start tag
    with pytest.raises(ValueError):
        unwrap_ieee754_span(good[:-1])  # missing end tag
    with pytest.raises(ValueError):
        unwrap_ieee754_span([IEEE754_START_TOKEN, BYTE_TOKENS[0], IEEE754_END_TOKEN])  # width
    with pytest.raises(ValueError):
        byte_tokens_to_float64(["0"] * 8)      # literal '0' is not a byte token
    with pytest.raises(ValueError):
        byte_tokens_to_float64(["<b0>"] * 8)   # retired BIT token; byte tokens are 2-digit
    with pytest.raises(ValueError):
        byte_tokens_to_float64(["<h0>"] * 8)   # retired NIBBLE token
