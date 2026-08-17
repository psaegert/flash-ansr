"""T1 — IEEE-754 binary32 token codec round-trip (v24 constants representation contract).

float32 -> 32 bit tokens -> float32 must be the identity for every finite float32,
including denormals and extremes. Non-finite values must raise at serialization time
(assert, don't assume).
"""
import math
import struct

import numpy as np
import pytest

from flash_ansr.utils.ieee754 import (
    BIT_ONE_TOKEN,
    BIT_ZERO_TOKEN,
    IEEE754_END_TOKEN,
    IEEE754_START_TOKEN,
    bit_tokens_to_float32,
    float32_to_bit_tokens,
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


def test_t1_roundtrip_extremes_and_denormals() -> None:
    for value in EXTREME_VALUES:
        tokens = float32_to_bit_tokens(value)
        assert len(tokens) == 32
        assert all(token in (BIT_ZERO_TOKEN, BIT_ONE_TOKEN) for token in tokens)
        recovered = bit_tokens_to_float32(tokens)
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
        tokens = float32_to_bit_tokens(value)
        recovered = bit_tokens_to_float32(tokens)
        assert struct.pack(">f", recovered) == int(pattern).to_bytes(4, "big")
        checked += 1
    assert checked > 9_000  # the sample must actually be large


def test_t1_bit_order_is_msb_first_sign_exponent_mantissa() -> None:
    # -2.0 = sign 1, exponent 1000_0000, mantissa 0: an asymmetric witness for bit order.
    tokens = float32_to_bit_tokens(-2.0)
    expected = "1" + "10000000" + "0" * 23
    assert "".join("1" if token == BIT_ONE_TOKEN else "0" for token in tokens) == expected


def test_t1_nonfinite_serialization_raises() -> None:
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            float32_to_bit_tokens(value)
    # float64 value that overflows float32 must raise too, not silently become inf.
    with pytest.raises(ValueError):
        float32_to_bit_tokens(1e39)


def test_t1_span_wrap_unwrap() -> None:
    span = wrap_float32(1.5)
    assert len(span) == 34
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
        unwrap_ieee754_span([IEEE754_START_TOKEN, BIT_ZERO_TOKEN, IEEE754_END_TOKEN])  # wrong width
    with pytest.raises(ValueError):
        bit_tokens_to_float32(["0"] * 32)  # literal '0' is not a bit token
