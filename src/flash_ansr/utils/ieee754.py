"""Token-level IEEE-754 binary32 codec for the v24 ``ieee754_mixed`` constants representation.

A constant serialized in EXPANDED form occupies 34 tokens: ``<ieee754>``, the 32 bits of the
value's IEEE-754 binary32 encoding MSB-first (sign bit, 8 exponent bits, 23 mantissa bits) as
dedicated bit tokens ``<b0>``/``<b1>``, and ``</ieee754>``.

This is the exact (``struct``-based) sibling of the tensor-level
:func:`flash_ansr.model.pre_encoder.float32_to_ieee754_bits` used for embeddings; values follow
numpy ``float32`` semantics. Non-finite values (inf/nan, including float64 magnitudes that
overflow float32) refuse to serialize: the data generator must never emit them, and this codec
asserts that instead of assuming it.
"""
import math
import struct
from typing import Sequence

import numpy as np

#: Opening tag of an expanded-constant span.
IEEE754_START_TOKEN = "<ieee754>"
#: Closing tag of an expanded-constant span.
IEEE754_END_TOKEN = "</ieee754>"
#: Dedicated bit tokens -- deliberately NOT the numeric literals ``0``/``1``.
BIT_ZERO_TOKEN = "<b0>"
BIT_ONE_TOKEN = "<b1>"

#: All special tokens the expanded form introduces.
IEEE754_SPECIAL_TOKENS = (IEEE754_START_TOKEN, IEEE754_END_TOKEN, BIT_ZERO_TOKEN, BIT_ONE_TOKEN)

#: Number of bit tokens in an expanded constant.
IEEE754_N_BITS = 32
#: Total width of an expanded-constant span including both tags.
IEEE754_SPAN_LENGTH = IEEE754_N_BITS + 2


def float32_to_bit_tokens(value: float) -> list[str]:
    """Encode ``value`` as its 32 IEEE-754 binary32 bit tokens, MSB-first.

    Parameters
    ----------
    value : float
        The value to encode. It is converted to ``float32`` (numpy semantics) first.

    Returns
    -------
    list[str]
        32 tokens, each :data:`BIT_ZERO_TOKEN` or :data:`BIT_ONE_TOKEN`, ordered
        ``[sign, exp[7:0], mantissa[22:0]]``.

    Raises
    ------
    ValueError
        If ``value`` is non-finite, or overflows to a non-finite ``float32``.
    """
    if not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite constant {value!r} to IEEE-754 bit tokens.")

    value32 = float(np.float32(value))
    if not math.isfinite(value32):
        raise ValueError(
            f"Constant {value!r} overflows float32 ({value32!r}); non-finite constants must never "
            f"be serialized."
        )

    (pattern,) = struct.unpack(">I", struct.pack(">f", value32))
    return [BIT_ONE_TOKEN if (pattern >> shift) & 1 else BIT_ZERO_TOKEN for shift in range(31, -1, -1)]


def bit_tokens_to_float32(tokens: Sequence[str]) -> float:
    """Decode 32 bit tokens (MSB-first) back into the ``float32`` value they encode.

    Parameters
    ----------
    tokens : Sequence[str]
        Exactly 32 tokens, each :data:`BIT_ZERO_TOKEN` or :data:`BIT_ONE_TOKEN`.

    Returns
    -------
    float
        The decoded ``float32`` value (as a Python float, exactly representable).

    Raises
    ------
    ValueError
        If the sequence is not exactly 32 valid bit tokens.
    """
    tokens = list(tokens)
    if len(tokens) != IEEE754_N_BITS:
        raise ValueError(f"Expected exactly {IEEE754_N_BITS} bit tokens, got {len(tokens)}.")

    pattern = 0
    for token in tokens:
        if token == BIT_ONE_TOKEN:
            bit = 1
        elif token == BIT_ZERO_TOKEN:
            bit = 0
        else:
            raise ValueError(
                f"Invalid bit token {token!r}: expected {BIT_ZERO_TOKEN!r} or {BIT_ONE_TOKEN!r}."
            )
        pattern = (pattern << 1) | bit

    (value,) = struct.unpack(">f", struct.pack(">I", pattern))
    return value


def wrap_float32(value: float) -> list[str]:
    """Serialize ``value`` as a full expanded-constant span: ``<ieee754>`` + 32 bits + ``</ieee754>``."""
    return [IEEE754_START_TOKEN, *float32_to_bit_tokens(value), IEEE754_END_TOKEN]


def unwrap_ieee754_span(tokens: Sequence[str]) -> float:
    """Decode a full 34-token expanded-constant span back into its ``float32`` value.

    Raises
    ------
    ValueError
        If the span is not exactly ``<ieee754>`` + 32 bit tokens + ``</ieee754>``.
    """
    tokens = list(tokens)
    if len(tokens) != IEEE754_SPAN_LENGTH:
        raise ValueError(f"Expected a {IEEE754_SPAN_LENGTH}-token span, got {len(tokens)} tokens.")
    if tokens[0] != IEEE754_START_TOKEN or tokens[-1] != IEEE754_END_TOKEN:
        raise ValueError(
            f"Malformed span: expected {IEEE754_START_TOKEN!r} ... {IEEE754_END_TOKEN!r}, "
            f"got {tokens[0]!r} ... {tokens[-1]!r}."
        )
    return bit_tokens_to_float32(tokens[1:-1])
