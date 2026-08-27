"""Token-level IEEE-754 binary32 codec for the v24 ``ieee754_mixed`` constants representation.

A constant serialized in EXPANDED form occupies 10 tokens: ``<ieee754>``, the 8 HEX NIBBLES
of the value's IEEE-754 binary32 encoding as dedicated nibble tokens ``<h0>``..``<hf>``, and
``</ieee754>``.

NIBBLE ORDER IS BIG-ENDIAN, most-significant nibble first: the tokens read off the
big-endian hex spelling of the 32-bit pattern, so ``tokens[0]`` carries bits ``31..28``
(the sign bit and the top three exponent bits) and ``tokens[7]`` carries bits ``3..0`` (the
last four mantissa bits). ``-2.0`` (pattern ``0xc0000000``) therefore serializes as
``<hc> <h0> <h0> <h0> <h0> <h0> <h0> <h0>``.

Owner ruling 2026-08-18: "All numbers will be represented by their ieee754 representation
[...] And I think we could also format the ieee754 bit strings as hexadecimal. Same
representative power, fewer tokens." The 32 ``<b0>``/``<b1>`` bit tokens this replaces are
retired: same float32 value semantics, a 16-symbol alphabet instead of 2, and 4x fewer
autoregressive steps per constant.

This is the exact (``struct``-based) sibling of the tensor-level
:func:`flash_ansr.model.pre_encoder.float32_to_ieee754_bits` used for embeddings (which
stays BIT-valued -- it is a numeric pre-embedding, not a token serialization); values follow
numpy ``float32`` semantics. Non-finite values (inf/nan, including float64 magnitudes that
overflow float32) refuse to serialize: the data generator must never emit them, and this
codec asserts that instead of assuming it.
"""
import math
import struct
from typing import Sequence

import numpy as np

#: Opening tag of an expanded-constant span.
IEEE754_START_TOKEN = "<ieee754>"
#: Closing tag of an expanded-constant span.
IEEE754_END_TOKEN = "</ieee754>"

#: Number of hex nibbles in an expanded constant (32 bits / 4).
IEEE754_N_NIBBLES = 8
#: Size of the per-position alphabet inside a span.
IEEE754_N_NIBBLE_SYMBOLS = 16

#: Dedicated hex-nibble tokens, indexed BY NIBBLE VALUE (``NIBBLE_TOKENS[10] == '<ha>'``)
#: -- deliberately NOT the numeric literals ``0``..``9``/``a``..``f`` they spell.
NIBBLE_TOKENS = tuple(f"<h{value:x}>" for value in range(IEEE754_N_NIBBLE_SYMBOLS))

#: All special tokens the expanded form introduces.
IEEE754_SPECIAL_TOKENS = (IEEE754_START_TOKEN, IEEE754_END_TOKEN, *NIBBLE_TOKENS)

#: Total width of an expanded-constant span including both tags.
IEEE754_SPAN_LENGTH = IEEE754_N_NIBBLES + 2

#: Reverse map, nibble token -> nibble value.
_NIBBLE_VALUES = {token: value for value, token in enumerate(NIBBLE_TOKENS)}


def float32_to_nibble_tokens(value: float) -> list[str]:
    """Encode ``value`` as its 8 IEEE-754 binary32 hex-nibble tokens, big-endian.

    Parameters
    ----------
    value : float
        The value to encode. It is converted to ``float32`` (numpy semantics) first.

    Returns
    -------
    list[str]
        8 tokens drawn from :data:`NIBBLE_TOKENS`, most-significant nibble first (bits
        ``31..28`` down to bits ``3..0``).

    Raises
    ------
    ValueError
        If ``value`` is non-finite, or overflows to a non-finite ``float32``.
    """
    if not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite constant {value!r} to IEEE-754 nibble tokens.")

    value32 = float(np.float32(value))
    if not math.isfinite(value32):
        raise ValueError(
            f"Constant {value!r} overflows float32 ({value32!r}); non-finite constants must never "
            f"be serialized."
        )

    return [NIBBLE_TOKENS[nibble]
            for nibble in float32_to_nibble_values(np.float32(value32))]


def float32_to_nibble_values(values: "np.ndarray") -> "np.ndarray":
    """Vectorized encoder: ``(...)`` float array -> ``(..., 8)`` uint8 nibble VALUES, big-endian.

    The array-level sibling of :func:`float32_to_nibble_tokens`, sharing this module's single
    definition of the bit layout. It returns nibble VALUES (``0..15``), not tokens: per-point
    heads own a 16-way softmax per nibble position and never touch the decoder vocabulary.

    Unlike the scalar token encoder this does NOT refuse non-finite input. That guard protects
    a serialization contract into the token stream; a head target is not serialized, and inf/nan
    have perfectly well-defined bit patterns. Callers mask invalid points by their own validity
    mask (for the residual head, ``data_attn_mask``).

    Parameters
    ----------
    values : np.ndarray
        Any shape. Cast to ``float32`` (numpy semantics) before decomposition.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``values.shape + (8,)``, most-significant nibble first, so
        ``[..., 0]`` carries bits ``31..28`` and ``[..., 7]`` carries bits ``3..0``.
    """
    patterns = np.asarray(values, dtype=np.float32).view(np.uint32)
    shifts = np.arange(4 * (IEEE754_N_NIBBLES - 1), -1, -4, dtype=np.uint32)
    return ((patterns[..., None] >> shifts) & np.uint32(0xF)).astype(np.uint8)


def nibble_values_to_float32(nibbles: "np.ndarray") -> "np.ndarray":
    """Vectorized decoder: ``(..., 8)`` nibble values -> ``(...)`` float32. Inverse of the above."""
    nibbles = np.asarray(nibbles, dtype=np.uint32)
    if nibbles.shape[-1] != IEEE754_N_NIBBLES:
        raise ValueError(
            f"Expected a trailing axis of {IEEE754_N_NIBBLES} nibbles, got {nibbles.shape[-1]}.")
    shifts = np.arange(4 * (IEEE754_N_NIBBLES - 1), -1, -4, dtype=np.uint32)
    return (((nibbles & np.uint32(0xF)) << shifts).sum(axis=-1, dtype=np.uint32)).view(np.float32)


def nibble_tokens_to_float32(tokens: Sequence[str]) -> float:
    """Decode 8 hex-nibble tokens (big-endian) back into the ``float32`` value they encode.

    Parameters
    ----------
    tokens : Sequence[str]
        Exactly 8 tokens drawn from :data:`NIBBLE_TOKENS`, most-significant nibble first.

    Returns
    -------
    float
        The decoded ``float32`` value (as a Python float, exactly representable).

    Raises
    ------
    ValueError
        If the sequence is not exactly 8 valid nibble tokens.
    """
    tokens = list(tokens)
    if len(tokens) != IEEE754_N_NIBBLES:
        raise ValueError(f"Expected exactly {IEEE754_N_NIBBLES} nibble tokens, got {len(tokens)}.")

    pattern = 0
    for token in tokens:
        nibble = _NIBBLE_VALUES.get(token)
        if nibble is None:
            raise ValueError(
                f"Invalid nibble token {token!r}: expected one of {list(NIBBLE_TOKENS)}."
            )
        pattern = (pattern << 4) | nibble

    (value,) = struct.unpack(">f", struct.pack(">I", pattern))
    return value


def wrap_float32(value: float) -> list[str]:
    """Serialize ``value`` as a full expanded-constant span: ``<ieee754>`` + 8 nibbles + ``</ieee754>``."""
    return [IEEE754_START_TOKEN, *float32_to_nibble_tokens(value), IEEE754_END_TOKEN]


def unwrap_ieee754_span(tokens: Sequence[str]) -> float:
    """Decode a full 10-token expanded-constant span back into its ``float32`` value.

    Raises
    ------
    ValueError
        If the span is not exactly ``<ieee754>`` + 8 nibble tokens + ``</ieee754>``.
    """
    tokens = list(tokens)
    if len(tokens) != IEEE754_SPAN_LENGTH:
        raise ValueError(f"Expected a {IEEE754_SPAN_LENGTH}-token span, got {len(tokens)} tokens.")
    if tokens[0] != IEEE754_START_TOKEN or tokens[-1] != IEEE754_END_TOKEN:
        raise ValueError(
            f"Malformed span: expected {IEEE754_START_TOKEN!r} ... {IEEE754_END_TOKEN!r}, "
            f"got {tokens[0]!r} ... {tokens[-1]!r}."
        )
    return nibble_tokens_to_float32(tokens[1:-1])
