"""Token-level IEEE-754 binary64 codec for the v25 constants representation.

A serialized constant occupies 10 tokens: ``<ieee754>``, the 8 BYTES of the value's
IEEE-754 binary64 encoding as dedicated byte tokens ``<b00>``..``<bff>``, and ``</ieee754>``.

BYTE ORDER IS BIG-ENDIAN, most-significant byte first: the tokens read off the big-endian
spelling of the 64-bit pattern, so ``tokens[0]`` carries bits ``63..56`` (the sign bit and
the top seven exponent bits) and ``tokens[7]`` carries bits ``7..0`` (the last eight
mantissa bits). ``-2.0`` (pattern ``0xc000000000000000``) therefore serializes as
``<bc0> <b00> <b00> <b00> <b00> <b00> <b00> <b00>``.

Owner ruling 2026-08-27: "f64 and bytes, not nibbles." The 8 hex-nibble tokens over a
16-symbol alphabet this replaces are retired, and so is the 32-bit value semantics beneath
them. The SPAN WIDTH IS UNCHANGED -- 8 nibbles of binary32 and 8 bytes of binary64 are both
8 content positions -- so :data:`IEEE754_SPAN_LENGTH` and every index computed from it
carry over untouched. What changed is the alphabet (16 -> 256 symbols) and the value
semantics (float32 -> float64).

Byte tokens are spelled with TWO hex digits, always (Q3). One digit would re-emit
``<b0>``..``<bf>`` for values 0..15 -- the RETIRED bit tokens -- so a stale vocabulary would
satisfy a presence check while meaning something else entirely.

This is the exact (``struct``-based) sibling of the tensor-level
:func:`flash_ansr.model.pre_encoder.float_to_ieee754_bits` used for embeddings (which stays
BIT-valued -- it is a numeric pre-embedding, not a token serialization). Non-finite values
refuse to serialize: the data generator must never emit them, and this codec asserts that
instead of assuming it. There is no longer an overflow refusal -- a float64 magnitude has no
wider format to overflow, which is the point of the migration.
"""
import math
import struct
from typing import Sequence

import numpy as np

#: Opening tag of a serialized-constant span.
IEEE754_START_TOKEN = "<ieee754>"
#: Closing tag of a serialized-constant span.
IEEE754_END_TOKEN = "</ieee754>"

#: Number of bytes in a serialized constant (64 bits / 8).
IEEE754_N_BYTES = 8
#: Size of the per-position alphabet inside a span.
IEEE754_N_BYTE_SYMBOLS = 256

#: Dedicated byte tokens, indexed BY BYTE VALUE (``BYTE_TOKENS[10] == '<b0a>'``)
#: -- deliberately NOT the numeric literals they spell, and always two hex digits.
BYTE_TOKENS = tuple(f"<b{value:02x}>" for value in range(IEEE754_N_BYTE_SYMBOLS))

#: All special tokens the serialized form introduces.
IEEE754_SPECIAL_TOKENS = (IEEE754_START_TOKEN, IEEE754_END_TOKEN, *BYTE_TOKENS)

#: Total width of a serialized-constant span including both tags. UNCHANGED across the
#: nibble -> byte migration: 8 content positions either way.
IEEE754_SPAN_LENGTH = IEEE754_N_BYTES + 2

#: The lane name this codec serves, declared by ``constants_format`` in tokenizer.yaml.
#: A vocabulary from the retired nibble lane shares no content token with this one, so the
#: mismatch is worth naming rather than discovering as an out-of-vocabulary error.
CONSTANTS_FORMAT = "ieee754_bytes_f64"

#: Reverse map, byte token -> byte value.
_BYTE_VALUES = {token: value for value, token in enumerate(BYTE_TOKENS)}


def float64_to_byte_tokens(value: float) -> list[str]:
    """Encode ``value`` as its 8 IEEE-754 binary64 byte tokens, big-endian.

    Parameters
    ----------
    value : float
        The value to encode, at full float64 precision.

    Returns
    -------
    list[str]
        8 tokens drawn from :data:`BYTE_TOKENS`, most-significant byte first (bits
        ``63..56`` down to bits ``7..0``).

    Raises
    ------
    ValueError
        If ``value`` is non-finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite constant {value!r} to IEEE-754 byte tokens.")

    return [BYTE_TOKENS[byte] for byte in struct.pack(">d", float(value))]


def float64_to_byte_values(values: "np.ndarray") -> "np.ndarray":
    """Vectorized encoder: ``(...)`` float array -> ``(..., 8)`` uint8 byte VALUES, big-endian.

    The array-level sibling of :func:`float64_to_byte_tokens`, sharing this module's single
    definition of the bit layout. It returns byte VALUES (``0..255``), not tokens.

    Unlike the scalar token encoder this does NOT refuse non-finite input. That guard
    protects a serialization contract into the token stream; inf/nan have perfectly
    well-defined bit patterns. Callers mask invalid points by their own validity mask.

    Parameters
    ----------
    values : np.ndarray
        Any shape. Cast to ``float64`` before decomposition.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``values.shape + (8,)``, most-significant byte first, so
        ``[..., 0]`` carries bits ``63..56`` and ``[..., 7]`` carries bits ``7..0``.
    """
    patterns = np.asarray(values, dtype=np.float64).view(np.uint64)
    shifts = np.arange(8 * (IEEE754_N_BYTES - 1), -1, -8, dtype=np.uint64)
    return ((patterns[..., None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)


def byte_values_to_float64(byte_values: "np.ndarray") -> "np.ndarray":
    """Vectorized decoder: ``(..., 8)`` byte values -> ``(...)`` float64. Inverse of the above."""
    byte_values = np.asarray(byte_values, dtype=np.uint64)
    if byte_values.shape[-1] != IEEE754_N_BYTES:
        raise ValueError(
            f"Expected a trailing axis of {IEEE754_N_BYTES} bytes, got {byte_values.shape[-1]}.")
    shifts = np.arange(8 * (IEEE754_N_BYTES - 1), -1, -8, dtype=np.uint64)
    return (((byte_values & np.uint64(0xFF)) << shifts).sum(axis=-1, dtype=np.uint64)).view(np.float64)


def byte_tokens_to_float64(tokens: Sequence[str]) -> float:
    """Decode 8 byte tokens (big-endian) back into the ``float64`` value they encode.

    Parameters
    ----------
    tokens : Sequence[str]
        Exactly 8 tokens drawn from :data:`BYTE_TOKENS`, most-significant byte first.

    Returns
    -------
    float
        The decoded ``float64`` value.

    Raises
    ------
    ValueError
        If the sequence is not exactly 8 valid byte tokens.
    """
    tokens = list(tokens)
    if len(tokens) != IEEE754_N_BYTES:
        raise ValueError(f"Expected exactly {IEEE754_N_BYTES} byte tokens, got {len(tokens)}.")

    pattern = 0
    for token in tokens:
        byte = _BYTE_VALUES.get(token)
        if byte is None:
            raise ValueError(
                f"Invalid byte token {token!r}: expected one of {list(BYTE_TOKENS)}."
            )
        pattern = (pattern << 8) | byte

    (value,) = struct.unpack(">d", struct.pack(">Q", pattern))
    return value


def wrap_float64(value: float) -> list[str]:
    """Serialize ``value`` as a full constant span: ``<ieee754>`` + 8 bytes + ``</ieee754>``."""
    return [IEEE754_START_TOKEN, *float64_to_byte_tokens(value), IEEE754_END_TOKEN]


def unwrap_ieee754_span(tokens: Sequence[str]) -> float:
    """Decode a full 10-token constant span back into its ``float64`` value.

    Raises
    ------
    ValueError
        If the span is not exactly ``<ieee754>`` + 8 byte tokens + ``</ieee754>``.
    """
    tokens = list(tokens)
    if len(tokens) != IEEE754_SPAN_LENGTH:
        raise ValueError(f"Expected a {IEEE754_SPAN_LENGTH}-token span, got {len(tokens)} tokens.")
    if tokens[0] != IEEE754_START_TOKEN or tokens[-1] != IEEE754_END_TOKEN:
        raise ValueError(
            f"Malformed span: expected {IEEE754_START_TOKEN!r} ... {IEEE754_END_TOKEN!r}, "
            f"got {tokens[0]!r} ... {tokens[-1]!r}."
        )
    return byte_tokens_to_float64(tokens[1:-1])
