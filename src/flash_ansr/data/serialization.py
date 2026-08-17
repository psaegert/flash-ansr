"""Constant serialization for training sequences (the v24 ``constant_representation`` gate).

``'v23'`` (the default) keeps today's behavior byte-identical: ``<constant>`` placeholder
tokens stay in the sequence and the fitted values ride out-of-band on the parallel numeric
channel (``input_num``), injected downstream by
:func:`flash_ansr.utils.numeric.build_numeric_sequence`.

``'ieee754_mixed'`` replaces each constant occurrence -- per constant independently with
probability ``expanded_probability`` (default 0.5, seeded RNG) -- with one of two forms:

* EXPANDED (34 tokens): ``<ieee754>`` + 32 bit tokens + ``</ieee754>``; the numeric channel is
  NaN across the whole span (the value lives in the bits).
* COMPACT (1 token): the existing ``<float>`` special token, with the float32 value on the
  numeric channel at that position. The label at positions whose TARGET is ``<float>`` is
  loss-masked downstream (:func:`flash_ansr.data.collate.mask_float_targets`): the model must
  never be trained to emit ``<float>``; the token exists so histories can carry compacted
  constants (at inference the model emits bits and the pipeline compacts them).

Non-finite constants raise at serialization time -- assert, don't assume.
"""
import math
import re
from typing import Sequence

import numpy as np

from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_START_TOKEN,
    wrap_float32,
)

#: The compact-constant token. Deliberately the EXISTING ``<float>`` special (it already
#: carries a value on the numeric channel in the prompt-serialization path); no new token.
COMPACT_CONSTANT_TOKEN = "<float>"

#: Legal values of the ``constant_representation`` config key.
CONSTANT_REPRESENTATION_V23 = "v23"
CONSTANT_REPRESENTATION_IEEE754_MIXED = "ieee754_mixed"
CONSTANT_REPRESENTATIONS = (CONSTANT_REPRESENTATION_V23, CONSTANT_REPRESENTATION_IEEE754_MIXED)

_INDEXED_CONSTANT_PATTERN = re.compile(r"C_\d+$")


def _is_constant_placeholder(token: str) -> bool:
    return token == "<constant>" or bool(_INDEXED_CONSTANT_PATTERN.match(token))


def serialize_constant_tokens(
    tokens: Sequence[str],
    constants: Sequence[float],
    *,
    representation: str = CONSTANT_REPRESENTATION_V23,
    rng: np.random.Generator | None = None,
    expanded_probability: float = 0.5,
) -> tuple[list[str], list[float]]:
    """Serialize the constant placeholders of a token sequence under ``representation``.

    Parameters
    ----------
    tokens : Sequence[str]
        The expression tokens; constants are the ``<constant>`` / ``C_<i>`` placeholders.
    constants : Sequence[float]
        The fitted values, one per placeholder occurrence, in order.
    representation : str, default 'v23'
        ``'v23'`` returns the tokens unchanged (values stay out-of-band);
        ``'ieee754_mixed'`` applies the per-constant 50/50 expanded/compact mixing.
    rng : numpy.random.Generator, optional
        The (seeded) RNG driving the per-constant coin flips. Required for
        ``'ieee754_mixed'``.
    expanded_probability : float, default 0.5
        Probability that a constant is serialized in EXPANDED form.

    Returns
    -------
    tuple[list[str], list[float]]
        ``(serialized_tokens, numeric_values)`` where ``numeric_values`` is aligned per
        OUTPUT token: NaN everywhere except at compact ``<float>`` positions, which carry
        the float32 constant value.

    Raises
    ------
    ValueError
        On an unknown representation, a missing RNG for the mixed representation, a
        placeholder/constants count mismatch, or a non-finite constant.
    """
    if representation not in CONSTANT_REPRESENTATIONS:
        raise ValueError(
            f"Unknown constant_representation {representation!r}; expected one of {CONSTANT_REPRESENTATIONS}."
        )

    tokens = list(tokens)

    if representation == CONSTANT_REPRESENTATION_V23:
        # Byte-identical passthrough: placeholders stay, values stay out-of-band.
        return tokens, [float("nan")] * len(tokens)

    if rng is None:
        raise ValueError("The 'ieee754_mixed' representation requires a (seeded) rng.")

    values = [float(value) for value in constants]
    n_placeholders = sum(1 for token in tokens if _is_constant_placeholder(token))
    if n_placeholders != len(values):
        raise ValueError(
            f"Constant count mismatch: {n_placeholders} placeholder(s) in {tokens!r} but "
            f"{len(values)} value(s)."
        )

    serialized: list[str] = []
    numeric: list[float] = []
    constant_index = 0

    for token in tokens:
        if not _is_constant_placeholder(token):
            serialized.append(token)
            numeric.append(float("nan"))
            continue

        value = values[constant_index]
        constant_index += 1
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite constant {value!r} at placeholder {constant_index - 1}: the data "
                f"generator must never emit inf/nan constants."
            )
        value32 = float(np.float32(value))

        if rng.random() < expanded_probability:
            span = wrap_float32(value32)  # raises if the value overflowed float32 to non-finite
            serialized.extend(span)
            numeric.extend([float("nan")] * len(span))
        else:
            if not math.isfinite(value32):
                raise ValueError(
                    f"Constant {value!r} overflows float32 ({value32!r}); non-finite constants "
                    f"must never be serialized."
                )
            serialized.append(COMPACT_CONSTANT_TOKEN)
            numeric.append(value32)

    return serialized, numeric


def find_ieee754_spans(token_ids: Sequence[int], start_id: int, end_id: int) -> list[tuple[int, int]]:
    """Locate the ``(start, end)`` index pairs (inclusive) of ``<ieee754>`` ... ``</ieee754>`` spans."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, token_id in enumerate(token_ids):
        if token_id == start_id:
            start = index
        elif token_id == end_id and start is not None:
            spans.append((start, index))
            start = None
    return spans


def truncation_cuts_ieee754_span(
    token_ids: Sequence[int],
    max_seq_len: int,
    start_id: int,
    end_id: int,
) -> bool:
    """Whether truncating ``token_ids`` to ``max_seq_len`` would cut inside an ``<ieee754>`` span.

    Truncation keeps indices ``0 .. max_seq_len - 2`` and overwrites index
    ``max_seq_len - 1`` with ``<eos>`` (the streaming worker's rule). A span is cut when it
    starts within the kept content but does not complete within it. Spans that survive whole,
    or are dropped whole, are fine.
    """
    if len(token_ids) <= max_seq_len:
        return False
    last_kept = max_seq_len - 2  # index max_seq_len - 1 becomes <eos>
    for start, end in find_ieee754_spans(token_ids, start_id, end_id):
        if start <= last_kept < end:
            return True
    return False
