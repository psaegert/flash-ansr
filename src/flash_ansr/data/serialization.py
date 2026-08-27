"""Constant serialization for training sequences.

Every constant occurrence becomes an ``<ieee754>`` span: the open tag, the 8 BYTES of
its IEEE-754 binary64 encoding, the close tag. The numeric channel is NaN across the whole
span -- the value lives in the bytes.

There is no second form. Under the owner's 2026-08-27 ruling a number the model PREDICTS
is spelled in bits everywhere, and the constants of an expression are predicted. The
compact ``<float>`` token survives for the numbers a CALLER supplies -- ``predict_y``'s x
coordinates, a prompted complexity -- and is forbidden inside an expression, so a history
the model conditions on never carries a spelling it was not trained to continue from.

Non-finite constants raise at serialization time -- assert, don't assume.
"""
import math
import re
from typing import Sequence

import numpy as np

from flash_ansr.utils.ieee754 import (
    IEEE754_N_BYTES,
    IEEE754_SPAN_LENGTH,
    BYTE_TOKENS,
    byte_tokens_to_float64,
    wrap_float64,
)

#: The compact-constant token. Deliberately the EXISTING ``<float>`` special (it already
#: carries a value on the numeric channel in the prompt-serialization path); no new token.
COMPACT_CONSTANT_TOKEN = "<float>"

# v24 task-block grammar (owner ruling 2026-08-24): the harness owns the structure --
# every opener/selector below is force-fed and loss-masked; the model owns content
# content bytes and closing tags. <complexity>/<float> already exist in the vocabulary;
# the predict_y tokens are new with this feature.
COMPLEXITY_START_TOKEN = "<complexity>"
COMPLEXITY_END_TOKEN = "</complexity>"
PREDICT_Y_START_TOKEN = "<predict_y>"
PREDICT_Y_END_TOKEN = "</predict_y>"
POINT_START_TOKEN = "<point>"
POINT_END_TOKEN = "</point>"
PREDICT_Y_TOKENS = (PREDICT_Y_START_TOKEN, PREDICT_Y_END_TOKEN, POINT_START_TOKEN, POINT_END_TOKEN)
# The BOUNDARY (owner rulings 2026-08-24, refined 2026-08-27). A harness-inserted marker,
# uttered at most once, that hands the pen over: everything BEFORE it is given -- fixed,
# compact, force-fed and loss-masked -- and everything after it is the model's own, spelled
# in ieee754 bits and supervised (opener, content and closers alike). A property stated
# before the flag may not be hypothesized after it; the decode grammar enforces that as
# at-most-once per opener (decoding/constrained.py). The flag is never supervised: only the
# harness may put the model into hypothesis mode.
#
# It is a marker of its own, NOT part of any block, so one flag licenses the whole run of
# property blocks that follows it. The query/answer blocks (<predict_y>, and <predict_
# residual> when it lands) are EXEMPT from the positional rule: inside them the loss mask
# decides, so their force-fed x coordinates stay compact wherever the block sits.
HYPOTHESIS_TOKEN = "<hypothesize>"

#: Promptable-mask flags (owner ruling 2026-08-24): harness-owned, never supervised.
#: Named after simplipy's masking policies. Absence of a flag means the unmasked
#: (constants-spelled) target -- the 90% default -- so the common path carries no
#: new token.
MASK_ALL_TOKEN = "<mask_all>"
MASK_FITTABLE_TOKEN = "<mask_fittable>"
MASK_MODE_TOKENS = {"all": MASK_ALL_TOKEN, "fittable": MASK_FITTABLE_TOKEN}

#: The valueless placeholder the model reads and (under a flag) emits for a masked
#: constant: simplipy's native `<constant>` (owner ruling 2026-08-24, reverting the
#: earlier `<masked_constant>` spelling -- simplipy depends on `<constant>`, so
#: emitted skeletons feed collection/to_skeleton/refit without a rename shim).
#: Inside the worker the serializer's None-entry support tells value slots from
#: placeholders; in finished targets every remaining `<constant>` IS a placeholder.
MASKED_CONSTANT_TOKEN = "<constant>"

#: The constant-infilling block (owner ruling 2026-08-24): appended after
#: `</expression>`, one `<ieee754>` span per `<masked_constant>` placeholder, in
#: positional order. Named in the `<predict_y>` family.
PREDICT_CONSTANTS_START_TOKEN = "<predict_constants>"
PREDICT_CONSTANTS_END_TOKEN = "</predict_constants>"
PREDICT_CONSTANTS_TOKENS = (PREDICT_CONSTANTS_START_TOKEN, PREDICT_CONSTANTS_END_TOKEN)
COMPLEXITY_TOKENS = (COMPLEXITY_START_TOKEN, COMPLEXITY_END_TOKEN)

#: The tagged canonical dialect's delimiter set (verified against a live generation-2
#: engine by the v24 template tests). ``<sub>`` and ``<div>`` are role markers inside
#: their enclosing bag, not paired tags.
TAGGED_DELIMITER_TOKENS = ("<add>", "</add>", "<sub>", "<mul>", "</mul>", "<div>")

#: v24 target-dialect gate: 'explicit' (default) targets the binary-prefix expression
#: exactly as today; 'tagged' targets the engine's TAGGED CANONICAL output (contract
#: A3) -- simplify run IN the tagged dialect per problem, every numeric literal riding
#: the ieee754 constants format, np.pi/np.e staying symbolic.
TARGET_DIALECT_EXPLICIT = "explicit"
TARGET_DIALECT_TAGGED = "tagged"
TARGET_DIALECTS = (TARGET_DIALECT_EXPLICIT, TARGET_DIALECT_TAGGED)

_INDEXED_CONSTANT_PATTERN = re.compile(r"C_\d+$")


def _is_constant_placeholder(token: str) -> bool:
    return token == "<constant>" or bool(_INDEXED_CONSTANT_PATTERN.match(token))


def serialize_constant_tokens(
    tokens: Sequence[str],
    constants: "Sequence[float] | np.ndarray",
) -> tuple[list[str], list[float]]:
    """Serialize the constant placeholders of a token sequence into ``<ieee754>`` spans.

    Parameters
    ----------
    tokens : Sequence[str]
        The expression tokens; constants are the ``<constant>`` / ``C_<i>`` placeholders.
    constants : Sequence[float or None]
        The fitted values, one per placeholder occurrence, in order. A ``None`` entry
        KEEPS that occurrence as a literal ``<constant>`` placeholder (nan on the
        numeric channel) -- the promptable-mask target format, where the policy's
        placeholders survive serialization while kept structural literals still ride
        the ieee754 spelling.

    Returns
    -------
    tuple[list[str], list[float]]
        ``(serialized_tokens, numeric_values)`` where ``numeric_values`` is aligned per
        OUTPUT token and is NaN throughout: no expression token carries a value on the
        numeric channel.

    Raises
    ------
    ValueError
        On a placeholder/constants count mismatch, or a non-finite constant.
    """
    tokens = list(tokens)

    values = [None if value is None else float(value) for value in constants]
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
        if value is None:
            # The policy's placeholder: it IS the target token. No span.
            serialized.append(token)
            numeric.append(float("nan"))
            continue
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite constant {value!r} at placeholder {constant_index - 1}: the data "
                f"generator must never emit inf/nan constants."
            )
        # No narrowing: the value is serialized at the precision it was fitted at.
        # Until S4 this read float(np.float32(value)) and could refuse a finite float64
        # for overflowing binary32 -- the refusal the migration removes.
        span = wrap_float64(float(value))
        serialized.extend(span)
        numeric.extend([float("nan")] * len(span))

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


def replace_ieee754_spans_with_constants(
    token_ids: Sequence[int],
    *,
    start_id: int,
    end_id: int,
    byte_ids: Sequence[int],
    constant_id: int,
) -> tuple[list[int], list[float] | None]:
    """Map ``<ieee754>`` spans to ``<constant>`` slots + their float64 values (T11).

    The DESERIALIZATION half of the refiner handshake: each well-formed 10-token span in a
    generated beam collapses to one ``constant_id`` token, and the decoded values (exact,
    via the token codec -- no decimal round-trip) become the refiner's verbatim ``p0`` in
    order of appearance.

    Parameters
    ----------
    byte_ids : Sequence[int]
        The 256 byte token ids IN BYTE-VALUE ORDER (``byte_ids[10]`` is ``<b0a>``),
        i.e. ``[int(tokenizer[token]) for token in BYTE_TOKENS]``.

    Returns
    -------
    tuple[list[int], list[float] | None]
        ``(mapped_ids, values)``. ``values`` is the per-span float64 list ONLY when the
        init is sound: at least one span, every span well-formed, every value finite, and
        no pre-existing ``constant_id`` in the input (a bare placeholder -- e.g.
        constantified sugar -- would break the slot alignment; the skeleton is still
        mapped, the init is withheld and refinement falls back to its own seed). Any MALFORMED
        span returns the input unchanged with ``None`` (the beam is not a v24 carrier;
        downstream validity checks dispose of it as today).
    """
    if len(byte_ids) != len(BYTE_TOKENS):
        raise ValueError(
            f"Expected {len(BYTE_TOKENS)} byte ids in byte-value order, got {len(byte_ids)}."
        )
    id_to_byte = {token_id: BYTE_TOKENS[value] for value, token_id in enumerate(byte_ids)}

    ids = list(token_ids)
    mapped: list[int] = []
    values: list[float] = []
    index = 0
    while index < len(ids):
        token = ids[index]
        if token == start_id:
            inner = ids[index + 1:index + 1 + IEEE754_N_BYTES]
            closed = (
                index + IEEE754_SPAN_LENGTH <= len(ids)
                and ids[index + IEEE754_SPAN_LENGTH - 1] == end_id
                and all(byte in id_to_byte for byte in inner)
            )
            if not closed:
                return ids, None
            values.append(byte_tokens_to_float64([id_to_byte[byte] for byte in inner]))
            mapped.append(constant_id)
            index += IEEE754_SPAN_LENGTH
            continue
        if token == end_id or token in id_to_byte:
            # A stray close/content byte outside a span: not a well-formed carrier.
            return ids, None
        mapped.append(token)
        index += 1

    if not values:
        return mapped, None
    if constant_id in ids or not all(math.isfinite(value) for value in values):
        return mapped, None
    return mapped, values
