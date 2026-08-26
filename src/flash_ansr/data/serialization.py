"""Constant serialization for training sequences (the ``constant_representation`` gate).

``'ieee754_mixed'``, the only representation, replaces each constant occurrence -- per
constant independently with probability ``expanded_probability`` (default 0.5, seeded RNG)
-- with one of two forms:

* EXPANDED (10 tokens): ``<ieee754>`` + 8 hex-nibble tokens + ``</ieee754>``; the numeric channel
  is NaN across the whole span (the value lives in the nibbles).
* COMPACT (1 token): the existing ``<float>`` special token, with the float32 value on the
  numeric channel at that position. The label at positions whose TARGET is ``<float>`` is
  loss-masked downstream (:func:`flash_ansr.data.collate.mask_float_targets`): the model must
  never be trained to emit ``<float>``; the token exists so histories can carry compacted
  constants (at inference the model emits bits and the pipeline compacts them).

Non-finite constants raise at serialization time -- assert, don't assume.
"""
import struct
import math
import re
from typing import Sequence

import numpy as np

from flash_ansr.utils.ieee754 import (
    IEEE754_N_NIBBLES,
    IEEE754_SPAN_LENGTH,
    NIBBLE_TOKENS,
    nibble_tokens_to_float32,
    wrap_float32,
)

#: The compact-constant token. Deliberately the EXISTING ``<float>`` special (it already
#: carries a value on the numeric channel in the prompt-serialization path); no new token.
COMPACT_CONSTANT_TOKEN = "<float>"

# v24 task-block grammar (owner ruling 2026-08-24): the harness owns the structure --
# every opener/selector below is force-fed and loss-masked; the model owns content
# nibbles and closing tags. <complexity>/<float> already exist in the v24 vocabulary;
# the predict_y tokens are new with this feature.
COMPLEXITY_START_TOKEN = "<complexity>"
COMPLEXITY_END_TOKEN = "</complexity>"
PREDICT_Y_START_TOKEN = "<predict_y>"
PREDICT_Y_END_TOKEN = "</predict_y>"
POINT_START_TOKEN = "<point>"
POINT_END_TOKEN = "</point>"
PREDICT_Y_TOKENS = (PREDICT_Y_START_TOKEN, PREDICT_Y_END_TOKEN, POINT_START_TOKEN, POINT_END_TOKEN)
# Hypothesis mode (owner ruling 2026-08-24): a harness-inserted flag that LICENSES the
# model to open and fill property blocks on its own (opener + content supervised). The
# flag itself is never supervised -- only the harness may put the model into hypothesis
# mode; without it, openers stay force-fed and loss-masked.
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

#: Legal values of the ``constant_representation`` config key.
CONSTANT_REPRESENTATION_IEEE754_MIXED = "ieee754_mixed"
CONSTANT_REPRESENTATIONS = (CONSTANT_REPRESENTATION_IEEE754_MIXED,)

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
    *,
    representation: str = CONSTANT_REPRESENTATION_IEEE754_MIXED,
    rng: np.random.Generator | None = None,
    expanded_probability: float = 0.5,
    expanded_mask: "Sequence[bool] | None" = None,
    zero_tail_bits: int = 0,
) -> tuple[list[str], list[float]]:
    """Serialize the constant placeholders of a token sequence under ``representation``.

    Parameters
    ----------
    tokens : Sequence[str]
        The expression tokens; constants are the ``<constant>`` / ``C_<i>`` placeholders.
    constants : Sequence[float or None]
        The fitted values, one per placeholder occurrence, in order. A ``None`` entry
        KEEPS that occurrence as a literal ``<constant>`` placeholder (nan on the
        numeric channel, no rng draw consumed) -- the promptable-mask target format,
        where the policy's placeholders survive serialization while kept structural
        literals still ride the ieee754 spelling.
    representation : str, default 'ieee754_mixed'
        ``'ieee754_mixed'`` applies the per-constant 50/50 expanded/compact mixing.
    rng : numpy.random.Generator, optional
        The (seeded) RNG driving the per-constant coin flips. Required for
        ``'ieee754_mixed'``.
    expanded_probability : float, default 0.5
        Probability that a constant is serialized in EXPANDED form.
    expanded_mask : Sequence[bool], optional
        Force the per-constant form: ``True`` = expanded span, ``False`` = compact
        ``<float>``. One entry per placeholder, in order. When given, NO rng draws are
        consumed (the paired T12 eval builds both views of one instance this way).
    zero_tail_bits : int, default 0
        Zero the lowest ``n`` mantissa bits of the float32 in the EXPANDED span
        spelling (the pilot's D-arm tail policy; the T13 re-check compares 0 vs 16).
        The compact form's numeric-channel value keeps full float32 precision.

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

    if rng is None and expanded_mask is None:
        raise ValueError("The 'ieee754_mixed' representation requires a (seeded) rng "
                         "(or an explicit expanded_mask).")
    if not 0 <= zero_tail_bits <= 23:
        raise ValueError(f"zero_tail_bits must lie in the float32 mantissa (0..23), got {zero_tail_bits}.")

    values = [None if value is None else float(value) for value in constants]
    n_placeholders = sum(1 for token in tokens if _is_constant_placeholder(token))
    if n_placeholders != len(values):
        raise ValueError(
            f"Constant count mismatch: {n_placeholders} placeholder(s) in {tokens!r} but "
            f"{len(values)} value(s)."
        )
    if expanded_mask is not None and len(expanded_mask) != n_placeholders:
        raise ValueError(
            f"expanded_mask length mismatch: {n_placeholders} placeholder(s) but "
            f"{len(expanded_mask)} mask entrie(s)."
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
            # The policy's placeholder: it IS the target token. No coin flip, no span.
            serialized.append(token)
            numeric.append(float("nan"))
            continue
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite constant {value!r} at placeholder {constant_index - 1}: the data "
                f"generator must never emit inf/nan constants."
            )
        value32 = float(np.float32(value))

        if expanded_mask is not None:
            expand = bool(expanded_mask[constant_index - 1])
        else:
            assert rng is not None  # narrowed by the guard above
            expand = bool(rng.random() < expanded_probability)

        if expand:
            span_value = value32
            if zero_tail_bits:
                # D-arm tail policy: zero the low mantissa bits of the SPELLING only.
                pattern = int.from_bytes(struct.pack(">f", np.float32(span_value)), "big")
                pattern &= ~((1 << zero_tail_bits) - 1)
                span_value = float(struct.unpack(">f", pattern.to_bytes(4, "big"))[0])
            span = wrap_float32(span_value)  # raises if the value overflowed float32 to non-finite
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


def replace_ieee754_spans_with_constants(
    token_ids: Sequence[int],
    *,
    start_id: int,
    end_id: int,
    nibble_ids: Sequence[int],
    constant_id: int,
) -> tuple[list[int], list[float] | None]:
    """Map expanded ``<ieee754>`` spans to ``<constant>`` slots + their float32 values (T11).

    The DESERIALIZATION half of the refiner handshake: each well-formed 10-token span in a
    generated beam collapses to one ``constant_id`` token, and the decoded values (exact,
    via the token codec -- no decimal round-trip) become the refiner's verbatim ``p0`` in
    order of appearance.

    Parameters
    ----------
    nibble_ids : Sequence[int]
        The 16 hex-nibble token ids IN NIBBLE-VALUE ORDER (``nibble_ids[10]`` is ``<ha>``),
        i.e. ``[int(tokenizer[token]) for token in NIBBLE_TOKENS]``.

    Returns
    -------
    tuple[list[int], list[float] | None]
        ``(mapped_ids, values)``. ``values`` is the per-span float32 list ONLY when the
        init is sound: at least one span, every span well-formed, every value finite, and
        no pre-existing ``constant_id`` in the input (a bare placeholder -- e.g.
        constantified sugar -- would break the slot alignment; the skeleton is still
        mapped, the init is withheld and refinement falls back to its own seed). Any MALFORMED
        span returns the input unchanged with ``None`` (the beam is not a v24 carrier;
        downstream validity checks dispose of it as today).
    """
    if len(nibble_ids) != len(NIBBLE_TOKENS):
        raise ValueError(
            f"Expected {len(NIBBLE_TOKENS)} nibble ids in nibble-value order, got {len(nibble_ids)}."
        )
    id_to_nibble = {token_id: NIBBLE_TOKENS[value] for value, token_id in enumerate(nibble_ids)}

    ids = list(token_ids)
    mapped: list[int] = []
    values: list[float] = []
    index = 0
    while index < len(ids):
        token = ids[index]
        if token == start_id:
            inner = ids[index + 1:index + 1 + IEEE754_N_NIBBLES]
            closed = (
                index + IEEE754_SPAN_LENGTH <= len(ids)
                and ids[index + IEEE754_SPAN_LENGTH - 1] == end_id
                and all(nibble in id_to_nibble for nibble in inner)
            )
            if not closed:
                return ids, None
            values.append(nibble_tokens_to_float32([id_to_nibble[nibble] for nibble in inner]))
            mapped.append(constant_id)
            index += IEEE754_SPAN_LENGTH
            continue
        if token == end_id or token in id_to_nibble:
            # A stray close/nibble outside a span: not a v24-well-formed carrier.
            return ids, None
        mapped.append(token)
        index += 1

    if not values:
        return mapped, None
    if constant_id in ids or not all(math.isfinite(value) for value in values):
        return mapped, None
    return mapped, values
