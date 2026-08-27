"""Compaction of closed ``<ieee754>`` spans on the DYNAMIC KV decode path (contract T8).

At inference the model emits expanded constants (``<ieee754>`` + 8 hex nibbles + ``</ieee754>``,
10 tokens); the pipeline compacts each closed span into ONE ``<float>`` token carrying the
decoded float32 value on the numeric channel (``input_num``). Compaction must be a
MECHANICAL NO-OP relative to the compact view: the golden test (T8) checks that logits for
every post-compaction position computed via this incremental path equal, to float
tolerance, a fresh forward pass over the compact-view sequence.

Mechanics (per batch of rows that each JUST closed a span, i.e. the span is the cache tail):

1. nibbles -> float32 value (exact, via the token codec);
2. span collapse: the 10 span tokens are replaced by ``<float>`` at the span start, the
   tail is cleared to ``<pad>``;
3. ``<float>`` + input_num: the value rides the numeric channel at the collapsed slot;
4. KV drop + re-encode at COLLAPSED positions: the 10 span entries are dropped from every
   layer's self-attention K/V (cross-attention K/V are untouched -- they are position-free
   encoder memory), then the compact token is re-encoded through the decoder with the
   truncated cache, which places it at RoPE position ``span_start`` (the dynamic path
   derives the position from the cached length). Re-encoding at the collapsed position is
   REQUIRED: the cached span keys were rotary-encoded at their ORIGINAL positions, so
   keeping them (or re-encoding at the original tail position) breaks the compact-view
   equivalence -- exactly the class of bug T8 exists to catch.

Scope: the dynamic (cat-grow) KV path; batch rows must share the span position (rows that
close spans at the SAME step always do, since spans have fixed length 10). Per-beam
compaction (desynchronized close steps, beam-reindexed caches) is T9, layered on top in
``beam_compaction.py``: it keeps per-beam caches, groups equal-length rows, and calls this
function per group -- the precondition above then holds by construction.

WHAT THE STATIC PATH NEEDS (not implemented here -- position-indexed StaticKVCache):
* a per-row (or chunk-uniform) POSITION REWIND: dropping the span is just
  ``position = span_start`` -- stale K/V beyond it need no zeroing because
  ``attend_mask(position)`` already excludes slots ``> position`` and later writes
  overwrite in place;
* per-row ``position`` support in ``forward_static``/``attend_mask`` (both take a single
  scalar today), because chunk rows desynchronize once spans close at different steps;
* the numeric step-input must become per-row (the decode loop currently expands one shared
  ``numeric_template``), so each row's ``<float>`` re-encode carries its OWN value.
"""
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch

from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_N_NIBBLES,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
    nibble_tokens_to_float32,
)
from flash_ansr.decoding.constrained import COMPACT_CONSTANT_TOKEN
from flash_ansr.utils.numeric import NUMERIC_DTYPE

if TYPE_CHECKING:  # import for typing only: the model module imports this package lazily
    from flash_ansr.model.flash_ansr_model import FlashANSRModel


@dataclass
class CompactionResult:
    """The post-compaction decode state.

    Attributes
    ----------
    sequences : torch.Tensor
        ``(B, L_buffer)`` token buffer with the span collapsed to ``<float>`` at
        ``length - 1`` and the freed tail cleared to ``<pad>``.
    input_num : torch.Tensor
        ``(B, L_buffer)`` numeric channel: the caller's values preserved, the decoded
        constant at the collapsed slot, NaN elsewhere.
    past_key_values : list
        Dynamic per-layer ``((sa_k, sa_v), (ca_k, ca_v))`` cache of length ``length``
        (the 10 span entries dropped, the compact token re-encoded at the collapsed
        position).
    length : int
        The post-compaction sequence length (``span_start + 1``).
    values : torch.Tensor
        ``(B,)`` float32 tensor of the decoded constants.
    logits : torch.Tensor
        ``(B, 1, vocab)`` next-token logits from the compact-token re-encode -- the
        continuation distribution of the compact view.
    """

    sequences: torch.Tensor
    input_num: torch.Tensor
    past_key_values: list
    length: int
    values: torch.Tensor
    logits: torch.Tensor


def compact_closed_ieee754_spans(
    model: "FlashANSRModel",
    sequences: torch.Tensor,
    current_length: int,
    past_key_values: list,
    memory: torch.Tensor,
    input_num: torch.Tensor | None = None,
) -> CompactionResult:
    """Compact the just-closed ``<ieee754>`` span of every row (dynamic KV path).

    Parameters
    ----------
    model : FlashANSRModel
        The model whose decoder re-encodes the compact token.
    sequences : torch.Tensor
        ``(B, L_buffer)`` long tensor; positions ``[0, current_length)`` are the generated
        tokens and EVERY row's last 10 of them must form a complete span (the row just
        emitted ``</ieee754>``).
    current_length : int
        Number of generated tokens; the cache must cover exactly these positions.
    past_key_values : list
        The dynamic per-layer cache built while generating the expanded sequence.
    memory : torch.Tensor
        Encoder memory (used by cross-attention on the re-encode step; the cached
        cross-attention K/V make it a pass-through).
    input_num : torch.Tensor, optional
        ``(B, L_buffer)`` or ``(B, L_buffer, 1)`` numeric channel accumulated so far
        (earlier compacted constants). ``None`` -> all-NaN.

    Raises
    ------
    ValueError
        If any row's tail is not a complete, well-formed span, or a decoded value is
        non-finite (the caller should leave such spans expanded rather than compact a
        value the numeric channel cannot carry -- NaN means "no value" there).
    """
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2-D (B, L), got shape {tuple(sequences.shape)}")
    if current_length < IEEE754_SPAN_LENGTH:
        raise ValueError(
            f"current_length={current_length} cannot contain a {IEEE754_SPAN_LENGTH}-token span."
        )
    cached_length = past_key_values[0][0][0].shape[2]
    if cached_length != current_length:
        raise ValueError(
            f"KV cache covers {cached_length} positions but current_length={current_length}; "
            f"compaction requires the span to be the cache tail."
        )

    tokenizer = model.tokenizer
    open_id = int(tokenizer[IEEE754_START_TOKEN])
    close_id = int(tokenizer[IEEE754_END_TOKEN])
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]
    id_to_nibble = {token_id: NIBBLE_TOKENS[value] for value, token_id in enumerate(nibble_ids)}
    nibble_id_tensor = torch.tensor(nibble_ids, dtype=torch.long, device=sequences.device)
    float_id = int(tokenizer[COMPACT_CONSTANT_TOKEN])
    pad_id = int(tokenizer["<pad>"])

    span_start = current_length - IEEE754_SPAN_LENGTH
    span = sequences[:, span_start:current_length]

    if not bool((span[:, 0] == open_id).all()) or not bool((span[:, -1] == close_id).all()):
        raise ValueError(
            "Every row must have JUST closed an expanded span: expected "
            f"<ieee754> ... </ieee754> at positions [{span_start}, {current_length})."
        )
    inner = span[:, 1:-1]
    if not bool(torch.isin(inner, nibble_id_tensor).all()):
        raise ValueError("Malformed span: non-nibble token between the <ieee754> tags.")

    # 1. nibbles -> float32 values (exact decode via the token codec).
    batch = sequences.shape[0]
    decoded = []
    for row in inner.tolist():
        decoded.append(nibble_tokens_to_float32([id_to_nibble[token] for token in row]))
    # dtype is load-bearing: without it torch.tensor() builds float32 and this rejects
    # exactly the magnitudes the v25 migration exists to admit -- while beam_compaction's
    # math.isfinite on the same value ADMITS them, and the beam loop then crashes here.
    if not all(torch.isfinite(torch.tensor(decoded, dtype=NUMERIC_DTYPE)).tolist()):
        raise ValueError(
            f"Non-finite decoded constant(s) {decoded!r}: refusing to compact -- NaN on the "
            f"numeric channel means 'no value'. Leave such spans expanded instead."
        )
    values = torch.tensor(decoded, dtype=NUMERIC_DTYPE, device=sequences.device)

    # 2. span collapse: one <float> at span_start, freed tail cleared to <pad>.
    new_length = span_start + 1
    new_sequences = sequences.clone()
    new_sequences[:, span_start] = float_id
    new_sequences[:, new_length:current_length] = pad_id

    # 3. the value rides the numeric channel at the collapsed slot.
    if input_num is None:
        new_input_num = torch.full(
            (batch, sequences.shape[1]), float("nan"),
            dtype=NUMERIC_DTYPE, device=sequences.device)
    else:
        if input_num.ndim == 3:
            input_num = input_num.squeeze(-1)
        new_input_num = input_num.clone().to(dtype=NUMERIC_DTYPE, device=sequences.device)
    new_input_num[:, span_start:current_length] = float("nan")
    new_input_num[:, span_start] = values

    # 4. KV drop: remove the 10 span entries from every layer's SELF-attention cache;
    # cross-attention K/V (position-free encoder memory) are reused untouched.
    trimmed = [
        (
            (sa_k[:, :, :span_start, :], sa_v[:, :, :span_start, :]),
            (ca_k, ca_v),
        )
        for ((sa_k, sa_v), (ca_k, ca_v)) in past_key_values
    ]

    # ... and re-encode the compact token at the COLLAPSED position: the dynamic path
    # derives the RoPE position from the cached length (= span_start), so the <float>
    # token lands exactly where the compact view places it.
    result = model.forward(
        new_sequences[:, span_start:new_length],
        None,
        input_num=values.view(batch, 1, 1),
        memory=memory,
        past_key_values=trimmed,
        use_cache=True,
    )
    assert isinstance(result, tuple)  # use_cache=True -> (logits, past); narrow the union
    logits, new_past = result

    return CompactionResult(
        sequences=new_sequences,
        input_num=new_input_num,
        past_key_values=new_past,
        length=new_length,
        values=values,
        logits=logits,
    )


# ---------------------------------------------------------------------------
# Static-path compaction: no cache surgery, just a per-row position rewind.
#
# The function above rewrites the dynamic cat-grow cache. The static
# position-indexed cache needs none of that: a compacting row sets its own
# position back to `span_start`, the stale slots fall outside `attend_mask`, and
# later writes overwrite them in place. All the caller needs from here is the
# VALUE and the go/no-go, which is what these two provide.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpanCodecIds:
    """Token ids for the expanded-span codec, resolved once per decode rather than per step."""

    open_id: int
    close_id: int
    float_id: int
    id_to_nibble: dict


def span_codec_ids(tokenizer: "Any") -> SpanCodecIds:
    """Resolve the span token ids for `tokenizer` (256 lookups per call -- hoist it)."""
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]
    return SpanCodecIds(
        open_id=int(tokenizer[IEEE754_START_TOKEN]),
        close_id=int(tokenizer[IEEE754_END_TOKEN]),
        float_id=int(tokenizer[COMPACT_CONSTANT_TOKEN]),
        id_to_nibble={token_id: NIBBLE_TOKENS[value] for value, token_id in enumerate(nibble_ids)},
    )


def closed_span_value(ids: SpanCodecIds, row: "Sequence[int]", end_index: int) -> float | None:
    """Decode the span whose CLOSING tag sits at ``row[end_index]``.

    Returns ``None`` for "do not compact", which covers both refusals:
    * the span is malformed or runs off the start of the buffer; and
    * it decodes to a non-finite value -- the numeric channel carries NaN to mean
      "no value", so a non-finite constant must stay expanded (the landed T10 rule,
      honoured here by exclusion rather than by raising).
    """
    span_start = end_index - IEEE754_SPAN_LENGTH + 1
    if span_start < 0 or int(row[end_index]) != ids.close_id or int(row[span_start]) != ids.open_id:
        return None
    tokens = []
    for token_id in row[span_start + 1:end_index]:
        nibble = ids.id_to_nibble.get(int(token_id))
        if nibble is None:
            return None
        tokens.append(nibble)
    if len(tokens) != IEEE754_N_NIBBLES:
        return None
    value = nibble_tokens_to_float32(tokens)
    return value if math.isfinite(value) else None
