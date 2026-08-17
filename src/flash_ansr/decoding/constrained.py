"""Constrained decoding for the v24 ``ieee754_mixed`` constants representation.

A small state machine over the vocabulary mask (contract tests T6/T7):

* ``<float>`` is forbidden OUTRIGHT at every generation position -- the compact-constant
  token only ever enters a sequence by pipeline compaction, never by model emission.
  This is the decode-time (belt-and-braces) half of the T6 anti-training contract; the
  trained-model half (the logit's rank is suppressed after warmup) is T13.
* Outside an ``<ieee754>`` span: bit tokens and the closing tag are forbidden; the
  opening tag is additionally forbidden when fewer than 34 slots remain, so an opened
  span can ALWAYS terminate within the length budget.
* Inside a span: exactly ``<b0>``/``<b1>`` are admissible until 32 bits are down, then
  exactly ``</ieee754>`` -- every emission parses to a float (T7).

The mask is STATELESS per step: the per-row grammar state is recomputed from the token
prefix by a vectorized scan. That makes it trivially correct under KV-cached decoding,
mini-batching, beam reindexing and the static (position-indexed) path -- there is no
carried state to reindex, which is exactly the class of bug this avoids.

Invariant: the scan assumes the prefix's spans are themselves well-formed (true for
teacher-forced prompt prefixes serialized by the training pipeline and for any prefix
generated under this mask).
"""
import torch

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.ieee754 import (
    BIT_ONE_TOKEN,
    BIT_ZERO_TOKEN,
    IEEE754_END_TOKEN,
    IEEE754_N_BITS,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
)

#: The compact-constant token the mask forbids outright (see data.serialization).
COMPACT_CONSTANT_TOKEN = "<float>"


class IEEE754GrammarConstraint:
    """Vocabulary mask implementing the expanded-constant grammar over ``<ieee754>`` spans."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        required = (IEEE754_START_TOKEN, IEEE754_END_TOKEN, BIT_ZERO_TOKEN, BIT_ONE_TOKEN,
                    COMPACT_CONSTANT_TOKEN)
        missing = [token for token in required if token not in tokenizer]
        if missing:
            raise ValueError(
                f"Constrained ieee754 decoding requires the special tokens {list(required)}, "
                f"but the tokenizer is missing {missing}."
            )
        self.open_id = int(tokenizer[IEEE754_START_TOKEN])
        self.close_id = int(tokenizer[IEEE754_END_TOKEN])
        self.b0_id = int(tokenizer[BIT_ZERO_TOKEN])
        self.b1_id = int(tokenizer[BIT_ONE_TOKEN])
        self.float_id = int(tokenizer[COMPACT_CONSTANT_TOKEN])
        self.vocab_size = len(tokenizer)

    def forbidden(self, prefixes: torch.Tensor, remaining: int | None = None) -> torch.Tensor:
        """Boolean ``(N, vocab)`` mask of FORBIDDEN next tokens for each row of ``prefixes``.

        Parameters
        ----------
        prefixes : torch.Tensor
            ``(N, L)`` long tensor of the tokens generated so far (the decoder input whose
            last-position logits are about to be masked). ``L`` may be 0.
        remaining : int, optional
            Slots left in the sequence INCLUDING the one about to be emitted
            (``max_len - current_length``). When given and ``< 34``, opening a new span is
            forbidden so every opened span can close within the budget. ``None`` disables
            the budget rule (callers without a hard length cap).

        Returns
        -------
        torch.Tensor
            ``(N, vocab)`` boolean tensor; ``True`` marks a forbidden token. Apply with
            ``logits.masked_fill(mask, -inf)``.
        """
        if prefixes.ndim != 2:
            raise ValueError(f"prefixes must be 2-D (N, L), got shape {tuple(prefixes.shape)}")
        n_rows, length = prefixes.shape
        device = prefixes.device

        if length == 0:
            inside = torch.zeros(n_rows, dtype=torch.bool, device=device)
            n_bits = torch.zeros(n_rows, dtype=torch.long, device=device)
        else:
            positions = torch.arange(length, device=device)
            no_match = positions.new_full((n_rows, length), -1)
            last_open = torch.where(prefixes == self.open_id, positions, no_match).amax(dim=1)
            last_close = torch.where(prefixes == self.close_id, positions, no_match).amax(dim=1)
            inside = last_open > last_close
            is_bit = (prefixes == self.b0_id) | (prefixes == self.b1_id)
            after_open = positions.unsqueeze(0) > last_open.unsqueeze(1)
            n_bits = (is_bit & after_open).sum(dim=1)

        mask = torch.zeros(n_rows, self.vocab_size, dtype=torch.bool, device=device)

        # Anti-training, decode-time half (T6): <float> is forbidden in EVERY state.
        mask[:, self.float_id] = True

        outside = ~inside
        # Outside a span: bits and the close tag are grammar violations.
        mask[outside, self.b0_id] = True
        mask[outside, self.b1_id] = True
        mask[outside, self.close_id] = True
        # Budget rule: a span needs 34 slots (open + 32 bits + close) to terminate.
        if remaining is not None and remaining < IEEE754_SPAN_LENGTH:
            mask[outside, self.open_id] = True

        # Inside, fewer than 32 bits: EXACTLY <b0>/<b1> are admissible.
        bits_pending = inside & (n_bits < IEEE754_N_BITS)
        mask[bits_pending, :] = True
        mask[bits_pending, self.b0_id] = False
        mask[bits_pending, self.b1_id] = False

        # Inside, 32 bits down: EXACTLY </ieee754> is admissible.
        close_pending = inside & (n_bits >= IEEE754_N_BITS)
        mask[close_pending, :] = True
        mask[close_pending, self.close_id] = False

        return mask
