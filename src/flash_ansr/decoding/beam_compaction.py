"""Beam search with PER-BEAM KV compaction of closed ``<ieee754>`` spans (contract T9/T10).

The per-beam extension of :func:`flash_ansr.decoding.compaction.compact_closed_ieee754_spans`
(T8, batch-uniform): under beam search, spans close at DIFFERENT steps, so beams
desynchronize -- each beam carries its own cache length and RoPE frontier. This loop keeps
the dynamic cache PER BEAM (batch-1 per row), groups equal-length rows for each forward
(the dynamic path derives positions from the cached length, so only equal-length rows may
batch), and compacts exactly the rows that JUST emitted ``</ieee754>`` -- those rows share
their span position by construction (spans are fixed 10-long), satisfying the T8 helper's
uniform-tail precondition per group.

Safety rules (T10):

* compaction fires ONLY on a closed tag: the trigger is the close token at the row's cache
  tail, never a mid-span state; a beam pruned mid-expansion simply stops being selected as
  a parent -- per-beam caches are rebuilt from the surviving parents each step, so pruned
  state is dropped wholesale, never orphaned;
* a closed span decoding to a NON-FINITE value is left expanded (the numeric channel
  cannot carry it -- the landed T8 refusal, honored here by exclusion instead of an error).

Score accounting (T9 comparability): scores are cumulative log-probs over EMITTED tokens,
exactly as in the uniform loop. A compacted constant contributes its 10 emitted-token
log-probs once, whenever they were emitted, and the continuation is scored from the
compacted state (whose logits the T8/T9 golden equality pins to the compact view) -- no
rescoring, no double counting, so beams that compacted at different steps rank on one
scale. There is no length normalization to skew.

Return convention: sequences present constants in EXPANDED form -- compacted ``<float>``
slots are re-expanded from their numeric-channel values before leaving this function
(exact: float32 -> hex nibbles is the T1 round-trip). Downstream (validity, refiner handshake)
therefore sees one uniform carrier, and the compact token never leaks out of the decode.
Prompt-prefix ``<float>`` tokens (e.g. the complexity slot) are NOT re-expanded: only
positions at or beyond the generated region are.

Completed-pool dedup under ``unique=True`` uses EXACT expanded sequences after a
validity gate (spans mapped to ``<constant>`` placeholders for the check): value-carrying
beams admit no placeholder-level simplify, so canonicalization is identity here.
"""
import heapq
import math
from typing import TYPE_CHECKING, cast

import torch
from tqdm import tqdm

from flash_ansr.data.serialization import replace_ieee754_spans_with_constants
from flash_ansr.decoding.compaction import compact_closed_ieee754_spans
from flash_ansr.decoding.constrained import COMPACT_CONSTANT_TOKEN, IEEE754GrammarConstraint
from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
    nibble_tokens_to_float32,
    wrap_float32,
)

if TYPE_CHECKING:  # import for typing only: the model module imports this package lazily
    from flash_ansr.model.flash_ansr_model import FlashANSRModel
    from flash_ansr.preprocessing import PromptPrefix


def _gather_rows(caches: list[list | None], indices: list[int]) -> list:
    """Concatenate per-beam (batch-1) caches into one batched cache, row order = ``indices``."""
    if len(indices) == 1:
        cache = caches[indices[0]]
        assert cache is not None
        return cache
    first = caches[indices[0]]
    assert first is not None
    return [
        (
            (
                torch.cat([caches[i][layer_index][0][0] for i in indices], dim=0),  # type: ignore[index]
                torch.cat([caches[i][layer_index][0][1] for i in indices], dim=0),  # type: ignore[index]
            ),
            (
                torch.cat([caches[i][layer_index][1][0] for i in indices], dim=0),  # type: ignore[index]
                torch.cat([caches[i][layer_index][1][1] for i in indices], dim=0),  # type: ignore[index]
            ),
        )
        for layer_index in range(len(first))
    ]


def _split_rows(past: list, n_rows: int) -> list[list]:
    """Slice a batched cache back into per-beam (batch-1) caches (views; caches are append-only)."""
    return [
        [
            (
                (layer[0][0][row:row + 1], layer[0][1][row:row + 1]),
                (layer[1][0][row:row + 1], layer[1][1][row:row + 1]),
            )
            for layer in past
        ]
        for row in range(n_rows)
    ]


def _tail_span_value(row: torch.Tensor, length: int, id_to_nibble: dict[int, str]) -> float:
    """Decode the value of the just-closed span at the tail of ``row`` (grammar-guaranteed nibbles)."""
    inner = row[length - IEEE754_SPAN_LENGTH + 1:length - 1].tolist()
    return nibble_tokens_to_float32([id_to_nibble[token] for token in inner])


def beam_search_with_compaction(
    model: "FlashANSRModel",
    data: torch.Tensor,
    beam_width: int = 4,
    max_len: int = 100,
    batch_size: int = 128,
    unique: bool = True,
    verbose: bool = False,
    limit_expansions: bool = True,
    *,
    prompt_prefix: "PromptPrefix | None" = None,
    initial_tokens: list[int] | None = None,
    input_num: list[float] | None = None,
) -> tuple[list[list[int]], list[float], list[bool]]:
    """Constrained beam search on the dynamic KV path, compacting closed spans per beam.

    The ``compact_ieee754=True`` arm of :meth:`FlashANSRModel.beam_search`; parameters and
    return value match it (the grammar mask is always on here -- see the module docstring
    for the compaction mechanics and the expanded-form return convention).
    """
    device = data.device
    tokenizer = model.tokenizer
    grammar = IEEE754GrammarConstraint(tokenizer)

    open_id = int(tokenizer[IEEE754_START_TOKEN])
    close_id = int(tokenizer[IEEE754_END_TOKEN])
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]
    id_to_nibble = {token_id: NIBBLE_TOKENS[value] for value, token_id in enumerate(nibble_ids)}
    float_id = int(tokenizer[COMPACT_CONSTANT_TOKEN])
    constant_id = int(tokenizer['<constant>'])
    eos_token_id = int(tokenizer['<eos>'])
    pad_token_id = int(tokenizer['<pad>'])

    base_tokens, base_input_num = model._resolve_generation_prefix(
        prompt_prefix=prompt_prefix,
        initial_tokens=initial_tokens,
        input_num=input_num,
    )
    if isinstance(base_input_num, torch.Tensor):
        base_input_num = base_input_num.tolist()

    prefix_length = len(base_tokens)
    if prefix_length >= max_len:
        raise ValueError(f"Initial token prefix length ({prefix_length}) exceeds max_len ({max_len}).")

    memory = model._create_memory(data)

    def expanded_view(row: torch.Tensor, num_row: torch.Tensor, length: int) -> list[int]:
        """The expanded-form sequence: compacted <float> slots (generated region only)
        re-serialized as spans from their numeric-channel values (T1: exact)."""
        out: list[int] = []
        for position in range(length):
            token = int(row[position])
            if token == float_id and position >= prefix_length:
                out.extend(int(tokenizer[nibble]) for nibble in wrap_float32(float(num_row[position])))
            else:
                out.append(token)
        return out

    # NaN = "no value" on the numeric channel, matching the training collate and the
    # compact-view forward the T9 golden equality compares against (never None here).
    sequences = torch.full((beam_width, max_len), pad_token_id, device=device, dtype=torch.long)
    sequences[:, :prefix_length] = torch.tensor(base_tokens, device=device, dtype=torch.long)
    input_nums = torch.full((beam_width, max_len), float('nan'), device=device, dtype=torch.float32)
    if base_input_num is not None:
        input_nums[:, :prefix_length] = torch.tensor(base_input_num, device=device, dtype=torch.float32)

    lengths = torch.full((beam_width,), prefix_length, device=device, dtype=torch.long)
    scores = torch.full((beam_width,), float('-inf'), device=device, dtype=torch.float)
    scores[0] = 0.0
    finished = torch.zeros(beam_width, dtype=torch.bool, device=device)
    if prefix_length and base_tokens[-1] == eos_token_id:
        finished[0] = True

    caches: list[list | None] = [None] * beam_width

    completed_sequences_heap: list[tuple[float, tuple[int, ...]]] = []
    completed_sequences_scores: dict[tuple[int, ...], float] = {}
    validity_cache: dict[tuple[int, ...], bool] = {}
    n_pruned = 0

    def register_completed_sequence(seq_tuple: tuple[int, ...], score: float) -> None:
        nonlocal n_pruned

        existing_score = completed_sequences_scores.get(seq_tuple)
        if existing_score is not None and score <= existing_score:
            n_pruned += 1
            return

        completed_sequences_scores[seq_tuple] = score
        heapq.heappush(completed_sequences_heap, (score, seq_tuple))

        while len(completed_sequences_scores) > beam_width:
            prune_score, prune_key = heapq.heappop(completed_sequences_heap)
            current_score = completed_sequences_scores.get(prune_key)
            if current_score is None:
                continue
            if current_score != prune_score:
                continue
            del completed_sequences_scores[prune_key]
            n_pruned += 1
            break

    def expression_is_registrable(expanded: list[int]) -> bool:
        """The unique-path validity gate: spans map to '<constant>' placeholders for the
        check (no simplify -- value-carrying beams dedup by exact sequence)."""
        key = tuple(expanded)
        cached = validity_cache.get(key)
        if cached is not None:
            return cached
        try:
            expression_tokens, _before, _after = tokenizer.extract_expression_from_beam(expanded)
        except ValueError:
            validity_cache[key] = False
            return False
        mapped, _values = replace_ieee754_spans_with_constants(
            expression_tokens, start_id=open_id, end_id=close_id,
            nibble_ids=nibble_ids, constant_id=constant_id)
        decoded = tokenizer.decode_expression(mapped)
        registrable = bool(model.simplipy_engine.is_valid(decoded)) and len(decoded) > 1
        validity_cache[key] = registrable
        return registrable

    # Each step emits one token per live beam; each compaction frees 9 slots and there is
    # at most one compaction per 10 emissions, so 2x the uniform budget caps the loop.
    max_steps = 2 * (max_len - prefix_length)
    pbar = tqdm(total=max_steps, disable=not verbose,
                desc=f"Generating beams with compaction (max length: {max_len})", smoothing=0.0)

    with torch.no_grad():
        for _step in range(max_steps):
            can_extend = lengths < max_len
            active_mask = (~finished) & torch.isfinite(scores) & can_extend
            if not torch.any(active_mask):
                break

            active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()

            # Group active beams by length: the dynamic path derives RoPE positions from
            # the cached length, so only equal-length rows may share a forward.
            groups: dict[int, list[int]] = {}
            for beam_index in active_indices:
                groups.setdefault(int(lengths[beam_index]), []).append(beam_index)

            candidate_scores_list: list[torch.Tensor] = []
            candidate_parents: list[int] = []
            candidate_tokens: list[torch.Tensor] = []

            for group_length, group_beams in groups.items():
                for start_index in range(0, len(group_beams), batch_size):
                    chunk = group_beams[start_index:start_index + batch_size]
                    chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.long)

                    if caches[chunk[0]] is None:
                        # Prefill (first step; only the seed beam is active): the whole
                        # prefix in one forward, numeric channel included.
                        result = model.forward(
                            sequences[chunk_tensor, :group_length], None,
                            input_num=input_nums[chunk_tensor, :group_length].unsqueeze(-1),
                            memory=memory, past_key_values=None, use_cache=True)
                    else:
                        result = model.forward(
                            sequences[chunk_tensor, group_length - 1:group_length], None,
                            input_num=input_nums[chunk_tensor, group_length - 1:group_length].unsqueeze(-1),
                            memory=memory, past_key_values=_gather_rows(caches, chunk),
                            use_cache=True)
                    assert isinstance(result, tuple)
                    logits, new_past = result
                    for row, row_cache in enumerate(_split_rows(new_past, len(chunk))):
                        caches[chunk[row]] = row_cache
                    last_logits = logits[:, -1, :].clone()

                    # Per-beam KV compaction (T9): EXACTLY the rows that just closed a
                    # span, and only when its value is finite (T10: non-finite spans stay
                    # expanded -- the numeric channel cannot carry them).
                    closing = [
                        (position, beam_index)
                        for position, beam_index in enumerate(chunk)
                        if group_length - IEEE754_SPAN_LENGTH >= prefix_length
                        and int(sequences[beam_index, group_length - 1]) == close_id
                        and math.isfinite(_tail_span_value(sequences[beam_index], group_length, id_to_nibble))
                    ]
                    if closing:
                        compact_positions = [position for position, _ in closing]
                        compact_beams = [beam_index for _, beam_index in closing]
                        compact_tensor = torch.tensor(compact_beams, device=device, dtype=torch.long)
                        compaction = compact_closed_ieee754_spans(
                            model, sequences[compact_tensor], current_length=group_length,
                            past_key_values=_gather_rows(caches, compact_beams),
                            memory=memory, input_num=input_nums[compact_tensor])
                        sequences[compact_tensor] = compaction.sequences
                        input_nums[compact_tensor] = compaction.input_num
                        lengths[compact_tensor] = compaction.length
                        for row, row_cache in enumerate(_split_rows(compaction.past_key_values, len(compact_beams))):
                            caches[compact_beams[row]] = row_cache
                        # The compact-token re-encode logits ARE the continuation
                        # distribution of the compact view (the T9 golden equality).
                        last_logits[compact_positions] = compaction.logits[:, -1, :]

                    step_log_probs = torch.log_softmax(last_logits, dim=-1)

                    # Grammar mask over the CURRENT (possibly compacted) per-beam prefix;
                    # forbidden -> -inf without renormalizing (scores stay comparable).
                    for position, beam_index in enumerate(chunk):
                        beam_length = int(lengths[beam_index])
                        forbidden = grammar.forbidden(
                            sequences[beam_index:beam_index + 1, :beam_length],
                            remaining=max_len - beam_length)[0]
                        step_log_probs[position] = step_log_probs[position].masked_fill(forbidden, float('-inf'))

                    vocab_size = step_log_probs.size(-1)
                    if limit_expansions:
                        expansion_factor = 2 if unique else 1
                        expansion_per_beam = max(1, min(vocab_size, beam_width * expansion_factor))
                        top_log_probs, top_token_ids = torch.topk(step_log_probs, k=expansion_per_beam, dim=-1)
                    else:
                        expansion_per_beam = vocab_size
                        top_log_probs = step_log_probs
                        top_token_ids = torch.arange(vocab_size, device=device, dtype=torch.long).unsqueeze(0).expand(len(chunk), -1)

                    candidate_scores_list.append(scores[chunk_tensor].unsqueeze(1) + top_log_probs)
                    candidate_parents.extend(beam_index for beam_index in chunk for _ in range(expansion_per_beam))
                    candidate_tokens.append(top_token_ids.reshape(-1))

            flat_scores = torch.cat(candidate_scores_list).reshape(-1)
            flat_parents = candidate_parents
            flat_tokens = torch.cat(candidate_tokens)

            sorted_scores, sorted_indices = torch.sort(flat_scores, descending=True)
            sorted_indices_cpu: list[int] = sorted_indices.tolist()
            sorted_tokens_cpu: list[int] = flat_tokens[sorted_indices].tolist()
            sorted_scores_cpu: list[float] = sorted_scores.tolist()

            next_sequences = torch.full_like(sequences, pad_token_id)
            next_input_nums = torch.full_like(input_nums, float('nan'))
            next_lengths = torch.zeros_like(lengths)
            next_scores = torch.full_like(scores, float('-inf'))
            next_finished = torch.zeros_like(finished)
            next_caches: list[list | None] = [None] * beam_width

            next_beam_set: set[tuple[int, ...]] = set()
            next_count = 0

            # Beams parked at max_len keep their slots (they are the length-capped
            # fallbacks); only the remaining slots go to new candidates.
            for beam_index in range(beam_width):
                if bool(torch.isfinite(scores[beam_index])) and not bool(finished[beam_index]) and int(lengths[beam_index]) >= max_len:
                    next_sequences[next_count] = sequences[beam_index]
                    next_input_nums[next_count] = input_nums[beam_index]
                    next_lengths[next_count] = lengths[beam_index]
                    next_scores[next_count] = scores[beam_index]
                    next_caches[next_count] = caches[beam_index]
                    next_count += 1

            for rank_index in range(len(sorted_scores_cpu)):
                parent_index = flat_parents[sorted_indices_cpu[rank_index]]
                token_id = sorted_tokens_cpu[rank_index]
                new_score = sorted_scores_cpu[rank_index]

                # -inf candidates are grammar-forbidden expansions (topk surfaces them
                # when fewer than k tokens are admissible inside a span).
                if new_score == float('-inf'):
                    continue

                parent_length = int(lengths[parent_index])

                if token_id == eos_token_id:
                    expanded = expanded_view(sequences[parent_index], input_nums[parent_index], parent_length)
                    expanded.append(eos_token_id)
                    if unique and not expression_is_registrable(expanded):
                        n_pruned += 1
                        continue
                    register_completed_sequence(tuple(expanded), new_score)
                    continue

                if next_count >= beam_width:
                    if (completed_sequences_heap
                            and len(completed_sequences_scores) >= beam_width
                            and new_score < completed_sequences_heap[0][0]):
                        break
                    continue

                if unique:
                    seq_tuple = (*sequences[parent_index, :parent_length].tolist(), token_id)
                    if seq_tuple in next_beam_set:
                        n_pruned += 1
                        continue
                    next_beam_set.add(seq_tuple)

                next_sequences[next_count] = sequences[parent_index]
                next_sequences[next_count, parent_length] = token_id
                next_input_nums[next_count] = input_nums[parent_index]
                next_lengths[next_count] = parent_length + 1
                next_scores[next_count] = new_score
                next_caches[next_count] = caches[parent_index]
                next_count += 1

            if next_count == 0 and completed_sequences_scores:
                break

            sequences = next_sequences
            input_nums = next_input_nums
            lengths = next_lengths
            scores = next_scores
            finished = next_finished
            caches = next_caches

            pbar.set_postfix({'completed': len(completed_sequences_scores), 'pruned': n_pruned})
            pbar.update(1)
        pbar.close()

    combined_sequences: list[tuple[list[int], float, bool]] = [
        (list(seq_tuple), score, True) for seq_tuple, score in completed_sequences_scores.items()
    ]

    # Active beams only as a last resort (no EOS penalty in their scores -- same rule as
    # the uniform loop); they may legitimately end mid-span, flagged not-completed.
    if len(combined_sequences) < beam_width:
        for beam_index in range(beam_width):
            if bool(torch.isfinite(scores[beam_index])):
                beam_length = int(lengths[beam_index].item())
                if beam_length == 0:
                    continue
                combined_sequences.append((
                    expanded_view(sequences[beam_index], input_nums[beam_index], beam_length),
                    float(scores[beam_index].item()), False))

    expr_start_token_id = tokenizer.token2idx.get('<expression>')
    expr_end_token_id = tokenizer.token2idx.get('</expression>')

    combined_sequences_final: list[tuple[list[int], float, bool]] = []
    for seq, score, is_complete in combined_sequences:
        # The uniform loop's repair rule: re-close an opened <expression> so downstream
        # parsing holds; sequences that never opened one are unparseable and drop.
        if expr_end_token_id is not None and expr_end_token_id not in seq:
            if expr_start_token_id is None or expr_start_token_id not in seq:
                continue
            if eos_token_id in seq:
                eos_position = seq.index(eos_token_id)
                seq = seq[:eos_position] + [expr_end_token_id] + seq[eos_position:]
            else:
                seq = seq + [expr_end_token_id]
        # constantify_expression mirrors its input representation; ids in -> ids out.
        constantified_seq = cast(list[int], tokenizer.constantify_expression(seq))
        combined_sequences_final.append((constantified_seq, score, is_complete))

    combined_sequences_final = sorted(combined_sequences_final, key=lambda x: x[1], reverse=True)
    top = combined_sequences_final[:beam_width]

    return [seq for seq, _, _ in top], [score for _, score, _ in top], [flag for _, _, flag in top]
