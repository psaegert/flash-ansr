"""T12 -- the paired e3 eval on the real stack (contract 'Training-run acceptance').

The pilot's e3 metric, re-derived for real validation data: for one instance, the NLL
of the TARGET constant's expanded hex-nibble span is computed twice under teacher
forcing -- once with every history constant EXPANDED (view E), once with every history
constant COMPACT (view C, the inference pattern) -- and the gap is their difference.
Acceptance mirrors the pilot's prediction: gap ~ 0 for the shipped per-constant mixing
policy. A persistent positive gap means compacted histories are out-of-distribution and
the mixing policy is broken (the pilot's B arm, +0.032, reproduced twice).

Views are built with ``serialize_constant_tokens(expanded_mask=...)`` -- forced forms,
no rng -- so both views of an instance carry byte-identical history VALUES and differ
only in their spelling. Instances need >= 2 constants (no history, no gap) and both
views must fit ``max_seq_len``.
"""
from typing import Any

import numpy as np
import torch

from flash_ansr.data.serialization import (
    CONSTANT_REPRESENTATION_IEEE754_MIXED,
    _is_constant_placeholder,
    serialize_constant_tokens,
)
from flash_ansr.utils.ieee754 import IEEE754_N_NIBBLES, IEEE754_START_TOKEN


def build_paired_views(
    skeleton: "list[str]",
    constants: "list[float]",
    tokenizer: Any,
    max_seq_len: int,
    zero_tail_bits: int = 0,
) -> "list[tuple[list[int], list[float], list[int]]] | None":
    """The two teacher-forcing views of one instance, or ``None`` if ineligible.

    Returns ``[(ids, numeric, span_positions), ...]`` for view E (all expanded) and
    view C (compact history, target expanded). ``span_positions`` are the indices of
    the TARGET span's 8 nibble tokens in ``ids`` (bos/wrapper offsets included), i.e.
    label positions: the NLL term for position ``p`` reads ``logits[p - 1]``.
    """
    values = [float(value) for value in constants]
    n_placeholders = sum(1 for token in skeleton if _is_constant_placeholder(token))
    if n_placeholders < 2 or len(values) != n_placeholders:
        return None

    has_wrappers = "<expression>" in tokenizer and "</expression>" in tokenizer
    views: list[tuple[list[int], list[float], list[int]]] = []
    masks = (
        [True] * n_placeholders,                       # view E: expanded history
        [False] * (n_placeholders - 1) + [True],       # view C: compacted history
    )
    for mask in masks:
        try:
            serialized, numeric = serialize_constant_tokens(
                skeleton, values,
                representation=CONSTANT_REPRESENTATION_IEEE754_MIXED,
                expanded_mask=mask, zero_tail_bits=zero_tail_bits,
            )
        except ValueError:
            return None
        tokens = list(serialized)
        nums = list(numeric)
        if has_wrappers:
            tokens = ["<expression>", *tokens, "</expression>"]
            nums = [float("nan"), *nums, float("nan")]
        body_ids = tokenizer.encode(tokens, oov="raise")
        ids = [int(tokenizer["<bos>"]), *[int(i) for i in body_ids], int(tokenizer["<eos>"])]
        nums = [float("nan"), *nums, float("nan")]
        if len(ids) > max_seq_len:
            return None
        # The TARGET span is the last <ieee754> opener; its nibbles follow directly.
        start_in_tokens = len(tokens) - 1 - tokens[::-1].index(IEEE754_START_TOKEN)
        span_positions = [1 + start_in_tokens + 1 + k for k in range(IEEE754_N_NIBBLES)]
        views.append((ids, nums, span_positions))
    return views


def paired_e3_gap(
    model: Any,
    tokenizer: Any,
    batch: "dict[str, Any]",
    device: Any,
    max_seq_len: int,
    zero_tail_bits: int = 0,
    max_instances: int = 64,
) -> "dict[str, float] | None":
    """Mean e3 gap over the eligible instances of one RAW (pre-collate) batch.

    ``None`` when no instance is eligible (fewer than two constants everywhere, or
    nothing fits the sequence budget).
    """
    skeletons = batch.get("skeleton")
    constants = batch.get("constants")
    if skeletons is None or constants is None:
        return None

    rows: list[tuple[int, list[tuple[list[int], list[float], list[int]]]]] = []
    for index, (skeleton, values) in enumerate(zip(skeletons, constants)):
        views = build_paired_views(list(skeleton), [float(v) for v in np.asarray(values).ravel()],
                                   tokenizer, max_seq_len, zero_tail_bits=zero_tail_bits)
        if views is not None:
            rows.append((index, views))
        if len(rows) >= max_instances:
            break
    if not rows:
        return None

    pad_id = int(tokenizer["<pad>"])
    x_tensors = torch.as_tensor(np.asarray([batch["x_tensors"][i] for i, _ in rows]), dtype=torch.float32)
    y_tensors = torch.as_tensor(np.asarray([batch["y_tensors"][i] for i, _ in rows]), dtype=torch.float32)
    attn_mask = torch.as_tensor(np.asarray([batch["data_attn_mask"][i] for i, _ in rows]), dtype=torch.float32)
    data = torch.cat([x_tensors, y_tensors], dim=-1).to(device)
    attn_mask = attn_mask.to(device)

    per_view_nll: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for view_index in (0, 1):
            sequences = [views[view_index][0] for _, views in rows]
            numerics = [views[view_index][1] for _, views in rows]
            length = max(len(sequence) for sequence in sequences)
            ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
            numeric = torch.full((len(sequences), length), float("nan"), dtype=torch.float32)
            for row, (sequence, values) in enumerate(zip(sequences, numerics)):
                ids[row, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
                numeric[row, :len(values)] = torch.as_tensor(values, dtype=torch.float32)
            ids = ids.to(device)
            numeric = numeric.to(device)
            logits = model(ids, data, input_num=numeric, data_attn_mask=attn_mask)
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            nlls = []
            for row, (_, views) in enumerate(rows):
                positions = views[view_index][2]
                nll = -sum(float(logprobs[row, p - 1, ids[row, p]]) for p in positions)
                nlls.append(nll)
            per_view_nll.append(nlls)

    gaps = [compacted - expanded for expanded, compacted in zip(per_view_nll[0], per_view_nll[1])]
    return {
        "e3_gap": float(np.mean(gaps)),
        "e3_n": float(len(gaps)),
        "e3_nll_expanded": float(np.mean(per_view_nll[0])),
        "e3_nll_compacted": float(np.mean(per_view_nll[1])),
    }
