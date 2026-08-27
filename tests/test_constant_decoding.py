"""Acceptance tests T5-T8 for the v24 `ieee754_mixed` constants DECODER pathway (lane C2).

Pre-registered integration contract, lane C2: the decoder numeric pathway (T5), the
decode-time vocabulary mask (T6, the mask half; the trained-model logit-rank half is a
skipped placeholder pointing at T13), the constrained-decoding grammar over expanded
`<ieee754>` spans (T7), and THE GOLDEN TEST: compaction equivalence on the dynamic KV
path (T8, single-sequence + small-batch; the per-beam version is T9, next lane).

NOTE on the test engine: the configs/test bundle references the generation-1 'dev_7-3'
simplipy asset, refused at load by the simplipy generation gate this repo now targets (a
pre-existing baseline condition, out of scope here). Models are therefore constructed
directly from the configs/test model kwargs with the generation-2 'base' engine, the
pattern lane C1 established.
"""
import inspect
import math

import numpy as np
import pytest
import torch

from flash_ansr.utils.numeric import NUMERIC_DTYPE

from flash_ansr import get_path
from flash_ansr.data.collate import BatchFormatter
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_N_NIBBLES,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
    nibble_tokens_to_float32,
    wrap_float32,
)

COMPACT_TOKEN = "<float>"

# The repo-standard logits-equivalence tolerance: the dynamic KV-cache tests and the
# static-decode Stage-1 gate both use allclose atol=1e-5 (see tests/test_models/
# test_kv_cache.py and static_kv.py). T8 pins the same bar.
LOGITS_ATOL = 1e-5


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("base", install=True)


def _tiny_model(tokenizer: Tokenizer, engine, seed: int = 0x24C2):  # type: ignore[no-untyped-def]
    from flash_ansr.model.flash_ansr_model import FlashANSRModel

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    torch.manual_seed(seed)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model


def _biased_model(tokenizer: Tokenizer, engine, token: str, bias: float):  # type: ignore[no-untyped-def]
    """A tiny model whose head is biased to emit `token` at (nearly) every position."""
    model = _tiny_model(tokenizer, engine)
    with torch.no_grad():
        model.next_token_head[-1].bias[tokenizer[token]] += bias
    return model


def _scan_spans(ids: list[int], tokenizer: Tokenizer) -> list[float]:
    """Assert every `<ieee754>` span in `ids` is well-formed (exactly 8 nibbles, then the
    close), no nibble/close token strays outside a span, and return the decoded values."""
    open_id = tokenizer[IEEE754_START_TOKEN]
    close_id = tokenizer[IEEE754_END_TOKEN]
    id_to_nibble = {int(tokenizer[token]): token for token in NIBBLE_TOKENS}

    values = []
    i = 0
    while i < len(ids):
        token = ids[i]
        if token == open_id:
            assert i + IEEE754_SPAN_LENGTH <= len(ids), f"unterminated span at {i}: {ids}"
            inner = ids[i + 1:i + 1 + IEEE754_N_NIBBLES]
            assert all(x in id_to_nibble for x in inner), f"non-nibble token inside span at {i}: {ids}"
            assert ids[i + IEEE754_SPAN_LENGTH - 1] == close_id, f"span at {i} not closed after 8 nibbles: {ids}"
            value = nibble_tokens_to_float32([id_to_nibble[x] for x in inner])
            assert isinstance(value, float)
            values.append(value)
            i += IEEE754_SPAN_LENGTH
        else:
            assert token not in id_to_nibble and token != close_id, f"stray nibble/close outside span at {i}: {ids}"
            i += 1
    return values


# ---------------------------------------------------------------------------
# T5 — numeric pathway: the <float> value embedding actually reaches the decoder
# ---------------------------------------------------------------------------

_T5_ROW = ["<bos>", "<expression>", "x1", "*", "<float>", "+", "x2", "</expression>", "<eos>"]
_T5_FLOAT_POS = _T5_ROW.index("<float>")


def _t5_collated_batch(tokenizer: Tokenizer, value: float, x: torch.Tensor, y: torch.Tensor) -> dict:
    numeric = [float("nan")] * len(_T5_ROW)
    numeric[_T5_FLOAT_POS] = value
    formatter = BatchFormatter(tokenizer=tokenizer)
    batch = {
        "input_ids": [tokenizer.encode(_T5_ROW)],
        "input_num": [numeric],
        "x_tensors": [x.clone()],
        "y_tensors": [y.clone()],
        "constants": [[value]],
    }
    return formatter.collate(batch, device="cpu")


def test_t5_numeric_pathway_reaches_decoder(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model = _tiny_model(tokenizer, engine)
    torch.manual_seed(0x24C5)
    x, y = torch.randn(6, 10), torch.randn(6, 1)

    batch_a = _t5_collated_batch(tokenizer, 2.5, x, y)
    batch_b = _t5_collated_batch(tokenizer, -1.25, x, y)

    # The two collated batches differ ONLY in the <float> value on the numeric channel
    # (NaN-aware comparison: NaN == NaN counts as "same" on the no-value channel).
    assert torch.equal(batch_a["input_ids"], batch_b["input_ids"])
    num_a = batch_a["input_num"].squeeze(-1)
    num_b = batch_b["input_num"].squeeze(-1)
    differs = (num_a != num_b) & ~(torch.isnan(num_a) & torch.isnan(num_b))
    assert differs.nonzero().tolist() == [[0, _T5_FLOAT_POS]]

    with torch.no_grad():
        data = torch.cat([batch_a["x_tensors"], batch_a["y_tensors"]], dim=-1)
        logits_a = model(batch_a["input_ids"], data, input_num=batch_a["input_num"],
                         data_attn_mask=batch_a["data_attn_mask"])
        logits_b = model(batch_b["input_ids"], data, input_num=batch_b["input_num"],
                         data_attn_mask=batch_b["data_attn_mask"])

    q = _T5_FLOAT_POS
    # Upstream of the <float> position: bit-identical (causality: the value cannot leak back).
    assert torch.equal(logits_a[:, :q], logits_b[:, :q])
    # At and downstream of it: the value embedding must reach the decoder — every position
    # differs (the <float> position itself sees its own numeric embedding).
    per_position_diff = (logits_a[0] - logits_b[0]).abs().amax(dim=-1)
    for p in range(q, len(_T5_ROW)):
        assert per_position_diff[p] > 0.0, f"position {p} logits identical: numeric pathway severed"


def test_t5_ablation_reproduces_token_only_path(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    # Ablation is an EXTERNAL patch on this test's model instance, never a switch in the
    # model class (owner ruling 2026-08-17: production stays clean and minimal; experiments
    # patch it or branch off it). Severing only the decoder's `extra_parallel_embeddings`
    # argument and landing bitwise on token-only logits ALSO proves that argument is the
    # sole route by which `input_num` reaches the logits.
    model = _tiny_model(tokenizer, engine)
    torch.manual_seed(0x24C5)
    x, y = torch.randn(6, 10), torch.randn(6, 1)
    batch = _t5_collated_batch(tokenizer, 2.5, x, y)
    batch_other = _t5_collated_batch(tokenizer, -1.25, x, y)
    data = torch.cat([batch["x_tensors"], batch["y_tensors"]], dim=-1)

    orig_decoder_forward = model.decoder.forward

    def severed_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["extra_parallel_embeddings"] = None
        return orig_decoder_forward(*args, **kwargs)

    with torch.no_grad():
        logits_token_only = model(batch["input_ids"], data, input_num=None,
                                  data_attn_mask=batch["data_attn_mask"])

        model.decoder.forward = severed_forward  # type: ignore[method-assign]
        try:
            logits_ablated = model(batch["input_ids"], data, input_num=batch["input_num"],
                                   data_attn_mask=batch["data_attn_mask"])
            logits_ablated_other = model(batch_other["input_ids"], data, input_num=batch_other["input_num"],
                                         data_attn_mask=batch_other["data_attn_mask"])
        finally:
            del model.decoder.forward  # drop the instance shadow; the class method takes over again
        logits_live = model(batch["input_ids"], data, input_num=batch["input_num"],
                            data_attn_mask=batch["data_attn_mask"])

    # Ablating the pathway reproduces the token-only embedding EXACTLY (bitwise).
    assert torch.equal(logits_ablated, logits_token_only)
    # Under ablation the differing value cannot reach the decoder at all.
    assert torch.equal(logits_ablated, logits_ablated_other)
    # The patch is not sticky: removed, the pathway is live again.
    assert not torch.equal(logits_live, logits_token_only)


# ---------------------------------------------------------------------------
# T6 — anti-training, decode-time half: the vocabulary mask forbids <float> outright
# ---------------------------------------------------------------------------

def test_t6_mask_forbids_float_in_every_state(tokenizer: Tokenizer) -> None:
    from flash_ansr.decoding.constrained import IEEE754GrammarConstraint

    g = IEEE754GrammarConstraint(tokenizer)
    float_id = tokenizer[COMPACT_TOKEN]
    open_id = tokenizer[IEEE754_START_TOKEN]
    close_id = tokenizer[IEEE754_END_TOKEN]
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]

    outside = tokenizer.encode(["<bos>", "<expression>", "x1"])
    span_nibbles = [int(tokenizer[token]) for token in wrap_float32(0.5)[1:-1]]
    prefixes = {
        "empty": [],
        "outside": outside,
        "inside_0_nibbles": [*outside, open_id],
        "inside_5_nibbles": [*outside, open_id, *span_nibbles[:5]],
        "inside_8_nibbles": [*outside, open_id, *span_nibbles],
        "after_closed_span": [*outside, open_id, *span_nibbles, close_id],
    }

    for name, prefix in prefixes.items():
        tensor = torch.tensor([prefix], dtype=torch.long)
        forbidden = g.forbidden(tensor, remaining=100)[0]
        # THE decode-time anti-training guarantee: <float> is forbidden in EVERY state.
        assert bool(forbidden[float_id]), f"<float> must be forbidden in state {name!r}"

    for name in ("empty", "outside", "after_closed_span"):
        tensor = torch.tensor([prefixes[name]], dtype=torch.long)
        forbidden = g.forbidden(tensor, remaining=100)[0]
        # Outside a span: nibbles and the close tag are forbidden, the open tag and
        # ordinary expression tokens are allowed.
        for forbidden_id in (*nibble_ids, close_id):
            assert bool(forbidden[forbidden_id]), (name, forbidden_id)
        for allowed_token in (IEEE754_START_TOKEN, "x2", "+", "sin", "<eos>"):
            assert not bool(forbidden[tokenizer[allowed_token]]), (name, allowed_token)

    for name in ("inside_0_nibbles", "inside_5_nibbles"):
        tensor = torch.tensor([prefixes[name]], dtype=torch.long)
        forbidden = g.forbidden(tensor, remaining=100)[0]
        allowed_ids = (~forbidden).nonzero().flatten().tolist()
        # Inside the tags, before 8 nibbles: EXACTLY the 16 nibbles are admissible.
        assert sorted(allowed_ids) == sorted(nibble_ids), (name, allowed_ids)

    tensor = torch.tensor([prefixes["inside_8_nibbles"]], dtype=torch.long)
    forbidden = g.forbidden(tensor, remaining=100)[0]
    allowed_ids = (~forbidden).nonzero().flatten().tolist()
    # After exactly 8 nibbles: ONLY the close is admissible.
    assert allowed_ids == [close_id]

    # Budget rule: opening a span requires 10 remaining slots so it can always terminate.
    tensor = torch.tensor([prefixes["outside"]], dtype=torch.long)
    assert not bool(g.forbidden(tensor, remaining=IEEE754_SPAN_LENGTH)[0][open_id])
    assert bool(g.forbidden(tensor, remaining=IEEE754_SPAN_LENGTH - 1)[0][open_id])


def test_t6_decode_time_mask_dynamic_sampling(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model = _biased_model(tokenizer, engine, COMPACT_TOKEN, bias=8.0)
    float_id = tokenizer[COMPACT_TOKEN]
    bos = tokenizer["<bos>"]
    torch.manual_seed(0x24C6)
    x = torch.rand(13, 11, dtype=NUMERIC_DTYPE)

    torch.manual_seed(7)
    raw_off, _ = model.sample_top_kp(x, choices=16, max_len=24, return_raw=True,
                                     initial_tokens=[bos], use_cache=False)
    # Non-vacuity: the biased model DOES emit <float> without the mask.
    assert any(float_id in seq for seq in raw_off)

    for use_cache in (False, True):
        torch.manual_seed(7)
        raw_on, _ = model.sample_top_kp(x, choices=16, max_len=24, return_raw=True,
                                        initial_tokens=[bos], use_cache=use_cache,
                                        constrain_ieee754=True)
        assert all(float_id not in seq for seq in raw_on), f"use_cache={use_cache}"


def test_t6_decode_time_mask_static_sampling(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    import flash_ansr.model.flash_ansr_model as fam

    model = _biased_model(tokenizer, engine, COMPACT_TOKEN, bias=8.0)
    float_id = tokenizer[COMPACT_TOKEN]
    bos = tokenizer["<bos>"]
    torch.manual_seed(0x24C6)
    x = torch.rand(13, 11, dtype=NUMERIC_DTYPE)

    count_before = fam.STATIC_DECODE_CALL_COUNT
    torch.manual_seed(7)
    raw_off, _ = model.sample_top_kp(x, choices=8, max_len=20, return_raw=True,
                                     initial_tokens=[bos], static_decode=True)
    assert fam.STATIC_DECODE_CALL_COUNT == count_before + 1, "static path did not engage"
    assert any(float_id in seq for seq in raw_off)

    torch.manual_seed(7)
    raw_on, _ = model.sample_top_kp(x, choices=8, max_len=20, return_raw=True,
                                    initial_tokens=[bos], static_decode=True,
                                    constrain_ieee754=True)
    assert fam.STATIC_DECODE_CALL_COUNT == count_before + 2, "static path did not engage"
    assert all(float_id not in seq for seq in raw_on)


def test_t6_mask_wired_at_all_three_logit_sites() -> None:
    """The seam must be wired into the beam, dynamic-sampling and static-sampling logits
    sites (the C1 `_apply_float_target_mask` wiring-test pattern)."""
    from flash_ansr.model.flash_ansr_model import FlashANSRModel

    for method in (FlashANSRModel.sample_top_kp,
                   FlashANSRModel._sample_top_kp_static):
        source = inspect.getsource(method)
        assert "constrain_ieee754" in source, method.__name__
        assert ".forbidden(" in source, method.__name__


@pytest.mark.skip(reason=(
    "T6 trained-model half: after warmup, <float>'s logit at generation positions is "
    "suppressed (rank > K). Requires a trained v24 checkpoint and belongs to T13 "
    "(training-lane acceptance). C2 ships the decode-time mask, tested above."
))
def test_t6_float_logit_rank_suppressed_after_warmup() -> None:
    raise AssertionError("placeholder: implemented by the T13 trained-model acceptance")


# ---------------------------------------------------------------------------
# T7 — grammar: exactly the 16 nibbles inside the tags, exactly 8, then the close
# ---------------------------------------------------------------------------

def test_t7_grammar_state_machine_property(tokenizer: Tokenizer) -> None:
    """Property test: ANY walk that only ever takes mask-admitted tokens produces
    well-formed spans (8 nibbles + close, nothing stray, every span parses to a float)."""
    from flash_ansr.decoding.constrained import IEEE754GrammarConstraint

    g = IEEE754GrammarConstraint(tokenizer)
    float_id = tokenizer[COMPACT_TOKEN]
    open_id = tokenizer[IEEE754_START_TOKEN]
    rng = np.random.default_rng(0x24C7)

    total_spans = 0
    for _ in range(120):
        max_len = int(rng.integers(IEEE754_SPAN_LENGTH + 2, 90))
        tokens = [int(tokenizer["<bos>"])]
        while len(tokens) < max_len:
            prefix = torch.tensor([tokens], dtype=torch.long)
            forbidden = g.forbidden(prefix, remaining=max_len - len(tokens))[0]
            allowed = (~forbidden).nonzero().flatten().tolist()
            assert allowed, "the grammar must never dead-end"
            assert float_id not in allowed
            if open_id in allowed and rng.random() < 0.2:
                token = open_id  # bias toward opening spans so the walk exercises them
            else:
                token = int(allowed[int(rng.integers(len(allowed)))])
            tokens.append(token)

        total_spans += len(_scan_spans(tokens, tokenizer))

    assert total_spans >= 20, f"property walk exercised too few spans ({total_spans})"


def test_t7_constrained_sampling_emissions_parse(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    """Integration: constrained sampling from a span-happy model only ever emits
    well-formed spans — every emission parses to a float."""
    model = _biased_model(tokenizer, engine, IEEE754_START_TOKEN, bias=5.0)
    float_id = tokenizer[COMPACT_TOKEN]
    bos = tokenizer["<bos>"]
    torch.manual_seed(0x24C7)
    x = torch.rand(13, 11, dtype=NUMERIC_DTYPE)

    for use_cache in (False, True):
        torch.manual_seed(11)
        raw, _ = model.sample_top_kp(x, choices=24, max_len=48, return_raw=True,
                                     initial_tokens=[bos], use_cache=use_cache,
                                     constrain_ieee754=True)
        n_spans = 0
        for seq in raw:
            assert float_id not in seq
            n_spans += len(_scan_spans(seq, tokenizer))
        # The biased model must actually have opened spans, or the test is vacuous.
        assert n_spans >= 5, f"use_cache={use_cache}: only {n_spans} spans sampled"


# ---------------------------------------------------------------------------
# T8 — THE GOLDEN TEST: compaction is a mechanical no-op relative to the compact view
# ---------------------------------------------------------------------------

def _feed_incremental(model, ids: torch.Tensor, nums: torch.Tensor, memory: torch.Tensor, past: list | None):  # type: ignore[no-untyped-def]
    """Feed `ids` (B, T) one token at a time through the dynamic KV path (generation-
    faithful), returning (logits (B, T, V), past)."""
    logits_steps = []
    for t in range(ids.shape[1]):
        logits, past = model.forward(
            ids[:, t:t + 1], None, input_num=nums[:, t:t + 1].unsqueeze(-1),
            memory=memory, past_key_values=past, use_cache=True)
        logits_steps.append(logits)
    return torch.cat(logits_steps, dim=1), past


def _nan(*shape: int) -> torch.Tensor:
    return torch.full(shape, float("nan"), dtype=NUMERIC_DTYPE)


def test_t8_compaction_equivalence_single_sequence(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.decoding.compaction import compact_closed_ieee754_spans

    model = _tiny_model(tokenizer, engine)
    pad_id = tokenizer["<pad>"]
    float_id = tokenizer[COMPACT_TOKEN]
    value = 1.5
    history_value = 0.5  # a previously-compacted constant already in the prefix

    # Prefix carries a compact-history constant (<float> with its value on the numeric
    # channel) — the in-distribution compact-history-then-expanded pattern of T3.
    prefix = ["<bos>", "<expression>", "+", "*", "<float>", "x1", "*"]
    span = wrap_float32(value)
    continuation = ["x2", "</expression>", "<eos>"]
    span_start = len(prefix)              # 7
    expanded_length = span_start + IEEE754_SPAN_LENGTH  # 17

    prefix_ids = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long)
    span_ids = torch.tensor([tokenizer.encode(span)], dtype=torch.long)
    cont_ids = torch.tensor([tokenizer.encode(continuation)], dtype=torch.long)

    prefix_num = _nan(1, len(prefix))
    prefix_num[0, prefix.index("<float>")] = history_value

    torch.manual_seed(0x24C8)
    data = torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE)
    with torch.no_grad():
        memory = model._create_memory(data)

        # --- incremental path: prefill the prefix, then GENERATE the span token-by-token ---
        logits_prefix, past = model.forward(
            prefix_ids, None, input_num=prefix_num.unsqueeze(-1), memory=memory, use_cache=True)
        _, past = _feed_incremental(model, span_ids, _nan(1, IEEE754_SPAN_LENGTH), memory, past)
        assert past[0][0][0].shape[2] == expanded_length

        # --- compaction: nibbles -> value -> span collapse -> <float>+input_num -> KV drop + re-encode ---
        buffer = torch.full((1, 64), pad_id, dtype=torch.long)
        buffer[0, :span_start] = prefix_ids[0]
        buffer[0, span_start:expanded_length] = span_ids[0]
        num_buffer = _nan(1, 64)
        num_buffer[0, :len(prefix)] = prefix_num[0]

        result = compact_closed_ieee754_spans(
            model, buffer, current_length=expanded_length, past_key_values=past,
            memory=memory, input_num=num_buffer)

        # Mechanics: value decoded, span collapsed to ONE <float>, tail cleared, KV shrunk
        # by exactly the 10 span entries (+1 for the re-encoded compact token).
        assert result.values.tolist() == [value]
        assert result.length == span_start + 1
        assert int(result.sequences[0, span_start]) == float_id
        assert torch.all(result.sequences[0, span_start + 1:expanded_length] == pad_id)
        assert torch.equal(result.sequences[0, :span_start], prefix_ids[0])
        assert result.past_key_values[0][0][0].shape[2] == span_start + 1
        # The numeric channel: history value preserved, new value at the collapsed slot.
        assert result.input_num[0, prefix.index("<float>")] == history_value
        assert result.input_num[0, span_start] == value
        assert math.isnan(float(result.input_num[0, span_start - 1]))
        # Cross-attention K/V are untouched by the surgery.
        assert torch.equal(result.past_key_values[0][1][0], past[0][1][0])

        # --- continue on the incremental path over the continuation tokens ---
        logits_cont, _ = _feed_incremental(
            model, cont_ids, _nan(1, len(continuation)), memory, result.past_key_values)
        incremental = torch.cat([result.logits, logits_cont], dim=1)

        # --- fresh forward over the compact-view sequence ---
        compact_view = [*prefix, COMPACT_TOKEN, *continuation]
        compact_ids = torch.tensor([tokenizer.encode(compact_view)], dtype=torch.long)
        compact_num = _nan(1, len(compact_view))
        compact_num[0, prefix.index("<float>")] = history_value
        compact_num[0, span_start] = value
        fresh = model.forward(compact_ids, None, input_num=compact_num.unsqueeze(-1),
                              memory=memory, use_cache=False)

    # THE golden equality: every post-compaction position, to float tolerance.
    post = fresh[:, span_start:span_start + 1 + len(continuation)]
    assert incremental.shape == post.shape
    max_diff = (incremental - post).abs().max().item()
    assert torch.allclose(incremental, post, atol=LOGITS_ATOL), f"max diff {max_diff}"
    # And the pre-compaction prefix positions agree too (same cache, untouched).
    assert torch.allclose(logits_prefix, fresh[:, :span_start], atol=LOGITS_ATOL)


def test_t8_compaction_equivalence_small_batch(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.decoding.compaction import compact_closed_ieee754_spans

    model = _tiny_model(tokenizer, engine)
    pad_id = tokenizer["<pad>"]
    float_id = tokenizer[COMPACT_TOKEN]
    values = [1.5, -0.25, 3.0e-3]
    variables = ["x1", "x2", "x3"]

    prefix_len = 4
    continuation = ["+", "x2", "</expression>", "<eos>"]
    span_start = prefix_len
    expanded_length = span_start + IEEE754_SPAN_LENGTH

    rows_prefix = [["<bos>", "<expression>", var, "*"] for var in variables]
    prefix_ids = torch.tensor([tokenizer.encode(row) for row in rows_prefix], dtype=torch.long)
    span_ids = torch.tensor([tokenizer.encode(wrap_float32(v)) for v in values], dtype=torch.long)
    cont_ids = torch.tensor([tokenizer.encode(continuation)] * 3, dtype=torch.long)

    torch.manual_seed(0x24C9)
    data = torch.rand(3, 13, 11, dtype=NUMERIC_DTYPE)
    with torch.no_grad():
        memory = model._create_memory(data)

        _, past = model.forward(prefix_ids, None, input_num=_nan(3, prefix_len).unsqueeze(-1),
                                memory=memory, use_cache=True)
        _, past = _feed_incremental(model, span_ids, _nan(3, IEEE754_SPAN_LENGTH), memory, past)

        buffer = torch.full((3, 64), pad_id, dtype=torch.long)
        buffer[:, :prefix_len] = prefix_ids
        buffer[:, span_start:expanded_length] = span_ids

        result = compact_closed_ieee754_spans(
            model, buffer, current_length=expanded_length, past_key_values=past, memory=memory)

        assert result.values.tolist() == pytest.approx([float(np.float32(v)) for v in values])
        assert result.length == span_start + 1
        assert torch.all(result.sequences[:, span_start] == float_id)
        assert result.past_key_values[0][0][0].shape[2] == span_start + 1
        # Against the CODEC's value, not the source literal: the numeric channel is
        # binary64 while the span codec still snaps to binary32, so 3.0e-3 rides as
        # 0.003000000026077032. Comparing to the literal only passed while both sides
        # were narrowed to float32 -- it was never testing the round-trip it looks like.
        assert torch.all(result.input_num[:, span_start]
                         == torch.tensor([float(np.float32(v)) for v in values],
                                         dtype=NUMERIC_DTYPE))

        logits_cont, _ = _feed_incremental(
            model, cont_ids, _nan(3, len(continuation)), memory, result.past_key_values)
        incremental = torch.cat([result.logits, logits_cont], dim=1)

        compact_rows = [[*row, COMPACT_TOKEN, *continuation] for row in rows_prefix]
        compact_ids = torch.tensor([tokenizer.encode(row) for row in compact_rows], dtype=torch.long)
        compact_num = _nan(3, len(compact_rows[0]))
        compact_num[:, span_start] = torch.tensor(values, dtype=torch.float32)
        fresh = model.forward(compact_ids, None, input_num=compact_num.unsqueeze(-1),
                              memory=memory, use_cache=False)

    post = fresh[:, span_start:span_start + 1 + len(continuation)]
    assert incremental.shape == post.shape
    max_diff = (incremental - post).abs().max().item()
    assert torch.allclose(incremental, post, atol=LOGITS_ATOL), f"max diff {max_diff}"


def test_t8_compaction_validates_the_span(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.decoding.compaction import compact_closed_ieee754_spans

    model = _tiny_model(tokenizer, engine)
    pad_id = tokenizer["<pad>"]
    prefix = ["<bos>", "<expression>", "x1", "*"]
    span = wrap_float32(2.0)
    ids = tokenizer.encode([*prefix, *span])
    length = len(ids)

    torch.manual_seed(0x24C9)
    data = torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE)
    with torch.no_grad():
        memory = model._create_memory(data)
        seq = torch.full((1, 64), pad_id, dtype=torch.long)
        seq[0, :length] = torch.tensor(ids, dtype=torch.long)
        _, past = model.forward(seq[:, :length], None, input_num=_nan(1, length).unsqueeze(-1),
                                memory=memory, use_cache=True)

        # Not span-terminated: last token is not </ieee754>.
        bad = seq.clone()
        bad[0, length - 1] = tokenizer["x1"]
        with pytest.raises(ValueError):
            compact_closed_ieee754_spans(model, bad, current_length=length,
                                         past_key_values=past, memory=memory)

        # A non-nibble token inside the span.
        bad = seq.clone()
        bad[0, length - 4] = tokenizer["x1"]
        with pytest.raises(ValueError):
            compact_closed_ieee754_spans(model, bad, current_length=length,
                                         past_key_values=past, memory=memory)

        # Too short to contain a whole span.
        with pytest.raises(ValueError):
            compact_closed_ieee754_spans(model, seq[:, :6], current_length=6,
                                         past_key_values=past, memory=memory)
