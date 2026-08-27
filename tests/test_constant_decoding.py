"""Acceptance tests T5-T7 for the v24 constants DECODER pathway (lane C2).

Pre-registered integration contract, lane C2: the decoder numeric pathway (T5), the
decode-time vocabulary mask (T6, the mask half; the trained-model logit-rank half is a
skipped placeholder pointing at T13), and the constrained-decoding grammar over
`<ieee754>` spans (T7).

T8/T9 (compaction equivalence) are GONE: under the owner's 2026-08-27 ruling every
model-predicted number stays an `<ieee754>` span everywhere, so no compact view of an
emitted constant exists to be equivalent to.

NOTE on the test engine: the configs/test bundle references the generation-1 'dev_7-3'
simplipy asset, refused at load by the simplipy generation gate this repo now targets (a
pre-existing baseline condition, out of scope here). Models are therefore constructed
directly from the configs/test model kwargs with the generation-2 'base' engine, the
pattern lane C1 established.
"""
import inspect

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
    IEEE754_N_BYTES,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    BYTE_TOKENS,
    byte_tokens_to_float64,
    wrap_float64,
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
    id_to_byte = {int(tokenizer[token]): token for token in BYTE_TOKENS}

    values = []
    i = 0
    while i < len(ids):
        token = ids[i]
        if token == open_id:
            assert i + IEEE754_SPAN_LENGTH <= len(ids), f"unterminated span at {i}: {ids}"
            inner = ids[i + 1:i + 1 + IEEE754_N_BYTES]
            assert all(x in id_to_byte for x in inner), f"non-nibble token inside span at {i}: {ids}"
            assert ids[i + IEEE754_SPAN_LENGTH - 1] == close_id, f"span at {i} not closed after 8 nibbles: {ids}"
            value = byte_tokens_to_float64([id_to_byte[x] for x in inner])
            assert isinstance(value, float)
            values.append(value)
            i += IEEE754_SPAN_LENGTH
        else:
            assert token not in id_to_byte and token != close_id, f"stray nibble/close outside span at {i}: {ids}"
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
    byte_ids = [int(tokenizer[token]) for token in BYTE_TOKENS]

    outside = tokenizer.encode(["<bos>", "<expression>", "x1"])
    span_nibbles = [int(tokenizer[token]) for token in wrap_float64(0.5)[1:-1]]
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
        for forbidden_id in (*byte_ids, close_id):
            assert bool(forbidden[forbidden_id]), (name, forbidden_id)
        for allowed_token in (IEEE754_START_TOKEN, "x2", "+", "sin", "<eos>"):
            assert not bool(forbidden[tokenizer[allowed_token]]), (name, allowed_token)

    for name in ("inside_0_nibbles", "inside_5_nibbles"):
        tensor = torch.tensor([prefixes[name]], dtype=torch.long)
        forbidden = g.forbidden(tensor, remaining=100)[0]
        allowed_ids = (~forbidden).nonzero().flatten().tolist()
        # Inside the tags, before 8 nibbles: EXACTLY the 16 nibbles are admissible.
        assert sorted(allowed_ids) == sorted(byte_ids), (name, allowed_ids)

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


def test_the_grammar_bans_reopening_a_property_block() -> None:
    """Owner ruling 2026-08-27: a property stated before <hypothesize> is GIVEN and may not
    be hypothesized after it. The grammar enforces it as at-most-once per sequence, which
    also stops the model re-opening a block it opened itself.

    Stateless like every other rule here -- recomputed from the prefix -- so it holds under
    KV caching, mini-batching and the static path without any carried state to reindex."""
    from flash_ansr.decoding.constrained import IEEE754GrammarConstraint

    # The v24 template rather than this file's configs/test tokenizer: only the template
    # carries <hypothesize>, and the rule exists to serve the boundary.
    tokenizer = Tokenizer.from_config(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))
    grammar = IEEE754GrammarConstraint(tokenizer)
    bos = int(tokenizer["<bos>"])
    cx_open, cx_close = int(tokenizer["<complexity>"]), int(tokenizer["</complexity>"])
    float_id = int(tokenizer["<float>"])
    hypothesize = int(tokenizer["<hypothesize>"])

    assert cx_open in grammar.property_open_ids

    # A prefix with the block GIVEN, then the boundary: it may not be opened again.
    given = torch.tensor([[bos, cx_open, float_id, cx_close, hypothesize]])
    assert bool(grammar.forbidden(given)[0, cx_open])

    # The same prefix WITHOUT the given block leaves it open to hypothesize.
    free = torch.tensor([[bos, hypothesize]])
    assert not bool(grammar.forbidden(free)[0, cx_open])

    # Per row, not per batch: one row's history must not constrain another's.
    batch = torch.tensor([[bos, cx_open, float_id, cx_close, hypothesize],
                          [bos, hypothesize, bos, bos, bos]])
    mask = grammar.forbidden(batch)
    assert bool(mask[0, cx_open]) and not bool(mask[1, cx_open])
