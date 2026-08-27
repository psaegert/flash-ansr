"""Per-row positions on the static KV path, and span compaction by POSITION REWIND.

`forward_static` took a scalar `position`, which forced every row of a chunk to sit at the
same slot. That is the only reason compaction lived on the beam path: compaction
desynchronizes rows (each closes its span at its own step), so the dynamic cat-grow path
grew a batch-1 cache per beam to cope. With a per-row `(batch,)` position tensor the static
path needs no cache surgery at all -- a compacting row just rewinds its own index to
`span_start`, stale slots beyond it fall outside `attend_mask`, and later writes overwrite
them in place.

Shapes never change, so this stays graph-capturable: a CUDA graph forbids SHAPE changes,
not content changes, and both the position tensor and the mask keep a fixed shape.

The bar is the repo-standard logits-equivalence tolerance (atol 1e-5) that the dynamic
KV-cache tests and the static-decode Stage-1 gate already use.
"""
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.decoders.static_kv import StaticKVCache
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import IEEE754_SPAN_LENGTH, wrap_float32
from flash_ansr.utils.numeric import NUMERIC_DTYPE

LOGITS_ATOL = 1e-5
COMPACT_TOKEN = "<float>"
NAN = float("nan")


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("base", install=True)


@pytest.fixture(scope="module")
def model(tokenizer, engine):  # type: ignore[no-untyped-def]
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    torch.manual_seed(0x5217)
    return FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs).eval()


@pytest.fixture(scope="module")
def memory(model):  # type: ignore[no-untyped-def]
    with torch.no_grad():
        return model._create_memory(torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE))


def _cache(model, batch: int, max_len: int) -> StaticKVCache:
    layer = model.decoder.layers[0]
    return StaticKVCache(
        n_layers=len(model.decoder.layers), batch=batch,
        n_heads=layer.self_attention.n_heads, head_dim=layer.self_attention.head_dim,
        max_len=max_len, device=torch.device("cpu"), dtype=torch.float32)


def _step(model, memory, cache, position, tokens, values):
    """One static decode step over a batch of single tokens."""
    ids = torch.tensor(tokens).unsqueeze(-1)
    num = torch.tensor(values, dtype=NUMERIC_DTYPE).unsqueeze(-1).unsqueeze(-1)
    return model.forward_static(ids, num, memory, cache, position=position)


def _compact_view_logits(model, tokenizer, memory, plan):
    """Reference: one fresh full forward over the COMPACT view of `plan`."""
    ids = torch.tensor([[tokenizer[token] for token, _ in plan]])
    num = torch.tensor([[value for _, value in plan]], dtype=NUMERIC_DTYPE)
    with torch.no_grad():
        return model.forward(ids, None, input_num=num.unsqueeze(-1), memory=memory)[0, -1, :]


def _decode_with_rewind(model, tokenizer, memory, prefix, value, tail, max_len=64):
    """Emit the EXPANDED span, then compact it by rewinding to span_start."""
    cache = _cache(model, 1, max_len)
    pos = torch.zeros(1, dtype=torch.long)
    logits = None
    with torch.no_grad():
        for token in [*prefix, *wrap_float32(value)]:
            _step(model, memory, cache, pos.clone(), [tokenizer[token]], [NAN])
            pos += 1
        pos[0] = len(prefix)                                   # <-- the compaction event
        _step(model, memory, cache, pos.clone(), [tokenizer[COMPACT_TOKEN]], [value])
        pos += 1
        for token in tail:
            logits = _step(model, memory, cache, pos.clone(), [tokenizer[token]], [NAN])
            pos += 1
    return logits[0, -1, :]


def test_a_per_row_position_tensor_is_bit_identical_to_the_scalar(model, tokenizer, memory):
    """The new path must not perturb the existing one when every row is aligned."""
    tokens = ["<bos>", "<expression>", "x1", "+", "x2"]
    out = []
    for as_tensor in (False, True):
        cache = _cache(model, 1, 32)
        logits = None
        with torch.no_grad():
            for step, token in enumerate(tokens):
                pos = torch.full((1,), step, dtype=torch.long) if as_tensor else step
                logits = _step(model, memory, cache, pos, [tokenizer[token]], [NAN])
        out.append(logits[0, -1, :])
    assert torch.equal(out[0], out[1]), "the tensor path must be bit-identical, not merely close"


def test_compaction_by_rewind_matches_a_fresh_compact_forward(model, tokenizer, memory):
    """THE GOLDEN EQUALITY, on the static path: emitting an expanded span and then rewinding
    must leave the model in exactly the state a compact-view forward would have produced."""
    prefix, value, tail = ["<bos>", "<expression>", "x1", "*"], 1.5, ["+", "x2"]
    assert len(wrap_float32(value)) == IEEE754_SPAN_LENGTH
    reference = _compact_view_logits(model, tokenizer, memory, [
        *[(t, NAN) for t in prefix], (COMPACT_TOKEN, value), *[(t, NAN) for t in tail]])
    got = _decode_with_rewind(model, tokenizer, memory, prefix, value, tail)
    torch.testing.assert_close(got, reference, atol=LOGITS_ATOL, rtol=0)


def test_the_golden_equality_is_not_vacuous(model, tokenizer, memory):
    """Control: WITHOUT the rewind the same decode must disagree. If this passes, the test
    above is proving nothing."""
    prefix, value, tail = ["<bos>", "<expression>", "x1", "*"], 1.5, ["+", "x2"]
    reference = _compact_view_logits(model, tokenizer, memory, [
        *[(t, NAN) for t in prefix], (COMPACT_TOKEN, value), *[(t, NAN) for t in tail]])
    cache = _cache(model, 1, 64)
    pos = torch.zeros(1, dtype=torch.long)
    logits = None
    with torch.no_grad():
        for token in [*prefix, *wrap_float32(value), *tail]:
            logits = _step(model, memory, cache, pos.clone(), [tokenizer[token]], [NAN])
            pos += 1
    assert (logits[0, -1, :] - reference).abs().max().item() > LOGITS_ATOL


def test_a_continuation_longer_than_the_span_walks_back_over_every_stale_slot(
        model, tokenizer, memory):
    """The span leaves 10 stale slots behind. A 12-token tail steps through all of them, so
    any that were readable rather than overwritten would show up here."""
    prefix, value = ["<bos>", "<expression>", "x1", "*"], 1.5
    tail = ["+", "x2", "*", "x3", "+", "x1", "*", "x2", "+", "x3", "-", "x1"]
    assert len(tail) > IEEE754_SPAN_LENGTH
    reference = _compact_view_logits(model, tokenizer, memory, [
        *[(t, NAN) for t in prefix], (COMPACT_TOKEN, value), *[(t, NAN) for t in tail]])
    got = _decode_with_rewind(model, tokenizer, memory, prefix, value, tail)
    torch.testing.assert_close(got, reference, atol=LOGITS_ATOL, rtol=0)


def test_two_spans_rewind_twice_over_recycled_slots(model, tokenizer, memory):
    """The second span is written into slots the first span already used and released."""
    v1, v2 = 1.5, -0.25
    reference = _compact_view_logits(model, tokenizer, memory, [
        ("<bos>", NAN), ("<expression>", NAN), ("x1", NAN), ("*", NAN), (COMPACT_TOKEN, v1),
        ("+", NAN), ("x2", NAN), ("*", NAN), (COMPACT_TOKEN, v2), ("+", NAN), ("x3", NAN)])

    cache = _cache(model, 1, 64)
    pos = torch.zeros(1, dtype=torch.long)
    logits = None
    with torch.no_grad():
        def feed(token, value=NAN):
            nonlocal logits
            logits = _step(model, memory, cache, pos.clone(), [tokenizer[token]], [value])
            pos.add_(1)

        for token in ["<bos>", "<expression>", "x1", "*"]:
            feed(token)
        start = int(pos[0])
        for token in wrap_float32(v1):
            feed(token)
        pos[0] = start
        feed(COMPACT_TOKEN, v1)
        for token in ["+", "x2", "*"]:
            feed(token)
        start = int(pos[0])
        for token in wrap_float32(v2):
            feed(token)
        pos[0] = start
        feed(COMPACT_TOKEN, v2)
        for token in ["+", "x3"]:
            feed(token)
    torch.testing.assert_close(logits[0, -1, :], reference, atol=LOGITS_ATOL, rtol=0)


def test_batch_rows_may_compact_at_different_steps(model, tokenizer, memory):
    """The case the dynamic path needed per-beam batch-1 caches for: two rows in ONE chunk,
    closing their spans three steps apart and rewinding to different slots."""
    plans = [
        (["<bos>", "<expression>", "x1", "*"], 1.5, ["+", "x2"]),
        (["<bos>", "<expression>", "x2", "+", "x3", "*", "x1", "*"], -0.25, ["+", "x1"]),
    ]
    references = [
        _compact_view_logits(model, tokenizer, memory,
                             [*[(t, NAN) for t in p], (COMPACT_TOKEN, v), *[(t, NAN) for t in c]])
        for p, v, c in plans]

    streams = [[*[(t, NAN) for t in p], *[(t, NAN) for t in wrap_float32(v)],
                ("<COMPACT>", v), *[(t, NAN) for t in c]] for p, v, c in plans]
    span_starts = [len(p) for p, _, _ in plans]

    mem2 = memory.expand(2, -1, -1).contiguous()
    cache = _cache(model, 2, 64)
    pos = torch.zeros(2, dtype=torch.long)
    out: list[torch.Tensor | None] = [None, None]
    with torch.no_grad():
        for step in range(max(len(s) for s in streams)):
            tokens, values = [], []
            for row, stream in enumerate(streams):
                if step >= len(stream):
                    tokens.append(tokenizer["<pad>"])
                    values.append(NAN)
                    continue
                token, value = stream[step]
                if token == "<COMPACT>":
                    pos[row] = span_starts[row]          # per-row rewind, at this row's own step
                    token = COMPACT_TOKEN
                tokens.append(tokenizer[token])
                values.append(value)
            logits = _step(model, mem2, cache, pos.clone(), tokens, values)
            for row, stream in enumerate(streams):
                if step == len(stream) - 1:
                    out[row] = logits[row, -1, :].clone()
            pos += 1

    for row, reference in enumerate(references):
        torch.testing.assert_close(out[row], reference, atol=LOGITS_ATOL, rtol=0)
