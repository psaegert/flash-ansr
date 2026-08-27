"""Acceptance tests T9-T11 for the v24 `ieee754_mixed` constants BEAM pathway (lane C3).

Pre-registered integration contract, lane C3: per-beam KV compaction under beam search
(T9, the per-beam version of the T8 golden compaction-equivalence, including beams that
compact at DIFFERENT steps), mid-span safety (T10: a beam pruned mid-expansion leaves no
orphaned state; compaction fires only on a closed, finite-valued tag), and the refiner
handshake (T11: the predicted float32 reaches the Refiner as its init VERBATIM, bit-exact,
with refiner failure falling back exactly to the init-free path).

NOTE on the test engine: models are constructed directly from the configs/test model
kwargs with the generation-2 'base' engine (the lane C1/C2 pattern); refiner-level tests
use the released 'acj-4-3' engine like tests/test_refine.py.

Format ruling 2026-08-18: expanded constants are HEX NIBBLE spans -- `<ieee754>` + 8
tokens over the 16-symbol `<h0>`..`<hf>` alphabet + `</ieee754>` = 10 tokens (was 34).
"""
import copy
import math
import struct

import numpy as np
import pytest
import torch

from flash_ansr.utils.numeric import NUMERIC_DTYPE

from flash_ansr import get_path
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

# The repo-standard logits-equivalence tolerance (the T8 bar, see test_constant_decoding.py).
LOGITS_ATOL = 1e-5
# Beam scores are SUMS of ~40 per-token log-probs, each carrying ~1e-6 float error, so the
# score-decomposition equality gets a proportionally looser (still tight) bar.
SCORE_ATOL = 1e-4


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("base", install=True)


@pytest.fixture(scope="module")
def refine_engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("acj-4-3", install=True)


def _tiny_model(tokenizer: Tokenizer, engine, seed: int = 0x24C3):  # type: ignore[no-untyped-def]
    from flash_ansr.model.flash_ansr_model import FlashANSRModel

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    torch.manual_seed(seed)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model


def _steered_model(tokenizer: Tokenizer, engine, biases: dict[str, float], seed: int = 0x24C3):  # type: ignore[no-untyped-def]
    """A tiny model whose head is biased so beam search follows a scripted token landscape."""
    model = _tiny_model(tokenizer, engine, seed)
    with torch.no_grad():
        for token, bias in biases.items():
            model.next_token_head[-1].bias[tokenizer[token]] += bias
    return model


def _scan_spans(ids: list[int], tokenizer: Tokenizer) -> list[float]:
    """Assert every `<ieee754>` span in `ids` is well-formed and return the decoded values
    (the C2 helper; stray nibble/close tokens outside a span fail the assertion)."""
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
            values.append(nibble_tokens_to_float32([id_to_nibble[x] for x in inner]))
            i += IEEE754_SPAN_LENGTH
        else:
            assert token not in id_to_nibble and token != close_id, \
                f"stray nibble/close outside span at {i}: {ids}"
            i += 1
    return values


def _nan(*shape: int) -> torch.Tensor:
    return torch.full(shape, float("nan"), dtype=NUMERIC_DTYPE)


# The T9/T10 steering landscape. Head biases alone cannot express "open a span EARLY,
# then stop opening spans" (an untrained model's short junk otherwise outscores every
# span-carrying completion), so the tests shadow the model instance's forward with a
# position-gated boost: <ieee754> is strongly favored only while the total sequence is
# <= 3 tokens (the first two generated slots). Three tracks emerge: B opens at position
# 2, A ('sin') and C ('</expression>' junk) open at position 3 -- so spans CLOSE at
# different steps AND different absolute positions, and after compacting, 'sin <constant>'
# is the one track that completes into a VALID single-span expression.
_T9_BIASES = {
    "sin": 13.9,
    "</expression>": 13.0,
    # Steer the 16-symbol in-span alphabet onto <h0> (the all-zero span decodes to +0.0,
    # a finite value compaction accepts); every other nibble is pushed far down.
    **{token: (11.0 if token == "<h0>" else -20.0) for token in NIBBLE_TOKENS},
    IEEE754_END_TOKEN: 8.0,
    "<eos>": 8.0,
}
#: Boost the forward shadow adds to <ieee754> while total length <= 3 (positions 2-3).
_OPEN_BOOST = 25.0
_T9_MAX_LEN = 38
_T9_WIDTH = 3
_T9_SEED = 0x24C9


def _install_early_open_boost(model, tokenizer: Tokenizer) -> None:  # type: ignore[no-untyped-def]
    """Shadow the INSTANCE's forward (the class stays untouched): +_OPEN_BOOST on the
    <ieee754> logit at the next-token position while the total length is <= 3."""
    open_id = int(tokenizer[IEEE754_START_TOKEN])
    orig_forward = model.forward

    def boosted_forward(input_tokens, data, input_num=None, memory=None,  # type: ignore[no-untyped-def]
                        past_key_values=None, use_cache=False, **kwargs):
        result = orig_forward(input_tokens, data, input_num=input_num, memory=memory,
                              past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        cached = past_key_values[0][0][0].shape[2] if past_key_values is not None else 0
        if cached + input_tokens.shape[1] > 3:
            return result
        if isinstance(result, tuple):
            logits, past = result
            logits = logits.clone()
            logits[:, -1, open_id] += _OPEN_BOOST
            return logits, past
        logits = result.clone()
        logits[:, -1, open_id] += _OPEN_BOOST
        return logits

    model.forward = boosted_forward


def _boosted_log_probs(logits: torch.Tensor, open_id: int) -> torch.Tensor:
    """Log-probs under the SAME scoring model the shadowed forward exposes: the boost
    applies at predict-positions 1-2 (which emit tokens 2-3)."""
    boosted = logits.clone()
    for predict_position in (1, 2):
        if predict_position < boosted.shape[1]:
            boosted[:, predict_position, open_id] += _OPEN_BOOST
    return torch.log_softmax(boosted, dim=-1)


def _spied_compaction(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Route the beam loop's compaction calls through a recording spy (observation from
    OUTSIDE: the production seam is monkeypatched, never instrumented)."""
    import flash_ansr.decoding.beam_compaction as beam_compaction_module
    from flash_ansr.decoding.compaction import compact_closed_ieee754_spans as real_compact

    calls: list[dict] = []

    def spy(model, sequences, current_length, past_key_values, memory, input_num=None):  # type: ignore[no-untyped-def]
        result = real_compact(model, sequences, current_length=current_length,
                              past_key_values=past_key_values, memory=memory, input_num=input_num)
        calls.append({
            "sequences": sequences.clone(),
            "current_length": current_length,
            "memory": memory,
            "result": result,
        })
        return result

    monkeypatch.setattr(beam_compaction_module, "compact_closed_ieee754_spans", spy)
    return calls


def _spied_grammar_states(monkeypatch: pytest.MonkeyPatch, tokenizer: Tokenizer) -> dict[str, int]:
    """Count, from OUTSIDE, how often the decode loop asked for a mask while a beam sat
    INSIDE an open span -- the mid-span state T10 is about (the production grammar stays
    uninstrumented; the class method is monkeypatched)."""
    from flash_ansr.decoding.constrained import IEEE754GrammarConstraint

    open_id = int(tokenizer[IEEE754_START_TOKEN])
    close_id = int(tokenizer[IEEE754_END_TOKEN])
    counts = {"mid_span": 0}
    real_forbidden = IEEE754GrammarConstraint.forbidden

    def spy(self, prefixes, remaining=None):  # type: ignore[no-untyped-def]
        for row in prefixes.tolist():
            if open_id not in row:
                continue
            tail = row[len(row) - 1 - row[::-1].index(open_id):]
            if close_id not in tail:
                counts["mid_span"] += 1
        return real_forbidden(self, prefixes, remaining)

    monkeypatch.setattr(IEEE754GrammarConstraint, "forbidden", spy)
    return counts


def _run_t9_beam_search(tokenizer: Tokenizer, engine, bias_overrides: dict[str, float] | None = None):  # type: ignore[no-untyped-def]
    biases = dict(_T9_BIASES)
    if bias_overrides:
        biases.update(bias_overrides)
    model = _steered_model(tokenizer, engine, biases)
    _install_early_open_boost(model, tokenizer)
    initial = [tokenizer["<bos>"], tokenizer["<expression>"]]
    torch.manual_seed(_T9_SEED)
    data = torch.rand(13, 11, dtype=NUMERIC_DTYPE)
    beams, log_probs, completed = model.beam_search(
        data, beam_width=_T9_WIDTH, max_len=_T9_MAX_LEN, unique=True, use_cache=True,
        initial_tokens=initial, constrain_ieee754=True, compact_ieee754=True)
    return model, data, beams, log_probs, completed


# ---------------------------------------------------------------------------
# T9 — per-beam KV compaction: the golden equivalence holds per beam, and beams
# compacting at different steps stay score-comparable
# ---------------------------------------------------------------------------

def test_t9_per_beam_compaction_golden_desynchronized(tokenizer: Tokenizer, engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    calls = _spied_compaction(monkeypatch)
    model, _data, beams, _log_probs, _completed = _run_t9_beam_search(tokenizer, engine)

    # Beams compacted at DIFFERENT steps (desynchronized closes -> per-row cache lengths).
    assert len(calls) >= 2, "expected at least two compaction events"
    assert len({call["current_length"] for call in calls}) >= 2, \
        f"expected desynchronized compaction lengths, got {[c['current_length'] for c in calls]}"

    close_id = tokenizer[IEEE754_END_TOKEN]
    with torch.no_grad():
        for call in calls:
            result = call["result"]
            length = call["current_length"]
            # Fires only on a JUST-closed span at the cache tail.
            assert bool((call["sequences"][:, length - 1] == close_id).all())
            # THE golden equality, per beam: the compact-token re-encode logits (the
            # continuation distribution actually used for selection) equal a fresh
            # forward over that beam's compact-view sequence, to the T8 bar.
            for row in range(result.sequences.shape[0]):
                fresh = model.forward(
                    result.sequences[row:row + 1, :result.length], None,
                    input_num=result.input_num[row:row + 1, :result.length].unsqueeze(-1),
                    memory=call["memory"], use_cache=False)
                max_diff = (fresh[:, -1:] - result.logits[row:row + 1]).abs().max().item()
                assert torch.allclose(fresh[:, -1:], result.logits[row:row + 1], atol=LOGITS_ATOL), \
                    f"per-beam golden equality violated: max diff {max_diff}"

    # The returned beams present constants in EXPANDED form (values ride the tokens,
    # bit-exact) -- no compact <float> token may leak out of the decode.
    float_id = tokenizer[COMPACT_TOKEN]
    assert all(float_id not in seq for seq in beams)
    spans_seen = sum(len(_scan_spans(seq, tokenizer)) for seq in beams)
    assert spans_seen >= 1, "no expanded span in any returned beam: compaction result lost"


def test_t9_scores_stay_comparable_across_compaction(tokenizer: Tokenizer, engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Score accounting: a completed beam's score equals expanded-history log-probs up to
    the span close PLUS compact-history log-probs after compaction (no rescoring, no
    double counting), so beams that compacted at different steps rank on one scale."""
    calls = _spied_compaction(monkeypatch)
    model, data, beams, log_probs, completed = _run_t9_beam_search(tokenizer, engine)
    assert calls, "no compaction event fired"

    open_id = tokenizer[IEEE754_START_TOKEN]
    close_id = tokenizer[IEEE754_END_TOKEN]
    float_id = tokenizer[COMPACT_TOKEN]
    prefix_len = 2  # [<bos>, <expression>]

    # Pick the best COMPLETED beam carrying exactly one expanded span.
    candidates = [
        (seq, score) for seq, score, flag in zip(beams, log_probs, completed)
        if flag and seq.count(open_id) == 1
    ]
    assert candidates, f"no completed single-span beam among {beams}"
    seq, score = candidates[0]

    span_start = seq.index(open_id)
    span_end = span_start + IEEE754_SPAN_LENGTH  # exclusive
    assert seq[span_end - 1] == close_id
    value = _scan_spans(seq, tokenizer)[0]

    with torch.no_grad():
        memory = model._create_memory(data)

        # Phase 1 (expanded history): tokens up to and including the close tag.
        expanded_ids = torch.tensor([seq[:span_end]], dtype=torch.long)
        logits = model.forward(expanded_ids, None, input_num=_nan(1, span_end).unsqueeze(-1),
                               memory=memory, use_cache=False)
        lp = _boosted_log_probs(logits, open_id)
        phase1 = sum(float(lp[0, t - 1, seq[t]]) for t in range(prefix_len, span_end))

        # Phase 2 (compact history): the continuation after the compaction event.
        compact_seq = seq[:span_start] + [float_id] + seq[span_end:]
        compact_num = _nan(1, len(compact_seq))
        compact_num[0, span_start] = value
        compact_ids = torch.tensor([compact_seq], dtype=torch.long)
        logits2 = model.forward(compact_ids, None, input_num=compact_num.unsqueeze(-1),
                                memory=memory, use_cache=False)
        lp2 = _boosted_log_probs(logits2, open_id)
        phase2 = sum(float(lp2[0, t - 1, compact_seq[t]])
                     for t in range(span_start + 1, len(compact_seq)))

        # Non-vacuity: scoring the continuation under the EXPANDED history must differ,
        # or this test could not distinguish compacted from uncompacted state.
        full_ids = torch.tensor([seq], dtype=torch.long)
        logits3 = model.forward(full_ids, None, input_num=_nan(1, len(seq)).unsqueeze(-1),
                                memory=memory, use_cache=False)
        lp3 = _boosted_log_probs(logits3, open_id)
        phase2_expanded = sum(float(lp3[0, t - 1, seq[t]]) for t in range(span_end, len(seq)))

    assert abs(phase2 - phase2_expanded) > 1e-3, "expanded and compact continuations indistinguishable (vacuous test)"
    assert score == pytest.approx(phase1 + phase2, abs=SCORE_ATOL), \
        f"score {score} != expanded-phase {phase1} + compact-phase {phase2}"
    assert abs(score - (phase1 + phase2_expanded)) > 1e-3, \
        "score matches the UNCOMPACTED accounting: the continuation was not scored from the compacted state"


def test_t9_compaction_requires_grammar_and_cache(tokenizer: Tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    """Compaction is KV surgery on grammar-guaranteed spans: without the constrained mask
    or without the dynamic cache there is nothing sound to compact -- refuse loudly."""
    model = _steered_model(tokenizer, engine, _T9_BIASES)
    initial = [tokenizer["<bos>"], tokenizer["<expression>"]]
    torch.manual_seed(_T9_SEED)
    data = torch.rand(13, 11, dtype=NUMERIC_DTYPE)

    with pytest.raises(ValueError):
        model.beam_search(data, beam_width=2, max_len=_T9_MAX_LEN, use_cache=True,
                          initial_tokens=initial, compact_ieee754=True)
    with pytest.raises(ValueError):
        model.beam_search(data, beam_width=2, max_len=_T9_MAX_LEN, use_cache=False,
                          initial_tokens=initial, constrain_ieee754=True, compact_ieee754=True)


# ---------------------------------------------------------------------------
# T10 — mid-span safety: a beam pruned mid-expansion leaves no orphaned state;
# compaction fires only on a closed (finite-valued) tag
# ---------------------------------------------------------------------------

def test_t10_compaction_fires_only_on_closed_tag(tokenizer: Tokenizer, engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """In the desynchronized T9 run, the FIRST compaction carries exactly the one beam
    that closed (batch 1): the other beams, mid-span at that step, are never swept into
    the surgery -- and no call ever fires on anything but a close-tag tail."""
    calls = _spied_compaction(monkeypatch)
    _model, _data, _beams, _log_probs, _completed = _run_t9_beam_search(tokenizer, engine)

    close_id = tokenizer[IEEE754_END_TOKEN]
    assert len(calls) >= 2
    assert calls[0]["sequences"].shape[0] == 1, \
        "the first close is unique to one beam; its compaction must not include mid-span rows"
    assert len({call["current_length"] for call in calls}) >= 2
    for call in calls:
        # Every compacted row ends on the close tag (fires ONLY on a closed span)...
        assert bool((call["sequences"][:, call["current_length"] - 1] == close_id).all())
        # ... and only finite values were compacted.
        assert bool(torch.isfinite(call["result"].values).all())


def test_t10_pruned_mid_span_leaves_no_orphaned_state(tokenizer: Tokenizer, engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Beams that open spans here are outcompeted within a step or two (nibble tokens carry
    no bias), so every span-carrying beam is pruned MID-expansion: compaction must never
    fire, the search must terminate cleanly, and no fragment of a pruned span may
    resurface in the results."""
    calls = _spied_compaction(monkeypatch)
    states = _spied_grammar_states(monkeypatch, tokenizer)
    biases = {
        IEEE754_START_TOKEN: 14.0,
        **{token: -20.0 for token in NIBBLE_TOKENS},
        IEEE754_END_TOKEN: 8.0,
        "*": 13.0,
        "x1": 12.0,
        "</expression>": 12.5,
    }
    model = _steered_model(tokenizer, engine, biases)
    initial = [tokenizer["<bos>"], tokenizer["<expression>"]]
    torch.manual_seed(_T9_SEED)
    data = torch.rand(13, 11, dtype=NUMERIC_DTYPE)

    beams, log_probs, completed = model.beam_search(
        data, beam_width=2, max_len=16, unique=True, use_cache=True,
        initial_tokens=initial, constrain_ieee754=True, compact_ieee754=True)

    assert calls == [], "compaction fired although no span ever closed"
    assert beams, "mid-span pruning starved the beam search"
    # Non-vacuity: a span really did open and the loop really did evaluate the MID-SPAN
    # state before pruning it. The retired 34-token spans never fit the max_len=16 budget
    # (the grammar's budget rule forbade opening outright), so this guard only became
    # exercisable with 10-token hex spans.
    assert states["mid_span"] > 0, "no beam ever entered a span: the mid-span path is untested"
    open_id = tokenizer[IEEE754_START_TOKEN]
    close_id = tokenizer[IEEE754_END_TOKEN]
    float_id = tokenizer[COMPACT_TOKEN]
    for seq, flag in zip(beams, completed):
        # Nothing was compacted, so no compact token may appear anywhere.
        assert float_id not in seq
        # No CLOSED span exists in any returned sequence (none ever completed)...
        assert close_id not in seq
        # ... and completed sequences are entirely span-free (a pruned span leaves no trace).
        if flag:
            assert open_id not in seq
            _scan_spans(seq, tokenizer)


def test_t10_nonfinite_span_is_left_expanded(tokenizer: Tokenizer, engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """An all-<hf> span (0xffffffff) decodes to NaN: the numeric channel cannot carry it, so compaction
    must NOT fire (the span stays expanded in history and in the returned beam) -- the
    landed T8 refusal, honored by the beam loop instead of crashing it."""
    calls = _spied_compaction(monkeypatch)
    _model, _data, beams, _log_probs, completed = _run_t9_beam_search(
        tokenizer, engine, bias_overrides={"<h0>": -20.0, "<hf>": 11.0})

    assert calls == [], "compaction fired on a non-finite constant"
    nan_spans = 0
    for seq in beams:
        for value in _scan_spans(seq, tokenizer):
            assert math.isnan(value)
            nan_spans += 1
    assert nan_spans >= 1, "no NaN span survived to the results: the refusal path was not exercised"


# ---------------------------------------------------------------------------
# T11 — refiner handshake: the predicted float32 reaches the Refiner as its init
# VERBATIM (bit-exact); refiner failure falls back exactly to the init-free path
# ---------------------------------------------------------------------------

def _span_ids(tokenizer: Tokenizer, value: float) -> list[int]:
    return [int(tokenizer[token]) for token in wrap_float32(value)]


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def test_t11_span_to_constant_mapping_is_bit_exact(tokenizer: Tokenizer) -> None:
    from flash_ansr.data.serialization import replace_ieee754_spans_with_constants

    constant_id = int(tokenizer["<constant>"])
    values = [float(np.float32(0.1)), 1e-42, -3.75]  # inexact decimal, denormal, exact
    ids = [int(tokenizer["*"]), *(_span_ids(tokenizer, values[0])),
           int(tokenizer["x1"])]
    for value in values[1:]:
        ids = [int(tokenizer["+"]), *ids, *(_span_ids(tokenizer, value))]

    mapped, extracted = replace_ieee754_spans_with_constants(
        ids,
        start_id=int(tokenizer[IEEE754_START_TOKEN]),
        end_id=int(tokenizer[IEEE754_END_TOKEN]),
        nibble_ids=[int(tokenizer[token]) for token in NIBBLE_TOKENS],
        constant_id=constant_id,
    )

    assert extracted is not None
    assert len(extracted) == 3
    # Bit-exact: the float32 round-trips into the p0 slot with NO precision loss.
    # Each wrap APPENDS its span, so order of appearance is values[0], [1], [2].
    assert [_bits(v) for v in extracted] == [_bits(float(np.float32(v))) for v in values]
    assert _bits(extracted[0]) == _bits(float(np.float32(0.1)))
    assert extracted[0] != 0.1  # the DOUBLE 0.1 would mean a lossy decimal round-trip
    assert mapped.count(constant_id) == 3
    assert all(token != int(tokenizer[IEEE754_START_TOKEN]) for token in mapped)


def test_t11_mapping_degenerate_cases(tokenizer: Tokenizer) -> None:
    from flash_ansr.data.serialization import replace_ieee754_spans_with_constants

    kwargs = dict(
        start_id=int(tokenizer[IEEE754_START_TOKEN]),
        end_id=int(tokenizer[IEEE754_END_TOKEN]),
        nibble_ids=[int(tokenizer[token]) for token in NIBBLE_TOKENS],
        constant_id=int(tokenizer["<constant>"]),
    )
    plain = [int(tokenizer["*"]), int(tokenizer["x1"]), int(tokenizer["x2"])]

    # No spans: identity, no init (the init-free path stays byte-identical).
    mapped, values = replace_ieee754_spans_with_constants(plain, **kwargs)
    assert mapped == plain and values is None

    # Malformed span (truncated): input returned unchanged, no init.
    truncated = plain + _span_ids(tokenizer, 2.0)[:-5]
    mapped, values = replace_ieee754_spans_with_constants(truncated, **kwargs)
    assert mapped == truncated and values is None

    # Well-formed but non-finite value: the SKELETON is still salvaged, the init is not.
    nan_span = [kwargs["start_id"], *([kwargs["nibble_ids"][0xf]] * IEEE754_N_NIBBLES), kwargs["end_id"]]
    mapped, values = replace_ieee754_spans_with_constants([int(tokenizer["*"]), *nan_span, int(tokenizer["x1"])], **kwargs)
    assert mapped == [int(tokenizer["*"]), kwargs["constant_id"], int(tokenizer["x1"])]
    assert values is None

    # A pre-existing bare '<constant>' (e.g. constantified sugar) breaks slot alignment:
    # map the skeleton, withhold the init.
    mixed = [int(tokenizer["+"]), kwargs["constant_id"], *(_span_ids(tokenizer, 2.5))]
    mapped, values = replace_ieee754_spans_with_constants(mixed, **kwargs)
    assert mapped == [int(tokenizer["+"]), kwargs["constant_id"], kwargs["constant_id"]]
    assert values is None


class _RecordingRefiner:
    """Class-level recording wrapper around the REAL Refiner (installed via monkeypatch;
    the production worker stays uninstrumented)."""

    calls: list[dict] = []

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from flash_ansr.refine import Refiner

        instance = Refiner(*args, **kwargs)
        original_fit = instance.fit

        def recording_fit(**fit_kwargs):  # type: ignore[no-untyped-def]
            record = {k: copy.deepcopy(v) for k, v in fit_kwargs.items() if k not in ("X", "y")}
            cls.calls.append(record)
            return original_fit(**fit_kwargs)

        instance.fit = recording_fit  # type: ignore[method-assign]
        return instance

    @classmethod
    def from_serialized(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from flash_ansr.refine import Refiner

        return Refiner.from_serialized(*args, **kwargs)


def _worker_payload(tokenizer: Tokenizer, refine_engine, expression: list[str], raw_beam: list[int],  # type: ignore[no-untyped-def]
                    p0: list[float] | None, X: np.ndarray, y: np.ndarray) -> dict:
    payload = {
        'raw_beam': raw_beam,
        'raw_beam_decoded': expression,
        'beam': [int(tokenizer[token]) for token in expression],
        'expression': expression,
        'log_prob': -1.0,
        'constant_count': sum(1 for token in expression if token == '<constant>'),
        'pruned_variant': False,
        'n_variables': 1,
        'n_restarts': 3,
        'method': 'curve_fit_lm',
        'p0_noise': 'normal',
        'p0_noise_kwargs': {'loc': 0.0, 'scale': 5.0},
        'converge_error': 'ignore',
        'numpy_errors': 'ignore',
        'y_variance': float(np.var(y)),
        'length_penalty': 0.05,
        'constants_penalty': 0.0,
        'likelihood_penalty': 0.0,
        'complexity': None,
        'metadata_snapshot': None,
        'seed': 1234,
        'X': X,
        'y': y,
        'simplipy_engine': refine_engine,
    }
    if p0 is not None:
        payload['p0'] = p0
    return payload


def test_t11_worker_uses_predicted_init_verbatim(tokenizer: Tokenizer, refine_engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import flash_ansr.flash_ansr as flash_ansr_module
    from flash_ansr.flash_ansr import _refine_candidate_worker

    monkeypatch.setattr(flash_ansr_module, "Refiner", _RecordingRefiner)
    _RecordingRefiner.calls = []

    value = float(np.float32(0.1))
    rng = np.random.default_rng(0x24CB)
    X = rng.uniform(0.5, 2.0, size=(32, 1))
    y = (value * X[:, 0]).reshape(-1, 1)
    expression = ['*', '<constant>', 'x1']

    payload = _worker_payload(tokenizer, refine_engine, expression, raw_beam=[1, 2, 3],
                              p0=[value], X=X, y=y)
    result, warning = _refine_candidate_worker(payload)

    assert warning is None and result is not None
    # ONE fit, from the predicted init, VERBATIM: p0 bit-equals the predicted float32,
    # no noise, a single restart (exploratory restarts exist only as the failure fallback).
    assert len(_RecordingRefiner.calls) == 1
    call = _RecordingRefiner.calls[0]
    assert call['n_restarts'] == 1
    assert call['p0_noise'] is None
    assert _bits(float(np.asarray(call['p0']).reshape(-1)[0])) == _bits(value)
    # The init converges on this data, and the fitted constant stays at the prediction.
    assert result['valid_fit']
    best_constants = result['fits'][0][0]
    assert float(best_constants[0]) == pytest.approx(value, abs=1e-6)


def test_t11_worker_failure_falls_back_exactly_to_the_init_free_path(tokenizer: Tokenizer, refine_engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import flash_ansr.flash_ansr as flash_ansr_module
    from flash_ansr.flash_ansr import _refine_candidate_worker

    monkeypatch.setattr(flash_ansr_module, "Refiner", _RecordingRefiner)

    rng = np.random.default_rng(0x24CC)
    X = rng.uniform(0.5, 2.0, size=(32, 1))
    y = np.log(3.0 * X[:, 0]).reshape(-1, 1)
    expression = ['log', '*', '<constant>', 'x1']

    # A BAD init: log(-1 * x1) is NaN on all of X, so the verbatim fit cannot converge.
    _RecordingRefiner.calls = []
    payload = _worker_payload(tokenizer, refine_engine, expression, raw_beam=[4, 5, 6],
                              p0=[-1.0], X=X, y=y)
    result_fallback, _warning = _refine_candidate_worker(payload)

    assert len(_RecordingRefiner.calls) == 2, "expected the verbatim attempt AND the init-free fallback"
    verbatim, fallback = _RecordingRefiner.calls
    assert verbatim['n_restarts'] == 1 and verbatim['p0_noise'] is None
    # The fallback IS the plain call: same restarts, same noise policy, no carried p0.
    assert fallback['n_restarts'] == payload['n_restarts']
    assert fallback['p0_noise'] == payload['p0_noise']
    assert fallback['p0_noise_kwargs'] == payload['p0_noise_kwargs']
    assert fallback.get('p0') is None

    # ... and it reproduces the init-free refinement EXACTLY: the verbatim attempt consumes
    # no RNG, so a plain worker run with the same seed yields bit-identical fits.
    _RecordingRefiner.calls = []
    payload_plain = _worker_payload(tokenizer, refine_engine, expression, raw_beam=[4, 5, 6],
                                    p0=None, X=X, y=y)
    result_plain, _warning = _refine_candidate_worker(payload_plain)

    assert len(_RecordingRefiner.calls) == 1
    assert (result_fallback is None) == (result_plain is None)
    if result_fallback is not None and result_plain is not None:
        assert len(result_fallback['fits']) == len(result_plain['fits'])
        for (c_a, cov_a, loss_a), (c_b, cov_b, loss_b) in zip(result_fallback['fits'], result_plain['fits']):
            assert np.array_equal(np.asarray(c_a), np.asarray(c_b))
            assert (loss_a == loss_b) or (math.isnan(loss_a) and math.isnan(loss_b))


def test_t11_worker_without_p0_is_byte_identical(tokenizer: Tokenizer, refine_engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import flash_ansr.flash_ansr as flash_ansr_module
    from flash_ansr.flash_ansr import _refine_candidate_worker

    monkeypatch.setattr(flash_ansr_module, "Refiner", _RecordingRefiner)
    _RecordingRefiner.calls = []

    rng = np.random.default_rng(0x24CD)
    X = rng.uniform(0.5, 2.0, size=(16, 1))
    y = (2.0 * X[:, 0]).reshape(-1, 1)
    payload = _worker_payload(tokenizer, refine_engine, ['*', '<constant>', 'x1'],
                              raw_beam=[7, 8, 9], p0=None, X=X, y=y)
    result, _warning = _refine_candidate_worker(payload)

    assert result is not None
    # Exactly the plain call and nothing else: default OFF means byte-identical behavior.
    assert len(_RecordingRefiner.calls) == 1
    call = _RecordingRefiner.calls[0]
    assert call.get('p0') is None
    assert call['p0_noise'] == 'normal'
    assert call['n_restarts'] == payload['n_restarts']


def test_t11_fit_refine_wires_beam_constants_into_p0(tokenizer: Tokenizer, refine_engine, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """End-to-end handshake through FlashANSR.fit: a generated beam carrying an expanded
    span reaches the Refiner as a '<constant>' skeleton whose p0 holds the decoded float32
    BIT-EXACTLY."""
    import flash_ansr.flash_ansr as flash_ansr_module
    from flash_ansr import FlashANSR, BeamSearchConfig

    value = float(np.float32(0.1))
    beam_tokens = [
        int(tokenizer['<bos>']), int(tokenizer['<expression>']),
        int(tokenizer['*']),
        *(_span_ids(tokenizer, value)),
        int(tokenizer['x1']),
        int(tokenizer['</expression>']), int(tokenizer['<eos>']),
    ]

    def _fake_generate(self, data, *, prompt_prefix=None, complexity=None, verbose=False, memory=None):  # type: ignore[no-untyped-def]
        return [list(beam_tokens)], [-1.0], [True], [float('nan')]

    monkeypatch.setattr(FlashANSR, 'generate', _fake_generate)
    monkeypatch.setattr(flash_ansr_module, 'Refiner', _RecordingRefiner)
    _RecordingRefiner.calls = []

    model = _tiny_model(tokenizer, refine_engine)
    ansr = FlashANSR(
        simplipy_engine=refine_engine,
        flash_ansr_model=model,
        tokenizer=tokenizer,
        generation_config=BeamSearchConfig(beam_width=1, max_len=8),
        n_restarts=2,
        refiner_workers=1,
    )

    rng = np.random.default_rng(0x24CE)
    X = rng.uniform(0.5, 2.0, size=(24, 1))
    y = (value * X[:, 0]).reshape(-1, 1)
    ansr.fit(X, y)

    assert _RecordingRefiner.calls, "the v24 beam never reached the Refiner"
    call = _RecordingRefiner.calls[0]
    assert call['expression'] == ['*', '<constant>', 'x1']
    assert call['n_restarts'] == 1 and call['p0_noise'] is None
    p0 = np.asarray(call['p0']).reshape(-1)
    # The span slot holds the predicted float32, bit-exactly.
    assert p0.shape == (1,)
    assert _bits(float(p0[0])) == _bits(value)
    assert len(_RecordingRefiner.calls) == 1, "the converged verbatim fit must not trigger restarts"
