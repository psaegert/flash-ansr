"""T12 mechanics: forced serialization views (``expanded_mask``), the D-arm tail-policy
knob (``zero_tail_bits``), and the paired e3 eval that reads real validation batches.
The ACCEPTANCE half of T12 (gap ~ 0 for the shipped mixing policy) runs in T13's
training; these tests pin the machinery that makes that number trustworthy."""
import math
import struct

import numpy as np
import pytest

from flash_ansr import get_path
from flash_ansr.data.serialization import (
    CONSTANT_REPRESENTATION_IEEE754_MIXED,
    COMPACT_CONSTANT_TOKEN,
    serialize_constant_tokens,
)
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.train.paired_eval import build_paired_views, paired_e3_gap
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_SPAN_LENGTH,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
    nibble_tokens_to_float32,
)

SKELETON = ["+", "<constant>", "*", "<constant>", "x1"]
MIXED = dict(representation=CONSTANT_REPRESENTATION_IEEE754_MIXED)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


def _tiny_model(tokenizer: Tokenizer):  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    from flash_ansr.model.flash_ansr_model import FlashANSRModel

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    engine = SimpliPyEngine.load("base", install=True)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# expanded_mask: forced forms, no rng
# ---------------------------------------------------------------------------

def test_expanded_mask_forces_forms() -> None:
    serialized, numeric = serialize_constant_tokens(
        SKELETON, [1.5, -2.25], expanded_mask=[True, False], **MIXED)
    # First constant expanded: a full span right after '+'.
    assert serialized[1] == IEEE754_START_TOKEN
    assert serialized[1 + IEEE754_SPAN_LENGTH - 1] == IEEE754_END_TOKEN
    # Second constant compact.
    assert serialized.count(COMPACT_CONSTANT_TOKEN) == 1
    compact_position = serialized.index(COMPACT_CONSTANT_TOKEN)
    assert numeric[compact_position] == np.float32(-2.25)
    # And the reverse pattern flips both.
    flipped, _ = serialize_constant_tokens(
        SKELETON, [1.5, -2.25], expanded_mask=[False, True], **MIXED)
    assert flipped[1] == COMPACT_CONSTANT_TOKEN
    assert IEEE754_START_TOKEN in flipped


def test_expanded_mask_needs_matching_length() -> None:
    with pytest.raises(ValueError, match="expanded_mask length"):
        serialize_constant_tokens(SKELETON, [1.5, -2.25], expanded_mask=[True], **MIXED)


# ---------------------------------------------------------------------------
# zero_tail_bits: the D-arm tail policy quantizes the SPELLING only
# ---------------------------------------------------------------------------

def test_zero_tail_bits_quantizes_only_the_spelling() -> None:
    value = 1.0 / 3.0
    serialized, numeric = serialize_constant_tokens(
        ["*", "<constant>", "<constant>"], [value, value],
        expanded_mask=[True, False], zero_tail_bits=16, **MIXED)
    start = serialized.index(IEEE754_START_TOKEN)
    nibbles = serialized[start + 1:start + 9]
    spelled = nibble_tokens_to_float32(nibbles)
    pattern = int.from_bytes(struct.pack(">f", np.float32(spelled)), "big")
    assert pattern & 0xFFFF == 0, "the low 16 mantissa bits must be zero in the spelling"
    expected = int.from_bytes(struct.pack(">f", np.float32(value)), "big") & ~0xFFFF
    assert pattern == expected
    assert spelled != float(np.float32(value))
    # The compact form's numeric-channel value keeps FULL float32 precision.
    compact_position = serialized.index(COMPACT_CONSTANT_TOKEN)
    assert numeric[compact_position] == float(np.float32(value))


def test_zero_tail_bits_bounds() -> None:
    with pytest.raises(ValueError, match="zero_tail_bits"):
        serialize_constant_tokens(SKELETON, [1.0, 2.0], expanded_mask=[True, True],
                                  zero_tail_bits=24, **MIXED)


# ---------------------------------------------------------------------------
# build_paired_views
# ---------------------------------------------------------------------------

def test_paired_views_align_the_target_span(tokenizer: Tokenizer) -> None:
    views = build_paired_views(SKELETON, [1.5, -2.25], tokenizer, max_seq_len=128)
    assert views is not None
    (ids_e, num_e, span_e), (ids_c, num_c, span_c) = views
    # View C compacts one history constant: 9 tokens shorter than view E.
    assert len(ids_e) - len(ids_c) == IEEE754_SPAN_LENGTH - 1
    # The target span spells the SAME constant in both views.
    nibble_ids = {int(tokenizer[token]) for token in NIBBLE_TOKENS}
    spelled = []
    for ids, span in ((ids_e, span_e), (ids_c, span_c)):
        assert all(ids[p] in nibble_ids for p in span)
        spelled.append([ids[p] for p in span])
    assert spelled[0] == spelled[1]
    # View C's history constant rides the numeric channel exactly once.
    float_id = int(tokenizer[COMPACT_CONSTANT_TOKEN])
    compact_positions = [p for p, t in enumerate(ids_c) if t == float_id]
    assert len(compact_positions) == 1
    assert num_c[compact_positions[0]] == np.float32(1.5)
    assert all(math.isnan(num_e[p]) for p in span_e)


def test_paired_views_reject_single_constant(tokenizer: Tokenizer) -> None:
    assert build_paired_views(["*", "<constant>", "x1"], [2.0], tokenizer, max_seq_len=128) is None
    assert build_paired_views(SKELETON, [1.0], tokenizer, max_seq_len=128) is None  # count mismatch
    assert build_paired_views(SKELETON, [1.0, 2.0], tokenizer, max_seq_len=8) is None  # budget


# ---------------------------------------------------------------------------
# paired_e3_gap on a raw batch
# ---------------------------------------------------------------------------

def test_paired_e3_gap_returns_finite_deterministic_metrics(tokenizer: Tokenizer) -> None:
    model = _tiny_model(tokenizer)
    n_support, n_vars = 4, 10
    batch = {
        "skeleton": [SKELETON, ["*", "<constant>", "x1"], SKELETON],
        "constants": [np.asarray([1.5, -2.25], dtype=np.float32),
                      np.asarray([3.0], dtype=np.float32),
                      np.asarray([0.5, 4.0], dtype=np.float32)],
        "x_tensors": [np.zeros((n_support, n_vars), dtype=np.float32)] * 3,
        "y_tensors": [np.ones((n_support, 1), dtype=np.float32)] * 3,
        "data_attn_mask": [np.ones(n_support, dtype=np.float32)] * 3,
    }
    first = paired_e3_gap(model, tokenizer, batch, device="cpu", max_seq_len=128)
    assert first is not None
    assert first["e3_n"] == 2.0  # the single-constant instance is ineligible
    for key in ("e3_gap", "e3_nll_expanded", "e3_nll_compacted"):
        assert math.isfinite(first[key])
    assert first["e3_gap"] == pytest.approx(first["e3_nll_compacted"] - first["e3_nll_expanded"])
    second = paired_e3_gap(model, tokenizer, batch, device="cpu", max_seq_len=128)
    assert second == first  # teacher forcing is deterministic


def test_paired_e3_gap_none_when_nothing_eligible(tokenizer: Tokenizer) -> None:
    model = _tiny_model(tokenizer)
    batch = {
        "skeleton": [["*", "<constant>", "x1"]],
        "constants": [np.asarray([3.0], dtype=np.float32)],
        "x_tensors": [np.zeros((4, 10), dtype=np.float32)],
        "y_tensors": [np.ones((4, 1), dtype=np.float32)],
        "data_attn_mask": [np.ones(4, dtype=np.float32)],
    }
    assert paired_e3_gap(model, tokenizer, batch, device="cpu", max_seq_len=128) is None


def test_paired_e3_gap_accepts_tensor_batch(tokenizer: Tokenizer) -> None:
    # The real pipeline hands RAW batches whose fields are torch tensors (cuda in
    # production: the first T13 validation crashed on np.asarray(cuda_tensor), and the
    # x/y/mask stacking would have crashed the same way one line later). The
    # tensor-typed batch must produce the same metrics as the list/ndarray-typed one.
    import torch

    model = _tiny_model(tokenizer)
    n_support, n_vars = 4, 10
    batch = {
        "skeleton": [SKELETON, ["*", "<constant>", "x1"], SKELETON],
        "constants": [np.asarray([1.5, -2.25], dtype=np.float32),
                      np.asarray([3.0], dtype=np.float32),
                      np.asarray([0.5, 4.0], dtype=np.float32)],
        "x_tensors": [np.zeros((n_support, n_vars), dtype=np.float32)] * 3,
        "y_tensors": [np.ones((n_support, 1), dtype=np.float32)] * 3,
        "data_attn_mask": [np.ones(n_support, dtype=np.float32)] * 3,
    }
    listy = paired_e3_gap(model, tokenizer, batch, device="cpu", max_seq_len=128)
    tensor_batch = dict(batch)
    for key in ("constants", "x_tensors", "y_tensors", "data_attn_mask"):
        tensor_batch[key] = [torch.as_tensor(item) for item in batch[key]]
    tensory = paired_e3_gap(model, tokenizer, tensor_batch, device="cpu", max_seq_len=128)
    assert listy is not None and tensory is not None
    assert tensory["e3_n"] == listy["e3_n"]
    assert tensory["e3_gap"] == pytest.approx(listy["e3_gap"])
    assert tensory["e3_nll_expanded"] == pytest.approx(listy["e3_nll_expanded"])
