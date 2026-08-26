"""Acceptance tests T2/T3/T4 for the `ieee754_mixed` constants representation.

Pre-registered integration contract, lane C1 (vocabulary + data serialization with
per-constant 50/50 mixing + loss mask). `ieee754_mixed` is the only representation
this generation serves, and the default of the `constant_representation` config key.

T1 (codec round-trip) lives in tests/test_ieee754.py.

Format ruling 2026-08-18: constants expand to HEX NIBBLES -- `<ieee754>` + 8 nibble
tokens over the 16-symbol `<h0>`..`<hf>` alphabet + `</ieee754>` = 10 tokens (was 34).
The v24.0 target dialect is simplipy's TAGGED canonical form with no masking and no
explicit number tokens; its template vocabulary is pinned by the T2 block below.

NOTE on the test engine: the configs/test bundle references the generation-1
'dev_7-3' simplipy asset, refused at load by the simplipy generation gate this
repo now targets (a pre-existing baseline condition, out of scope here). The
streaming tests therefore build a generation-2 'base'-engine catalog inline and
pin frozen placeholder-form skeletons (the `<constant>`-slot era that the
streaming worker consumes).
"""
import math
import struct
from typing import Any

import numpy as np
import pytest
import torch

from flash_ansr import FlashANSRDataset, get_path
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
)


NEW_SPECIAL_TOKENS = [IEEE754_START_TOKEN, IEEE754_END_TOKEN, *NIBBLE_TOKENS]
COMPACT_TOKEN = "<float>"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


def _placeholder_catalog():
    """A generation-2 ('base' engine) catalog pinned to placeholder-form skeletons."""
    from symbolic_data.generative import LampleChartonCatalog

    catalog_cfg = {
        "type": "lample_charton",
        "simplipy_engine": "base",
        "holdout_pools": [],
        "sample_strategy": {
            "n_operator_distribution": "length_proportional",
            "min_operators": 1, "max_operators": 6, "power": 1,
            "max_length": 21, "max_tries": 1, "independent_dimensions": True,
        },
        "allow_nan": False,
        "simplify": True,
        "literal_prior": {"name": "normal", "kwargs": {"loc": 0, "scale": 5}},
        "support_sampler": {
            "support_prior": {"name": "uniform", "kwargs": {"low": -5, "high": 5, "min_value": -5, "max_value": 5}},
            "n_support_prior": {"name": "uniform", "kwargs": {"low": 4, "high": 16, "min_value": 4, "max_value": 16}},
        },
        "variables": ["x1", "x2", "x3"],
        "operator_weights": {"+": 10, "-": 10, "*": 10, "sin": 2},
    }
    catalog = LampleChartonCatalog.from_config(catalog_cfg)
    catalog.skeletons = {
        ("*", "<constant>", "x1"),
        ("+", "<constant>", "*", "<constant>", "x2"),
        ("+", "*", "<constant>", "x1", "*", "<constant>", "sin", "x2"),
        ("+", "<constant>", "+", "*", "<constant>", "x1", "*", "<constant>", "x3"),
    }
    catalog.skeleton_codes = catalog.compile_codes()
    return catalog


def _placeholder_source():
    from symbolic_data import ProblemSource

    return ProblemSource({
        "catalog": _placeholder_catalog(),
        "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0},
    })


# ---------------------------------------------------------------------------
# T2 — vocabulary
# ---------------------------------------------------------------------------

def test_t2_new_special_tokens_exist(tokenizer: Tokenizer) -> None:
    for token in NEW_SPECIAL_TOKENS:
        assert token in tokenizer, f"missing special token {token!r}"
        assert token in tokenizer.special_tokens

    # The compact form reuses the EXISTING <float> token (owner ruling): assert, don't add.
    assert COMPACT_TOKEN in tokenizer
    assert COMPACT_TOKEN in tokenizer.special_tokens


def test_t2_tokens_disjoint_from_expression_tokens(tokenizer: Tokenizer) -> None:
    config = load_config(get_path("configs", "test", "tokenizer.yaml"))
    expression_tokens = set(config["operators"]) | set(config["variables"])
    assert not (set(NEW_SPECIAL_TOKENS) | {COMPACT_TOKEN}) & expression_tokens


def test_t2_nibble_tokens_are_not_the_literals(tokenizer: Tokenizer) -> None:
    # <h0>..<hf> are DEDICATED hex-nibble tokens, never the literals they spell.
    assert len(NEW_SPECIAL_TOKENS) == 2 + 16
    for value, token in enumerate(NIBBLE_TOKENS):
        literal = f"{value:x}"
        assert token != literal
        if literal in tokenizer:
            assert tokenizer[token] != tokenizer[literal]
    assert tokenizer.encode(["<h0>", "<h1>"]) != tokenizer.encode(["0", "1"])


# ---------------------------------------------------------------------------
# T2 (v24.0 target format) — the tagged-dialect template vocabulary
#
# Owner ruling 2026-08-18: v24.0 trains on simplipy's TAGGED CANONICAL form, without
# masking; every numeric literal rides the ieee754 constants format, so the vocabulary
# carries NO explicit number tokens (the -10..10 integers are gone) and no gen-1 sugar.
# The tag set is verified against a LIVE engine, never against memory.
# ---------------------------------------------------------------------------

#: Prefix expressions whose canonical simplify output exercises every delimiter the
#: tagged dialect emits (n-ary bags plus the subtract/divide role markers).
_TAGGED_PROBES = [
    ["+", "x1", "*", "2", "x2"],
    ["-", "x1", "x2"],
    ["/", "x1", "x2"],
    ["+", "*", "3", "x1", "/", "x2", "-", "x3", "1"],
    ["sin", "+", "x1", "1"],
]

#: Generation-1 sugar that must NOT survive into the v24 vocabulary.
_GEN1_SUGAR = ("pow2", "pow3", "pow4", "pow5", "pow1_2", "pow1_3", "pow1_4", "pow1_5",
               "mult2", "mult3", "mult4", "mult5", "div2", "div3", "div4", "div5")


@pytest.fixture(scope="module")
def gen2_engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("base", install=True)


@pytest.fixture(scope="module")
def v24_config() -> dict[str, Any]:
    return load_config(get_path("configs", "v24-template", "tokenizer.yaml"))


def _live_tagged_outputs(engine) -> list[list[str]]:  # type: ignore[no-untyped-def]
    # simplipy >= 0.14 simplify is dialect-preserving: a prefix probe answers in prefix.
    # The tagged canonical is simplify run IN the tagged dialect (contract A3), so the
    # probe converts first. (Under 0.13, where this helper was written, tagged was
    # simplify's default output form and the conversion was implicit.)
    return [list(engine.simplify(engine.to_tagged(list(probe)))) for probe in _TAGGED_PROBES]


def _live_tag_tokens(engine) -> set[str]:  # type: ignore[no-untyped-def]
    """The delimiter tokens simplipy ACTUALLY emits in canonical tagged form."""
    return {token for output in _live_tagged_outputs(engine) for token in output
            if token.startswith("<")}


def _parses_as_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _v24_vocabulary(config: dict[str, Any]) -> list[str]:
    return list(config["special_tokens"]) + list(config["operators"]) + list(config["variables"])


def test_v24_template_carries_the_live_tagged_dialect(gen2_engine, v24_config) -> None:  # type: ignore[no-untyped-def]
    tags = _live_tag_tokens(gen2_engine)
    # Sanity on the probe set itself: the engine really does emit the n-ary delimiters.
    assert {"<add>", "</add>", "<mul>", "</mul>", "<sub>", "<div>"} <= tags

    vocabulary = set(_v24_vocabulary(v24_config))
    missing = sorted(tags - vocabulary)
    assert not missing, f"tagged-dialect tokens missing from the v24 vocabulary: {missing}"


def test_v24_template_operators_are_the_gen2_set_plus_tags(gen2_engine, v24_config) -> None:  # type: ignore[no-untyped-def]
    operators = list(v24_config["operators"])
    assert len(operators) == len(set(operators)), "duplicate operator entries"
    assert set(operators) == set(gen2_engine.operators) | _live_tag_tokens(gen2_engine)

    for sugar in _GEN1_SUGAR:
        assert sugar not in operators, f"generation-1 sugar {sugar!r} leaked into v24"


def test_v24_template_has_no_explicit_number_tokens(v24_config) -> None:  # type: ignore[no-untyped-def]
    vocabulary = _v24_vocabulary(v24_config)
    assert len(vocabulary) == len(set(vocabulary)), "duplicate vocabulary entries"

    numbers = [token for token in vocabulary if _parses_as_number(token)]
    assert numbers == [], f"v24 carries explicit number tokens {numbers}"
    # The signed-literal spelling and the non-finite literals go with them.
    for retired in ("(-1)", 'float("inf")', 'float("-inf")', 'float("nan")'):
        assert retired not in vocabulary
    # Every integer the ruling retires, spelled out.
    for integer in range(-10, 11):
        assert str(integer) not in vocabulary


def test_v24_template_carries_the_constants_format(v24_config) -> None:  # type: ignore[no-untyped-def]
    specials = list(v24_config["special_tokens"])
    for token in (IEEE754_START_TOKEN, IEEE754_END_TOKEN, *NIBBLE_TOKENS, COMPACT_TOKEN):
        assert token in specials, f"missing constants-format token {token!r}"
    # '<constant>' survives for the refiner handshake (spans map back to skeleton slots).
    assert "<constant>" in specials


def test_v24_template_encodes_a_tagged_target_end_to_end(gen2_engine, v24_config) -> None:  # type: ignore[no-untyped-def]
    """The whole point: a canonical tagged expression, with every numeric literal ridden
    out on the ieee754 hex-nibble format, encodes under the v24 vocabulary -- no masking
    step, no number tokens, no `<unk>`."""
    from flash_ansr.utils.ieee754 import wrap_float32

    v24_tokenizer = Tokenizer.from_config(v24_config)
    unk_id = v24_tokenizer["<unk>"]

    saw_span = False
    for output in _live_tagged_outputs(gen2_engine):
        serialized: list[str] = []
        for token in output:
            if _parses_as_number(token):
                serialized.extend(wrap_float32(float(token)))
                saw_span = True
            else:
                serialized.append(token)
        ids = v24_tokenizer.encode(serialized, oov="unk")
        assert unk_id not in ids, f"unencodable tagged target {serialized}"
        assert v24_tokenizer.decode(ids, special_tokens=True) == serialized
    assert saw_span, "the probes must exercise at least one literal-bearing span"


# ---------------------------------------------------------------------------
# T3 — mixing policy (per-constant independent Bernoulli(0.5), seeded RNG)
# ---------------------------------------------------------------------------

def _skeleton_with_k_constants(k: int) -> list[str]:
    tokens = ["x1"]
    for _ in range(k):
        tokens = ["+", "*", "<constant>", "x2", *tokens]
    return tokens


def _form_pattern(serialized: list[str]) -> tuple[str, ...]:
    """Per-constant E(xpanded)/C(ompact) pattern, in sequence order."""
    pattern = []
    i = 0
    while i < len(serialized):
        token = serialized[i]
        if token == "<ieee754>":
            pattern.append("E")
            end = serialized.index("</ieee754>", i)
            i = end + 1
        elif token == COMPACT_TOKEN:
            pattern.append("C")
            i += 1
        else:
            i += 1
    return tuple(pattern)


def test_t3_mixing_policy_10k_instances() -> None:
    from flash_ansr.data.serialization import serialize_constant_tokens

    rng = np.random.default_rng(0x24C1)
    counts = {1: 3000, 2: 3500, 3: 3500}  # 10k instances total
    patterns: dict[int, list[tuple[str, ...]]] = {k: [] for k in counts}

    for k, n in counts.items():
        skeleton = _skeleton_with_k_constants(k)
        for i in range(n):
            constants = [float(np.float32(c)) for c in rng.normal(0, 5, size=k)]
            serialized, numeric = serialize_constant_tokens(
                skeleton, constants, representation="ieee754_mixed", rng=rng)
            pattern = _form_pattern(serialized)
            assert len(pattern) == k
            patterns[k].append(pattern)

            # Structural: numeric channel aligned per token; values only at compact positions.
            assert len(numeric) == len(serialized)
            compact_positions = [p for p, t in enumerate(serialized) if t == COMPACT_TOKEN]
            finite_positions = [p for p, v in enumerate(numeric) if not math.isnan(v)]
            assert finite_positions == compact_positions

    tol = 0.05

    # (a) per-constant expansion rate ~ 0.5 for every slot of every k.
    for k in counts:
        arr = np.array([[1 if f == "E" else 0 for f in p] for p in patterns[k]], dtype=float)
        for slot in range(k):
            assert abs(arr[:, slot].mean() - 0.5) < tol, (k, slot, arr[:, slot].mean())

    # (b) per-constant independence: joint pattern frequencies match the product law.
    arr2 = patterns[2]
    n2 = len(arr2)
    for joint in (("E", "E"), ("E", "C"), ("C", "E"), ("C", "C")):
        freq = sum(1 for p in arr2 if p == joint) / n2
        assert abs(freq - 0.25) < tol, (joint, freq)

    # (c) both forms present within single sequences at the expected Bernoulli rate.
    for k, expected in ((2, 0.5), (3, 0.75)):
        freq = sum(1 for p in patterns[k] if len(set(p)) == 2) / len(patterns[k])
        assert abs(freq - expected) < tol, (k, freq)

    # (d) compact-history-then-expanded (a compact constant strictly before an expanded one).
    def has_compact_before_expanded(p: tuple[str, ...]) -> bool:
        return any(a == "C" and b == "E" for i, a in enumerate(p) for b in p[i + 1:])

    for k, expected in ((2, 0.25), (3, 0.5)):
        freq = sum(1 for p in patterns[k] if has_compact_before_expanded(p)) / len(patterns[k])
        assert abs(freq - expected) < tol, (k, freq)

    # (e) the inference pattern (all-compact history + current constant expanded) is
    # in-distribution: it occurs at its expected rate 2^-k.
    for k in counts:
        target = tuple(["C"] * (k - 1) + ["E"])
        freq = sum(1 for p in patterns[k] if p == target) / len(patterns[k])
        assert abs(freq - 0.5 ** k) < tol, (k, freq)


def test_t3_expanded_nibbles_encode_the_constant() -> None:
    from flash_ansr.data.serialization import serialize_constant_tokens

    rng = np.random.default_rng(7)
    value = float(np.float32(-3.75))
    # Deterministically obtain one of each form by redrawing until both were seen.
    seen = set()
    for _ in range(64):
        serialized, numeric = serialize_constant_tokens(
            ["*", "<constant>", "x1"], [value], representation="ieee754_mixed", rng=rng)
        if "<ieee754>" in serialized:
            seen.add("E")
            start = serialized.index("<ieee754>")
            end = serialized.index("</ieee754>")
            assert end - start == IEEE754_SPAN_LENGTH - 1  # 8 nibbles + tags span 10 tokens
            assert nibble_tokens_to_float32(serialized[start + 1:end]) == value
            assert all(math.isnan(v) for v in numeric[start:end + 1])  # numeric NaN across the span
        else:
            seen.add("C")
            pos = serialized.index(COMPACT_TOKEN)
            assert numeric[pos] == value
        if seen == {"E", "C"}:
            break
    assert seen == {"E", "C"}


def test_t3_generator_never_emits_nonfinite_constants() -> None:
    # Assert, don't assume: non-finite constants must raise at serialization time.
    from flash_ansr.data.serialization import serialize_constant_tokens

    rng = np.random.default_rng(0)
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            serialize_constant_tokens(["*", "<constant>", "x1"], [bad],
                                      representation="ieee754_mixed", rng=rng)


def test_t3_constant_count_mismatch_raises() -> None:
    from flash_ansr.data.serialization import serialize_constant_tokens

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        serialize_constant_tokens(["*", "<constant>", "x1"], [], representation="ieee754_mixed", rng=rng)


# ---------------------------------------------------------------------------
# The representation gate, and collation under it
# ---------------------------------------------------------------------------


def test_representation_gate() -> None:
    """ieee754_mixed is the default and the only legal value, and it demands its tokens."""
    tokenizer = Tokenizer(
        vocab=["x1", "x2", "x3"],
        special_tokens=["<pad>", "<bos>", "<eos>", "<constant>", "<expression>", "</expression>"],
    )

    class _DummyCatalog:
        simplipy_engine = None
        variables = ["x1", "x2", "x3"]

    class _DummySource:
        config = {"catalog": {"type": "lample_charton"}, "sampling": {"n_support": "prior", "n_validation": 0}}
        max_n_support = 4
        catalog = _DummyCatalog()

    with pytest.raises(ValueError):
        FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                         constant_representation="not_a_representation")

    # The default IS ieee754_mixed, so a tokenizer without the span tokens is refused
    # outright -- there is no older serialization to fall back to.
    with pytest.raises(ValueError, match="ieee754_mixed"):
        FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero")


def test_t0_collate_labels_are_unmasked_shifted_inputs(tokenizer: Tokenizer) -> None:
    """Collation itself stays byte-identical: labels = input_ids[..., 1:], no masking."""
    formatter = BatchFormatter(tokenizer=tokenizer)
    batch = {
        "input_ids": [
            [tokenizer["<bos>"], tokenizer["x1"], tokenizer["<constant>"], tokenizer["<eos>"]],
            [tokenizer["<bos>"], tokenizer["x2"], tokenizer["<eos>"]],
        ],
        "x_tensors": [torch.zeros((2, 2)), torch.zeros((2, 2))],
        "y_tensors": [torch.zeros((2, 1)), torch.zeros((2, 1))],
        "constants": [[0.5], []],
    }
    collated = formatter.collate(batch, device="cpu")
    assert collated["labels"].tolist() == collated["input_ids"][:, 1:].tolist()


# ---------------------------------------------------------------------------
# ieee754_mixed end-to-end streaming (worker serialization + numeric channel)
# ---------------------------------------------------------------------------

def _spans(ids: list[int], start_id: int, end_id: int) -> list[tuple[int, int]]:
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] == start_id:
            end = ids.index(end_id, i)
            spans.append((i, end))
            i = end + 1
        else:
            i += 1
    return spans


def test_mixed_streaming_end_to_end(tokenizer: Tokenizer) -> None:
    start_id, end_id = tokenizer["<ieee754>"], tokenizer["</ieee754>"]
    id_to_nibble = {int(tokenizer[token]): token for token in NIBBLE_TOKENS}
    nibble_ids = set(id_to_nibble)
    float_id = tokenizer[COMPACT_TOKEN]
    constant_id = tokenizer["<constant>"]
    pad_id = tokenizer["<pad>"]

    n_expanded = n_compact = 0
    with FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero",
                          constant_representation="ieee754_mixed") as dataset:
        for batch in dataset.iterate(steps=3, batch_size=16):
            for row, numeric in zip(batch["input_ids"], batch["input_num"]):
                ids = [int(t) for t in row.tolist()]
                assert constant_id not in ids  # every <constant> was serialized away

                spans = _spans(ids, start_id, end_id)
                n_expanded += len(spans)
                for start, end in spans:
                    assert end - start == IEEE754_SPAN_LENGTH - 1  # 10-token span, never cut
                    inner = ids[start + 1:end]
                    assert len(inner) == IEEE754_N_NIBBLES and set(inner) <= nibble_ids
                    # input_num is NaN across the whole expanded span.
                    assert all(math.isnan(numeric[p]) for p in range(start, end + 1))
                    # The nibbles decode to a finite float32 (big-endian hex spelling).
                    hex_string = "".join(id_to_nibble[t][2] for t in inner)
                    value = struct.unpack(">f", int(hex_string, 16).to_bytes(4, "big"))[0]
                    assert math.isfinite(value)
                    assert value == nibble_tokens_to_float32([id_to_nibble[t] for t in inner])

                # No dangling tags or bits outside complete spans.
                in_span = set()
                for start, end in spans:
                    in_span.update(range(start, end + 1))
                for p, t in enumerate(ids):
                    if p not in in_span:
                        assert t not in nibble_ids and t != start_id and t != end_id

                compact_positions = [p for p, t in enumerate(ids) if t == float_id]
                n_compact += len(compact_positions)
                for p in compact_positions:
                    assert not math.isnan(numeric[p])  # value rides the numeric channel

                # NaN elsewhere (numeric values appear ONLY at compact positions).
                finite_positions = [p for p, v in enumerate(numeric) if not math.isnan(v)]
                assert finite_positions == compact_positions

                assert ids[0] != pad_id  # sanity: rows are populated

    # Both forms must be present across the stream (48 sequences, >=1 constant each).
    assert n_expanded > 0 and n_compact > 0


def test_mixed_streaming_drops_instances_instead_of_cutting_spans(tokenizer: Tokenizer) -> None:
    """Truncation must never cut inside an <ieee754> span: offending instances are
    dropped (and counted); surviving sequences contain only intact spans."""
    start_id, end_id = tokenizer["<ieee754>"], tokenizer["</ieee754>"]
    nibble_ids = {int(tokenizer[token]) for token in NIBBLE_TOKENS}

    saw_drop_counter = False
    with FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero",
                          constant_representation="ieee754_mixed") as dataset:
        # max_seq_len=12 cannot hold any expanded span intact (10 tokens, and the body
        # never starts before index 2, so a span always straddles the truncation point):
        # every instance whose serialization expands a constant must be dropped, so
        # surviving sequences are all-compact and structurally intact.
        for batch in dataset.iterate(steps=2, batch_size=8, max_seq_len=12):
            for row in batch["input_ids"]:
                ids = [int(t) for t in row.tolist()]
                assert start_id not in ids and end_id not in ids
                assert not (set(ids) & nibble_ids)
            pool = dataset._stream.metadata_pool
            for payload in list(pool):
                if isinstance(payload, dict) and payload.get("n_dropped_truncation", 0) > 0:
                    saw_drop_counter = True
    assert saw_drop_counter, "expected the worker to count dropped mid-span instances"


def test_truncation_span_cut_detector() -> None:
    from flash_ansr.data.serialization import truncation_cuts_ieee754_span

    start, end = 100, 101
    span = [start, *[102] * 8, end]  # 10 tokens
    ids = [1, 2, *span, 3]           # len 13; the span occupies indices [2, 11]
    # Cut inside the span -> True; cut at/after the span end or before its start -> False.
    assert truncation_cuts_ieee754_span(ids, max_seq_len=10, start_id=start, end_id=end)
    assert truncation_cuts_ieee754_span(ids, max_seq_len=12, start_id=start, end_id=end)
    assert not truncation_cuts_ieee754_span(ids, max_seq_len=len(ids), start_id=start, end_id=end)
    # last_kept = 11 == the span end: the span survives whole, only the tail is cut.
    assert not truncation_cuts_ieee754_span([*ids, 4, 5], max_seq_len=13, start_id=start, end_id=end)
    assert not truncation_cuts_ieee754_span([1, 2, 3, *span], max_seq_len=3, start_id=start, end_id=end)  # span fully dropped


def test_mixed_dataset_with_preprocessor_prompting_is_guarded(tokenizer: Tokenizer) -> None:
    """The prompt serializer rebuilds the expression body from the raw skeleton and would
    silently discard the mixed serialization: iterate(preprocess=True) must refuse."""
    from flash_ansr.preprocessing import FlashANSRPreprocessor

    source = _placeholder_source()
    preprocessor = FlashANSRPreprocessor(
        simplipy_engine=source.catalog.simplipy_engine,
        tokenizer=tokenizer,
        catalog=source.catalog,
        prompt_config=load_config(get_path("configs", "test", "dataset_train.yaml"))["preprocessor"]["prompt"],
    )
    with FlashANSRDataset(source=source, tokenizer=tokenizer, padding="zero",
                          preprocessor=preprocessor,
                          constant_representation="ieee754_mixed") as dataset:
        with pytest.raises(NotImplementedError):
            next(iter(dataset.iterate(steps=1, batch_size=2, preprocess=True)))


# ---------------------------------------------------------------------------
# T4 — loss mask: CE terms whose TARGET token is <float> are exactly zero
# ---------------------------------------------------------------------------

def _tiny_model(tokenizer: Tokenizer):
    from simplipy import SimpliPyEngine
    from flash_ansr.model.flash_ansr_model import FlashANSRModel

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    engine = SimpliPyEngine.load("base", install=True)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model


def _t4_batch(tokenizer: Tokenizer) -> dict[str, Any]:
    """Two sequences; row 0 carries BOTH a prompt-span <float> and a compact-constant
    <float> inside the expression, plus an expanded <ieee754> span."""
    from flash_ansr.utils.ieee754 import wrap_float32

    span = wrap_float32(0.25)
    tokens_row0 = [
        "<bos>",
        "<prompt>", "<complexity>", "<float>", "</complexity>", "</prompt>",
        "<expression>", "x1", "*", "<float>", "+", *span, "x2", "</expression>",
        "<eos>",
    ]
    prompt_mask_row0 = [False] + [True] * 5 + [False] * (len(tokens_row0) - 6)
    numeric_row0 = [float("nan")] * len(tokens_row0)
    numeric_row0[3] = 5.0    # prompt complexity value
    numeric_row0[9] = 2.5    # compacted constant value

    tokens_row1 = ["<bos>", "<expression>", "sin", "x1", "</expression>", "<eos>"]
    prompt_mask_row1 = [False] * len(tokens_row1)
    numeric_row1 = [float("nan")] * len(tokens_row1)

    return {
        "input_ids": [tokenizer.encode(tokens_row0), tokenizer.encode(tokens_row1)],
        "input_num": [numeric_row0, numeric_row1],
        "prompt_mask": [prompt_mask_row0, prompt_mask_row1],
        "x_tensors": [torch.randn(6, 10), torch.randn(6, 10)],
        "y_tensors": [torch.randn(6, 1), torch.randn(6, 1)],
        "constants": [[2.5, 0.25], []],
    }


def _pipeline_masks(trainer, batch: dict[str, Any]) -> None:
    trainer._apply_prompt_mask(batch)
    trainer._apply_float_target_mask(batch)


def test_t4_float_target_loss_is_exactly_zero(tokenizer: Tokenizer) -> None:
    from flash_ansr.train import Trainer

    model = _tiny_model(tokenizer)
    pad_id = tokenizer["<pad>"]
    float_id = tokenizer[COMPACT_TOKEN]

    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.metrics_ignore_index = pad_id
    trainer.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    formatter = BatchFormatter(tokenizer=tokenizer)
    batch = formatter.collate(_t4_batch(tokenizer), device="cpu")
    input_ids = batch["input_ids"]

    with torch.no_grad():
        data = torch.cat([batch["x_tensors"], batch["y_tensors"]], dim=-1)
        logits = model(input_ids, data, input_num=batch.get("input_num"),
                       data_attn_mask=batch["data_attn_mask"])

    target_is_float = input_ids[..., 1:] == float_id
    assert target_is_float.any(), "test batch must contain <float> targets"
    # Row 0 has one prompt-span <float> target and one expression <float> target.
    assert int(target_is_float[0].sum()) == 2

    def pipeline_loss(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work = dict(batch)
        work["labels"] = labels.clone()
        _pipeline_masks(trainer, work)
        flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        flat_labels = work["labels"].reshape(-1)
        total = trainer.cross_entropy_loss(flat_logits, flat_labels)
        per_position = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, ignore_index=pad_id, reduction="none"
        ).reshape(labels.shape)
        return total, per_position

    labels = batch["labels"]
    total, per_position = pipeline_loss(labels)

    # The CE terms whose TARGET token is <float> are exactly zero — prompt span AND
    # expression span alike.
    assert torch.all(per_position[target_is_float] == 0.0)

    # Loss at bit/tag positions is NONZERO (the model must learn to emit the bits).
    bit_tag_ids = torch.tensor([tokenizer[t] for t in NEW_SPECIAL_TOKENS])
    target_is_bit_or_tag = torch.isin(input_ids[..., 1:], bit_tag_ids)
    assert target_is_bit_or_tag.any()
    assert torch.all(per_position[target_is_bit_or_tag] > 0.0)

    # Perturbing the label at <float>-target positions leaves the total loss unchanged.
    perturbed = labels.clone()
    perturbed[target_is_float] = tokenizer["x2"]
    total_perturbed, _ = pipeline_loss(perturbed)
    assert torch.equal(total, total_perturbed)

    # Perturbing a bit-target label must CHANGE the loss (guards against over-masking).
    perturbed_bits = labels.clone()
    perturbed_bits[target_is_bit_or_tag] = tokenizer["x2"]
    total_bits, _ = pipeline_loss(perturbed_bits)
    assert not torch.equal(total, total_bits)


def test_t4_mask_is_keyed_on_shifted_targets_not_inputs(tokenizer: Tokenizer) -> None:
    """Guards the off-by-one: the term AT the <float> position (whose target is the
    NEXT token) must stay unmasked; the term whose TARGET is <float> is masked."""
    from flash_ansr.data.collate import mask_float_targets

    pad_id = tokenizer["<pad>"]
    float_id = tokenizer[COMPACT_TOKEN]
    row = ["<bos>", "<expression>", "x1", "*", "<float>", "+", "x2", "</expression>", "<eos>"]
    input_ids = torch.tensor([tokenizer.encode(row)])
    labels = input_ids[..., 1:].clone()

    masked = mask_float_targets(labels, input_ids, float_id, pad_id)

    q = row.index("<float>")
    # labels[p] is the target of position p (= input token p+1).
    assert masked[0, q - 1] == pad_id                    # target IS <float> -> masked
    assert masked[0, q] == tokenizer["+"]                # context is <float>, target '+': NOT masked
    for p, token in enumerate(row[1:]):
        if token != "<float>":
            assert masked[0, p] == tokenizer[token]


# ---------------------------------------------------------------------------
# Peripheral: constant counting must see one constant per form, not 10/1 tokens
# ---------------------------------------------------------------------------

def test_scoring_counts_each_constant_form_as_one() -> None:
    from flash_ansr.scoring import count_constants, is_constant_token
    from flash_ansr.utils.ieee754 import wrap_float32

    # Compact form: one constant.
    assert is_constant_token("<float>") is True
    assert count_constants(["*", "<float>", "x1"]) == 1

    # Expanded form: the whole 10-token span is ONE constant (opened by <ieee754>;
    # nibbles and the closing tag never count on their own, keeping naive per-token
    # sums over a span exact).
    assert is_constant_token("<ieee754>") is True
    for token in (*NIBBLE_TOKENS, "</ieee754>"):
        assert is_constant_token(token) is False
    assert count_constants(["*", *wrap_float32(2.5), "x1"]) == 1

    # Mixed expression: compact + expanded + legacy placeholder = three constants.
    expression = ["+", "*", "<float>", "x1", "+", "*", *wrap_float32(-1.5), "x2", "<constant>"]
    assert count_constants(expression) == 3


def test_t4_trainer_applies_float_mask_in_both_train_and_val() -> None:
    """The seam must be wired into both _train_step and _validate_step."""
    import inspect
    from flash_ansr.train import Trainer

    train_src = inspect.getsource(Trainer._train_step)
    val_src = inspect.getsource(Trainer._validate_step)
    assert "_apply_float_target_mask" in train_src
    assert "_apply_float_target_mask" in val_src
