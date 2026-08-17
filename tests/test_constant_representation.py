"""Acceptance tests T0/T2/T3/T4 for the v24 `ieee754_mixed` constants representation.

Pre-registered integration contract, lane C1 (vocabulary + data serialization with
per-constant 50/50 mixing + loss mask), gated behind the `constant_representation`
config key ('v23' default = byte-identical to current behavior).

T1 (codec round-trip) lives in tests/test_ieee754.py.

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


NEW_SPECIAL_TOKENS = ["<ieee754>", "</ieee754>", "<b0>", "<b1>"]
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


def test_t2_bit_tokens_are_not_the_literals(tokenizer: Tokenizer) -> None:
    # <b0>/<b1> are DEDICATED bit tokens, not the numeric literals 0/1.
    assert "<b0>" != "0" and "<b1>" != "1"
    assert tokenizer["<b0>"] != tokenizer["0"]
    assert tokenizer["<b1>"] != tokenizer["1"]
    assert tokenizer.encode(["<b0>", "<b1>"]) != tokenizer.encode(["0", "1"])


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


def test_t3_expanded_bits_encode_the_constant() -> None:
    from flash_ansr.data.serialization import serialize_constant_tokens
    from flash_ansr.utils.ieee754 import bit_tokens_to_float32

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
            assert end - start == 33  # 32 bits + tags span 34 tokens
            assert bit_tokens_to_float32(serialized[start + 1:end]) == value
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
# T0 — regression: absent / 'v23' representation is byte-identical to today
# ---------------------------------------------------------------------------

def test_t0_serialize_v23_is_identity() -> None:
    from flash_ansr.data.serialization import serialize_constant_tokens

    skeleton = ["+", "<constant>", "*", "<constant>", "x2"]
    constants = [1.5, -2.5]
    for kwargs in ({}, {"representation": "v23"}):
        serialized, numeric = serialize_constant_tokens(skeleton, constants, **kwargs)
        assert serialized == skeleton
        assert all(math.isnan(v) for v in numeric)  # v23 numeric channel stays out-of-band


def test_t0_dataset_defaults_to_v23() -> None:
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

    with FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero") as dataset:
        assert dataset.constant_representation == "v23"

    with pytest.raises(ValueError):
        FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                         constant_representation="not_a_representation")

    # ieee754_mixed requires the new special tokens: fail fast on a v23 tokenizer.
    with pytest.raises(ValueError):
        FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                         constant_representation="ieee754_mixed")


def test_t0_streaming_output_unchanged_without_config_key(tokenizer: Tokenizer) -> None:
    """With `constant_representation` absent the stream must look exactly like today."""
    new_token_ids = {tokenizer[token] for token in NEW_SPECIAL_TOKENS}
    float_id = tokenizer[COMPACT_TOKEN]
    constant_id = tokenizer["<constant>"]
    expected_keys = {"x_tensors", "y_tensors", "data_attn_mask", "input_ids", "constants",
                     "skeleton", "skeleton_hash", "expression", "n_support", "input_num"}

    with FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero") as dataset:
        for batch in dataset.iterate(steps=2, batch_size=8):
            assert set(batch.keys()) == expected_keys
            for row, numeric, constants in zip(batch["input_ids"], batch["input_num"], batch["constants"]):
                ids = [int(t) for t in row.tolist()]
                # No v24 token ever appears in a v23 stream.
                assert not (set(ids) & new_token_ids)
                assert float_id not in ids
                # The numeric channel carries the fitted values exactly at <constant>
                # positions, in order — current (v23) behavior.
                const_positions = [p for p, t in enumerate(ids) if t == constant_id]
                finite_positions = [p for p, v in enumerate(numeric) if not math.isnan(v)]
                assert finite_positions == const_positions
                values = [numeric[p] for p in const_positions]
                assert values == pytest.approx([float(c) for c in constants])


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
    bit_ids = {tokenizer["<b0>"], tokenizer["<b1>"]}
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
                    assert end - start == 33  # 34-token span, never cut
                    inner = ids[start + 1:end]
                    assert len(inner) == 32 and set(inner) <= bit_ids
                    # input_num is NaN across the whole expanded span.
                    assert all(math.isnan(numeric[p]) for p in range(start, end + 1))
                    # The bits decode to a finite float32.
                    bits = "".join("1" if t == tokenizer["<b1>"] else "0" for t in inner)
                    value = struct.unpack(">f", int(bits, 2).to_bytes(4, "big"))[0]
                    assert math.isfinite(value)

                # No dangling tags or bits outside complete spans.
                in_span = set()
                for start, end in spans:
                    in_span.update(range(start, end + 1))
                for p, t in enumerate(ids):
                    if p not in in_span:
                        assert t not in bit_ids and t != start_id and t != end_id

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
    bit_ids = {tokenizer["<b0>"], tokenizer["<b1>"]}

    saw_drop_counter = False
    with FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero",
                          constant_representation="ieee754_mixed") as dataset:
        # max_seq_len=16 cannot hold any expanded span (34 tokens): every instance whose
        # serialization expands a constant must be dropped, so surviving sequences are
        # all-compact and structurally intact.
        for batch in dataset.iterate(steps=2, batch_size=8, max_seq_len=16):
            for row in batch["input_ids"]:
                ids = [int(t) for t in row.tolist()]
                assert start_id not in ids and end_id not in ids
                assert not (set(ids) & bit_ids)
            pool = dataset._stream.metadata_pool
            for payload in list(pool):
                if isinstance(payload, dict) and payload.get("n_dropped_truncation", 0) > 0:
                    saw_drop_counter = True
    assert saw_drop_counter, "expected the worker to count dropped mid-span instances"


def test_truncation_span_cut_detector() -> None:
    from flash_ansr.data.serialization import truncation_cuts_ieee754_span

    start, end = 100, 101
    span = [start, *[102] * 32, end]  # 34 tokens
    ids = [1, 2, *span, 3]
    # Cut inside the span -> True; cut at/after the span end or before its start -> False.
    assert truncation_cuts_ieee754_span(ids, max_seq_len=10, start_id=start, end_id=end)
    assert truncation_cuts_ieee754_span(ids, max_seq_len=35, start_id=start, end_id=end)
    assert not truncation_cuts_ieee754_span(ids, max_seq_len=len(ids), start_id=start, end_id=end)
    assert not truncation_cuts_ieee754_span(ids, max_seq_len=37, start_id=start, end_id=end)  # span intact, tail cut
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


def test_t4_trainer_applies_float_mask_in_both_train_and_val() -> None:
    """The seam must be wired into both _train_step and _validate_step."""
    import inspect
    from flash_ansr.train import Trainer

    train_src = inspect.getsource(Trainer._train_step)
    val_src = inspect.getsource(Trainer._validate_step)
    assert "_apply_float_target_mask" in train_src
    assert "_apply_float_target_mask" in val_src
