"""C5 consumer wiring: ``target_dialect='tagged'`` streams the engine's TAGGED CANONICAL
output as the training target (contract A3) -- simplify run IN the tagged dialect per
problem, every NUMERIC literal serialized onto the ieee754 constants format, np.pi/np.e
kept symbolic, and no explicit-operator spelling in any target. 'explicit' (default)
keeps today's prefix targets byte-identical."""
import math

import pytest

from flash_ansr import FlashANSRDataset, get_path
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
from flash_ansr.utils.skeleton import mask_literals_positional

COMPACT_TOKEN = "<float>"
TAGGED_DELIMITERS = ("<add>", "</add>", "<sub>", "<mul>", "</mul>", "<div>")
EXPLICIT_OPERATORS = ("+", "-", "*", "/")


@pytest.fixture(scope="module")
def gen2_engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("base", install=True)


@pytest.fixture(scope="module")
def v24_tokenizer() -> Tokenizer:
    return Tokenizer.from_config(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))


def _placeholder_catalog():  # type: ignore[no-untyped-def]
    """The generation-2 'base'-engine catalog with pinned placeholder-form skeletons
    (mirrors tests/test_constant_representation.py -- the streaming worker substitutes
    the carried constants first, so both problem shapes take the same path)."""
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


def _placeholder_source():  # type: ignore[no-untyped-def]
    from symbolic_data import ProblemSource

    return ProblemSource({
        "catalog": _placeholder_catalog(),
        "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0},
    })


# ---------------------------------------------------------------------------
# Fail-fast configuration guards
# ---------------------------------------------------------------------------

def test_unknown_target_dialect_rejected(v24_tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="target_dialect"):
        FlashANSRDataset(source=_placeholder_source(), tokenizer=v24_tokenizer, padding="zero",
                         constant_representation="ieee754_mixed", target_dialect="prefix")


def test_tagged_requires_the_delimiter_tokens() -> None:
    # Mixed-capable vocabulary WITHOUT the tagged delimiters: the tagged gate must
    # fail fast, not stream OOV targets.
    tokenizer = Tokenizer(
        vocab=["x1", "x2", "x3", "sin"],
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<constant>", COMPACT_TOKEN,
                        IEEE754_START_TOKEN, IEEE754_END_TOKEN, *NIBBLE_TOKENS],
    )
    with pytest.raises(ValueError, match="<add>"):
        FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero",
                         constant_representation="ieee754_mixed", target_dialect="tagged")


# ---------------------------------------------------------------------------
# The keep_specials masking policy (unit)
# ---------------------------------------------------------------------------

def test_keep_specials_leaves_symbolic_constants_and_excludes_their_values(gen2_engine) -> None:  # type: ignore[no-untyped-def]
    skeleton, values = mask_literals_positional(gen2_engine, ["*", "np.pi", "2.5"], keep_specials=True)
    assert skeleton == ["*", "np.pi", "<constant>"]
    assert values == [2.5]

    # Tagged dialect input, same policy: the bag walk is native.
    skeleton, values = mask_literals_positional(
        gen2_engine, ["<mul>", "np.e", "3", "x1", "</mul>"], keep_specials=True)
    assert skeleton == ["<mul>", "np.e", "<constant>", "x1", "</mul>"]
    assert values == [3.0]


def test_default_policy_still_masks_specials(gen2_engine) -> None:  # type: ignore[no-untyped-def]
    skeleton, values = mask_literals_positional(gen2_engine, ["*", "np.pi", "2.5"])
    assert skeleton == ["*", "<constant>", "<constant>"]
    assert values == pytest.approx([math.pi, 2.5])


# ---------------------------------------------------------------------------
# Tagged streaming end-to-end
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


def test_tagged_streaming_end_to_end(v24_tokenizer: Tokenizer) -> None:
    start_id, end_id = v24_tokenizer[IEEE754_START_TOKEN], v24_tokenizer[IEEE754_END_TOKEN]
    id_to_nibble = {int(v24_tokenizer[token]): token for token in NIBBLE_TOKENS}
    nibble_ids = set(id_to_nibble)
    float_id = v24_tokenizer[COMPACT_TOKEN]
    constant_id = v24_tokenizer["<constant>"]
    unk_id = v24_tokenizer["<unk>"]
    explicit_ids = {int(v24_tokenizer[op]) for op in EXPLICIT_OPERATORS}
    delimiter_ids = {int(v24_tokenizer[token]) for token in TAGGED_DELIMITERS}

    saw_delimiter = False
    n_expanded = n_compact = 0
    with FlashANSRDataset(source=_placeholder_source(), tokenizer=v24_tokenizer, padding="zero",
                          constant_representation="ieee754_mixed",
                          target_dialect="tagged") as dataset:
        for batch in dataset.iterate(steps=3, batch_size=16):
            for row, numeric, skeleton, expression in zip(
                    batch["input_ids"], batch["input_num"], batch["skeleton"], batch["expression"]):
                ids = [int(t) for t in row.tolist()]

                # The target is the tagged canonical: no <constant> carrier survives
                # serialization, no <unk>, and NO explicit-operator spelling anywhere --
                # the delimiters carry +,-,*,/ (contract A3).
                assert constant_id not in ids
                assert unk_id not in ids
                assert not (set(ids) & explicit_ids)
                if set(ids) & delimiter_ids:
                    saw_delimiter = True

                # Spans are intact 10-token hex spellings of finite float32s.
                spans = _spans(ids, start_id, end_id)
                n_expanded += len(spans)
                for start, end in spans:
                    assert end - start == IEEE754_SPAN_LENGTH - 1
                    inner = ids[start + 1:end]
                    assert len(inner) == IEEE754_N_NIBBLES and set(inner) <= nibble_ids
                    assert math.isfinite(nibble_tokens_to_float32([id_to_nibble[t] for t in inner]))
                in_span = set()
                for start, end in spans:
                    in_span.update(range(start, end + 1))
                for position, token_id in enumerate(ids):
                    if position not in in_span:
                        assert token_id not in nibble_ids and token_id != start_id and token_id != end_id

                # Numeric channel: finite exactly at compact <float> positions.
                compact_positions = [p for p, t in enumerate(ids) if t == float_id]
                n_compact += len(compact_positions)
                finite_positions = [p for p, v in enumerate(numeric) if not math.isnan(v)]
                assert finite_positions == compact_positions

                # Metadata: the skeleton is the tagged target carrier (no explicit
                # operator spellings), the expression stays the concrete PREFIX ground
                # truth (executable, exactly as substituted).
                assert not (set(skeleton) & set(EXPLICIT_OPERATORS))
                assert any(op in expression for op in EXPLICIT_OPERATORS)

    assert saw_delimiter, "no tagged delimiter ever appeared -- the stream is not tagged"
    assert n_expanded > 0 and n_compact > 0
