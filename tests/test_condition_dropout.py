"""Condition dropout (owner ruling): "I'd like to have 10% condition dropout, meaning in
10% of the training instances, the data tokens are not included in the sequence and the
model has to learn to predict unconditionally."

In the CURRENT architecture (cross-attention over SetTransformer memory) this is
implemented via the EXISTING optional-condition/null-memory routing: a
`condition_dropout: 0.10` data config key drops the condition per instance with
probability 0.10 (seeded draw seam), routing the null-memory path during training.
Under the planned v24 self-attn-only decoder this becomes span omission — the data
tokens are literally omitted from the sequence — the cleaner formulation.

T3-style acceptance: dropout rate over N collated instances within tolerance, plus
seeded determinism of the draw seam.

NOTE on the test engine: the configs/test bundle references the generation-1 'dev_7-3'
simplipy asset (pre-existing baseline condition); streaming tests build a generation-2
'base'-engine catalog inline, the pattern lane C1 established.
"""
import numpy as np
import pytest
import torch

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.model.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


def _placeholder_catalog_config() -> dict:
    """A generation-2 ('base' engine) catalog config pinned to placeholder-form skeletons."""
    return {
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


def _placeholder_source():  # type: ignore[no-untyped-def]
    from symbolic_data import ProblemSource
    from symbolic_data.generative import LampleChartonCatalog

    catalog = LampleChartonCatalog.from_config(_placeholder_catalog_config())
    catalog.skeletons = {
        ("*", "<constant>", "x1"),
        ("+", "<constant>", "*", "<constant>", "x2"),
        ("+", "*", "<constant>", "x1", "*", "<constant>", "sin", "x2"),
    }
    catalog.skeleton_codes = catalog.compile_codes()
    return ProblemSource({
        "catalog": catalog,
        "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0},
    })


# ---------------------------------------------------------------------------
# The draw seam: seeded determinism + rate (the T3-style RNG discipline)
# ---------------------------------------------------------------------------

def test_condition_dropout_draw_seam_rate_and_determinism() -> None:
    from flash_ansr.data.streaming import draw_condition_mask

    rng = np.random.default_rng(0x24D0)
    draws = [draw_condition_mask(rng, 0.10) for _ in range(10_000)]
    assert all(isinstance(d, bool) for d in draws)

    # Rate: ~10% of instances are DROPPED (condition_mask False -> null-memory route).
    dropped_rate = 1.0 - float(np.mean(draws))
    assert abs(dropped_rate - 0.10) < 0.015, dropped_rate

    # Seeded determinism: the same seed reproduces the same draw sequence exactly.
    rng_again = np.random.default_rng(0x24D0)
    assert [draw_condition_mask(rng_again, 0.10) for _ in range(10_000)] == draws

    # A different seed gives a different sequence (the draws actually consume the rng).
    rng_other = np.random.default_rng(0x24D1)
    assert [draw_condition_mask(rng_other, 0.10) for _ in range(10_000)] != draws

    # Degenerate probabilities behave exactly.
    rng = np.random.default_rng(0)
    assert all(draw_condition_mask(rng, 0.0) for _ in range(100))
    assert not any(draw_condition_mask(rng, 1.0) for _ in range(100))


# ---------------------------------------------------------------------------
# The config key: `condition_dropout` (v24 canonical name for the routing probability)
# ---------------------------------------------------------------------------

class _DummyCatalog:
    simplipy_engine = None
    variables = ["x1", "x2", "x3"]


class _DummySource:
    config = {"catalog": {"type": "lample_charton"}, "sampling": {"n_support": "prior", "n_validation": 0}}
    max_n_support = 4
    catalog = _DummyCatalog()


def _dummy_tokenizer() -> Tokenizer:
    return Tokenizer(
        vocab=["x1", "x2", "x3"],
        special_tokens=["<pad>", "<bos>", "<eos>", "<constant>", "<expression>", "</expression>"],
    )


def test_condition_dropout_constructor_key() -> None:
    tokenizer = _dummy_tokenizer()

    with FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                          condition_dropout=0.10) as dataset:
        assert dataset.condition_dropout == 0.10
        # The key maps onto the existing optional-condition routing probability.
        assert dataset.unconditional_prob == 0.10

    # Default: no dropout.
    with FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero") as dataset:
        assert dataset.condition_dropout == 0.0

    # The legacy alias still works and must agree when both are given.
    with FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                          unconditional_prob=0.15) as dataset:
        assert dataset.condition_dropout == 0.15

    with pytest.raises(ValueError):
        FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                         condition_dropout=0.10, unconditional_prob=0.15)

    # Agreeing values are fine.
    with FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero",
                          condition_dropout=0.10, unconditional_prob=0.10) as dataset:
        assert dataset.condition_dropout == 0.10


def test_condition_dropout_from_config_key() -> None:
    config = {
        "source": {
            "catalog": _placeholder_catalog_config(),
            "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0},
        },
        "tokenizer": get_path("configs", "test", "tokenizer.yaml"),
        "padding": "zero",
        "condition_dropout": 0.10,
    }
    with FlashANSRDataset.from_config(config) as dataset:
        assert dataset.condition_dropout == 0.10

    conflicting = dict(config)
    conflicting["unconditional_prob"] = 0.2
    with pytest.raises(ValueError):
        FlashANSRDataset.from_config(conflicting)


# ---------------------------------------------------------------------------
# T3-style: dropout rate over N collated instances within tolerance
# ---------------------------------------------------------------------------

def test_condition_dropout_rate_over_collated_instances(tokenizer: Tokenizer) -> None:
    n_batches, batch_size = 25, 16
    masks: list[bool] = []
    with FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero",
                          condition_dropout=0.10) as dataset:
        for batch in dataset.iterate(steps=n_batches, batch_size=batch_size):
            assert "condition_mask" in batch, "condition_dropout must emit condition_mask"
            collated = dataset.collate(batch, device="cpu")
            mask = collated["condition_mask"]
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.bool and mask.shape == (batch_size,)
            masks.extend(bool(m) for m in mask.tolist())

    n = len(masks)
    assert n == n_batches * batch_size
    dropped_rate = 1.0 - (sum(masks) / n)
    # N=400 Bernoulli(0.1): sigma ~ 0.015 -> 0.06 is a 4-sigma band.
    assert abs(dropped_rate - 0.10) < 0.06, dropped_rate
    # Both routes must actually occur in the stream.
    assert any(masks) and not all(masks)


def test_no_condition_dropout_keeps_stream_byte_identical(tokenizer: Tokenizer) -> None:
    """Without the key the feature is off: no condition_mask ever enters a batch."""
    with FlashANSRDataset(source=_placeholder_source(), tokenizer=tokenizer, padding="zero") as dataset:
        for batch in dataset.iterate(steps=2, batch_size=4):
            assert "condition_mask" not in batch
