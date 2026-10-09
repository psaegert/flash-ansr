"""Units augmentation (flash_ansr.data.units; owner rulings 2026-10-08/09): a change of units along the law's own
dimensional symmetry. The dimension inference on hand-derived laws, the integer lattice and its guards, exactness of
the re-valued target on the scaled data, the identity / fallback paths, and the stream wiring end to end."""
import math

import numpy as np
import pytest

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.data.units import (
    STATUS_APPLIED,
    STATUS_APPLIED_Y,
    STATUS_FIXED,
    STATUS_HOLDOUT,
    STATUS_IDENTITY,
    STATUS_UNPARSED,
    UNITS_COUNTER_KEYS,
    apply_units,
    law_dimensions,
    literal_slots,
    validate_units_block,
)
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.metrics import build_expression_callable
from flash_ansr.utils.skeleton import mask_literals_positional

VARIABLES = [f"x{i}" for i in range(1, 6)]
CFG = {"p_identity": 0.0, "log2_scale_max": 100, "log2_literal_max": 100, "max_redraws": 400}


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("acj-4-3", install=True)


def _dims(engine, law: str):  # type: ignore[no-untyped-def]
    return law_dimensions(engine, law.split(), VARIABLES)


# ---------------------------------------------------------------------------------------------- config

def test_units_block_is_pinned_exactly() -> None:
    assert validate_units_block(None) is None
    assert validate_units_block(dict(CFG)) == CFG
    with pytest.raises(ValueError, match="exactly"):
        validate_units_block({k: v for k, v in CFG.items() if k != "max_redraws"})
    with pytest.raises(ValueError, match="exactly"):
        validate_units_block({**CFG, "rule": "I"})
    with pytest.raises(ValueError, match="probability"):
        validate_units_block({**CFG, "p_identity": 1.5})
    with pytest.raises(ValueError, match="positive integer"):
        validate_units_block({**CFG, "log2_scale_max": 2.5})


# ---------------------------------------------------------------------------------------------- dimensions

@pytest.mark.parametrize("law, x_movable, y_free_alone", [
    ("<mul> x1 x2 <div> x3 </mul>", True, False),          # constant-free monomial: y follows the inputs
    ("<mul> 2.5 x1 x2 <div> x3 </mul>", True, True),       # a sampled coefficient also frees y
    ("<mul> -1 x1 x2 <div> x3 </mul>", True, False),       # a sign is a pure number (rule I)
    ("<add> <mul> 2 x3 </mul> x4 </add>", True, False),    # a count is a pure number: x3, x4 share a unit
    ("<add> x1 <sub> x2 </add>", True, False),
    ("sin x1", False, False),                               # a bare variable inside sin cannot move
    ("<mul> -1 sin x1 </mul>", False, False),
    ("<mul> 2.5 sin x1 </mul>", False, True),               # only y moves, absorbed by the coefficient
    ("sin <mul> 2.7 x1 </mul>", True, False),               # omega * t: the frequency absorbs the time unit
    ("pow x1 2", True, False),
    ("exp <mul> 0.3 x1 <div> x2 </mul>", True, False),
])
def test_dimension_classes_on_hand_derived_laws(engine, law: str, x_movable: bool, y_free_alone: bool) -> None:  # type: ignore[no-untyped-def]
    dims = _dims(engine, law)
    assert (dims.x_movable, dims.y_free_alone) == (x_movable, y_free_alone)


def _laws_data(engine, law: str, rng: np.random.Generator, n: int = 64):  # type: ignore[no-untyped-def]
    """Concrete data for a law spelled in the tagged dialect (inputs in [0.5, 4])."""
    tokens = law.split()
    _, values = mask_literals_positional(engine, tokens, keep_specials=True)
    x = rng.uniform(0.5, 4.0, size=(n, len(VARIABLES)))
    f = build_expression_callable(engine, list(engine.to_prefix(tokens)), VARIABLES)
    y = np.asarray(f(x.T.copy()), dtype=np.float64).reshape(-1, 1)
    return tokens, np.asarray(values, dtype=np.float64), x, y


def _evaluate(engine, tokens: list[str], x: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    f = build_expression_callable(engine, list(engine.to_prefix(tokens)), VARIABLES)
    return np.asarray(f(x.T.copy()), dtype=np.float64).reshape(-1, 1)


@pytest.mark.parametrize("law", [
    "<mul> 2.5 x1 x2 <div> x3 </mul>",
    "<mul> x1 x2 <div> x3 </mul>",
    "sin <mul> 2.7 x1 </mul>",
    "<add> <mul> 2.7 x3 </mul> x4 </add>",
    "exp <mul> 0.3 x1 <div> x2 </mul>",
    "<mul> 1.5 rootn x1 3 </mul>",
])
def test_revalued_target_reproduces_the_scaled_data(engine, law: str) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    for seed in range(20):
        tokens, literals, x, y = _laws_data(engine, law, rng)
        result = apply_units(engine=engine, target_tokens=tokens, literals=literals, x_support=x, y_support=y,
                             y_encoder=y.copy(), variables=VARIABLES, rng=np.random.default_rng(seed), cfg=CFG,
                             is_held_out=None)
        assert result.status in (STATUS_APPLIED, STATUS_APPLIED_Y)
        change = result.change
        # Powers of two on an integer lattice, inside the bounds.
        assert all(isinstance(v, int) and abs(v) <= 100 for v in change.z.values())
        assert isinstance(change.zy, int) and abs(change.zy) <= 100
        assert all(isinstance(v, int) for v in change.w.values())
        # The target's tokens are untouched except for the re-valued literal sites ...
        positions, _ = literal_slots(engine, tokens)
        for i, (old, new) in enumerate(zip(tokens, result.revalued_tokens)):
            if i not in positions:
                assert old == new
        # ... and every literal stays inside the box.
        assert np.all(np.abs(np.log2(np.abs(result.literals[result.literals != 0]))) <= 100)
        # The re-valued law on the scaled inputs IS the scaled output.
        y_hat = _evaluate(engine, result.revalued_tokens, result.x_support)
        np.testing.assert_allclose(y_hat, result.y_support, rtol=1e-12)
        # The inputs scaled by exact powers of two, the output likewise.
        for var, zv in change.z.items():
            j = VARIABLES.index(var)
            np.testing.assert_array_equal(result.x_support[:, j], np.ldexp(x[:, j], zv))
        np.testing.assert_array_equal(result.y_support, np.ldexp(y, change.zy))


def test_integer_literals_are_pure_numbers(engine) -> None:  # type: ignore[no-untyped-def]
    """'2 * x3 + x4': the count 2 never absorbs a unit, so x3, x4 and y share one scale."""
    rng = np.random.default_rng(1)
    for seed in range(30):
        tokens, literals, x, y = _laws_data(engine, "<add> <mul> 2 x3 </mul> x4 </add>", rng)
        result = apply_units(engine=engine, target_tokens=tokens, literals=literals, x_support=x, y_support=y,
                             y_encoder=y.copy(), variables=VARIABLES, rng=np.random.default_rng(seed), cfg=CFG,
                             is_held_out=None)
        assert result.change.z["x3"] == result.change.z["x4"] == result.change.zy
        np.testing.assert_array_equal(result.literals, literals)


def test_identity_share_and_laws_without_symmetry(engine) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(2)
    tokens, literals, x, y = _laws_data(engine, "<mul> 2.5 x1 x2 <div> x3 </mul>", rng)
    result = apply_units(engine=engine, target_tokens=tokens, literals=literals, x_support=x, y_support=y,
                         y_encoder=y, variables=VARIABLES, rng=rng, cfg={**CFG, "p_identity": 1.0}, is_held_out=None)
    assert result.status == STATUS_IDENTITY and result.x_support is x and result.literals is literals

    tokens, literals, x, y = _laws_data(engine, "sin x1", rng)
    result = apply_units(engine=engine, target_tokens=tokens, literals=literals, x_support=x, y_support=y,
                         y_encoder=y, variables=VARIABLES, rng=rng, cfg=CFG, is_held_out=None)
    assert result.status == STATUS_FIXED and result.x_support is x


def test_holdout_hit_keeps_identity_units(engine) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(3)
    tokens, literals, x, y = _laws_data(engine, "<mul> 2.5 x1 x2 <div> x3 </mul>", rng)
    seen: list[list[str]] = []

    def held(revalued: list[str]) -> bool:
        seen.append(revalued)
        return True

    result = apply_units(engine=engine, target_tokens=tokens, literals=literals, x_support=x, y_support=y,
                         y_encoder=y, variables=VARIABLES, rng=rng, cfg=CFG, is_held_out=held)
    assert result.status == STATUS_HOLDOUT and result.x_support is x and result.literals is literals
    assert seen and seen[-1] != tokens


def test_draws_stay_finite_and_normal_near_float64_limits(engine) -> None:  # type: ignore[no-untyped-def]
    """Inputs near 1e300: most upward draws would overflow; whatever is applied stays finite and normal."""
    rng = np.random.default_rng(4)
    tokens, literals, x, y = _laws_data(engine, "<mul> 2.5 x1 x2 <div> x3 </mul>", rng)
    x = x.copy()
    x[:, 0] *= 1e300
    y = y * 1e300
    for seed in range(20):
        result = apply_units(engine=engine, target_tokens=tokens, literals=literals, x_support=x, y_support=y,
                             y_encoder=y, variables=VARIABLES, rng=np.random.default_rng(seed), cfg=CFG,
                             is_held_out=None)
        for a in (result.x_support[:, :3], result.y_support, result.literals):
            assert np.all(np.isfinite(a))
            nz = np.abs(a[a != 0])
            assert nz.size == 0 or nz.min() >= np.finfo(np.float64).tiny


def test_reproduces_rejects_intermediate_overflow(engine) -> None:  # type: ignore[no-untyped-def]
    """x1 * x2 / x3 with every input near 1e200: the output is ~1e200, the intermediate product is not finite."""
    from flash_ansr.data.units import reproduces

    rng = np.random.default_rng(5)
    x = rng.uniform(1.0, 2.0, size=(32, len(VARIABLES)))
    tokens = "<mul> x1 x2 <div> x3 </mul>".split()
    y = (x[:, 0] * x[:, 1] / x[:, 2]).reshape(-1, 1)
    assert reproduces(engine, tokens, x, y, VARIABLES)
    big = x.copy()
    big[:, :3] *= 1e200
    assert not reproduces(engine, tokens, big, y * 1e200, VARIABLES)
    assert not reproduces(engine, tokens, x, y * 1.001, VARIABLES)


def test_operator_outside_the_rules_keeps_identity_units(engine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Every operator of the shipped engines is covered (checked when this was written); an operator added later
    must fall back to identity units, never guess a dimension."""
    import flash_ansr.data.units as units

    class _Engine:
        operator_arity_compat = {**engine.operator_arity_compat, "mystery": 1}

    monkeypatch.setattr(units, "literal_slots", lambda _engine, _tokens: ([], []))
    x = np.ones((4, len(VARIABLES)))
    y = np.ones((4, 1))
    result = apply_units(engine=_Engine(), target_tokens=["mystery", "x1"], literals=np.zeros(0), x_support=x,
                         y_support=y, y_encoder=y, variables=VARIABLES, rng=np.random.default_rng(0), cfg=CFG,
                         is_held_out=None)
    assert result.status == STATUS_UNPARSED and result.x_support is x


# ---------------------------------------------------------------------------------------------- stream

@pytest.fixture(scope="module")
def v24_tokenizer() -> Tokenizer:
    return Tokenizer.from_config(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))


def _source():  # type: ignore[no-untyped-def]
    from symbolic_data import ProblemSource
    from symbolic_data.generative import LampleChartonCatalog

    catalog = LampleChartonCatalog.from_config({
        "type": "lample_charton",
        "simplipy_engine": "base",
        "holdout_pools": [],
        "sample_strategy": {"n_operator_distribution": "length_proportional", "min_operators": 1,
                            "max_operators": 6, "power": 1, "max_length": 21, "max_tries": 1,
                            "independent_dimensions": True},
        "allow_nan": False,
        "simplify": True,
        "literal_prior": {"name": "normal", "kwargs": {"loc": 0, "scale": 5}},
        "support_sampler": {
            "support_prior": {"name": "uniform", "kwargs": {"low": 0.5, "high": 4, "min_value": 0.5, "max_value": 4}},
            "n_support_prior": {"name": "uniform", "kwargs": {"low": 8, "high": 16, "min_value": 8, "max_value": 16}},
        },
        "variables": ["x1", "x2", "x3"],
        "operator_weights": {"+": 10, "-": 10, "*": 10, "sin": 2},
    })
    catalog.skeletons = {
        ("*", "<constant>", "x1"),
        ("*", "<constant>", "*", "x1", "x2"),
        ("sin", "*", "<constant>", "x3"),
        ("+", "*", "<constant>", "x1", "x2"),
    }
    catalog.skeleton_codes = catalog.compile_codes()
    return ProblemSource({"catalog": catalog, "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0}})


def test_units_block_requires_the_tagged_dialect(v24_tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="tagged"):
        FlashANSRDataset(source=_source(), tokenizer=v24_tokenizer, padding="zero", units_block=dict(CFG))


def test_stream_scales_data_and_literals_consistently(v24_tokenizer: Tokenizer) -> None:
    n_applied = 0
    with FlashANSRDataset(source=_source(), tokenizer=v24_tokenizer, padding="zero", target_dialect="tagged",
                          units_block=dict(CFG)) as dataset:
        engine = dataset.source.catalog.simplipy_engine
        variables = list(dataset.source.catalog.variables)
        for batch in dataset.iterate(steps=3, batch_size=16):
            for i, units in enumerate(batch["units"]):
                n = int(batch["data_attn_mask"][i].sum())
                x = batch["x_tensors"][i, :n].numpy().astype(np.float64)
                y = batch["y_tensors"][i, :n].numpy().astype(np.float64)
                # The stored concrete expression describes the stored (scaled) data ...
                f = build_expression_callable(engine, list(batch["expression"][i]), variables)
                y_hat = np.asarray(f(x.T.copy()), dtype=np.float64).reshape(-1, 1)
                np.testing.assert_allclose(y_hat, y, rtol=1e-9)
                # ... and so does the training target: the skeleton with the (re-valued) literal values.
                from simplipy.utils import substitute_constants
                tagged = substitute_constants(list(batch["skeleton"][i]),
                                              values=[float(c) for c in batch["constants"][i]], inplace=False)
                g = build_expression_callable(engine, list(engine.to_prefix(tagged)), variables)
                y_target = np.asarray(g(x.T.copy()), dtype=np.float64).reshape(-1, 1)
                np.testing.assert_allclose(y_target, y, rtol=1e-9)
                if units is not None:
                    n_applied += 1
                    assert set(units) == {"z", "zy"}
        counters = dict(dataset.stream_counters)
    assert n_applied > 0
    assert set(UNITS_COUNTER_KEYS) <= set(counters)
    assert counters["n_units_applied"] + counters["n_units_applied_y_only"] == n_applied
    statuses = ("applied", "applied_y_only", "identity_draw", "no_symmetry", "fallback_box", "fallback_holdout",
                "fallback_unparsed")
    assert sum(counters[f"n_units_{status}"] for status in statuses) == 48
    assert math.isfinite(float(counters["n_units_redraws"]))
