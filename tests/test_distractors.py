"""Distractors: unused input columns that keep their drawn values (owner rulings 2026-11-13)."""
from collections import Counter

import numpy as np
import pytest

from flash_ansr.data.distractors import draw_distractors, validate_distractor_block
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns

VARIABLES = [f"x{i}" for i in range(1, 18)]
RULED = {"p_instance": 0.2, "p_continue": 0.5}


def _x(rng, n=64):
    return rng.uniform(-10, 10, size=(n, len(VARIABLES)))


def test_the_block_is_pinned_explicitly():
    assert validate_distractor_block(None) is None
    assert validate_distractor_block(dict(RULED)) == RULED
    with pytest.raises(ValueError):
        validate_distractor_block({"p_instance": 0.2})
    with pytest.raises(ValueError):
        validate_distractor_block({"p_instance": 0.2, "p_continue": 0.5, "cap": 3})
    with pytest.raises(ValueError):
        validate_distractor_block({"p_instance": 1.5, "p_continue": 0.5})


def test_the_count_is_a_geometric_chain_over_the_zeroed_columns():
    rng = np.random.default_rng(0)
    used = {"x2", "x5", "x11"}
    counts = Counter()
    n = 40_000
    for _ in range(n):
        draw = draw_distractors(x_support=_x(rng, 4), variables=VARIABLES, used=used, rng=rng, cfg=RULED)
        assert not set(draw.columns) & used and len(set(draw.columns)) == len(draw.columns)
        counts[len(draw.columns)] += 1
    assert abs(1 - counts[0] / n - 0.2) < 0.01
    with_any = n - counts[0]
    for k, expected in ((1, 0.5), (2, 0.25), (3, 0.125)):
        assert abs(counts[k] / with_any - expected) < 0.02, (k, counts[k] / with_any)


def test_the_only_limit_is_the_number_of_zeroed_columns():
    rng = np.random.default_rng(1)
    used = set(VARIABLES[:15])
    draw = draw_distractors(x_support=_x(rng), variables=VARIABLES, used=used, rng=rng,
                            cfg={"p_instance": 1.0, "p_continue": 1.0})
    assert draw.columns == ["x16", "x17"]
    assert draw_distractors(x_support=_x(rng), variables=VARIABLES, used=set(VARIABLES), rng=rng,
                            cfg={"p_instance": 1.0, "p_continue": 1.0}).columns == []


def test_a_distractor_borrows_a_used_variables_change_of_units():
    rng = np.random.default_rng(2)
    used = {"x1", "x2", "x3"}
    unit_z = {"x1": 40, "x2": -7}           # x3 is a variable of the expression that the law left unscaled (0)
    seen = Counter()
    for _ in range(300):
        x = _x(rng)
        before = x.copy()
        draw = draw_distractors(x_support=x, variables=VARIABLES, used=used, rng=rng,
                                cfg={"p_instance": 1.0, "p_continue": 0.5}, unit_z=unit_z)
        for name in draw.columns:
            j = VARIABLES.index(name)
            z = draw.scales.get(name, 0)
            seen[z] += 1
            assert np.array_equal(x[:, j], np.ldexp(before[:, j], z))
        untouched = [j for j, v in enumerate(VARIABLES) if v not in draw.columns]
        assert np.array_equal(x[:, untouched], before[:, untouched])
    assert set(seen) == {40, -7, 0} and min(seen.values()) > 50


def test_identity_units_leave_the_values_as_drawn():
    rng = np.random.default_rng(3)
    x = _x(rng)
    before = x.copy()
    draw = draw_distractors(x_support=x, variables=VARIABLES, used={"x1"}, rng=rng,
                            cfg={"p_instance": 1.0, "p_continue": 0.9}, unit_z=None)
    assert draw.columns and draw.scales == {} and np.array_equal(x, before)


def test_a_scale_that_leaves_float64_is_refused_and_counted():
    rng = np.random.default_rng(4)
    x = np.full((8, len(VARIABLES)), 1e300)
    before = x.copy()
    draw = draw_distractors(x_support=x, variables=VARIABLES, used={"x1"}, rng=rng,
                            cfg={"p_instance": 1.0, "p_continue": 1.0}, unit_z={"x1": 100})
    assert len(draw.columns) == 16 and draw.scales == {} and draw.scale_fallbacks == 16
    assert np.array_equal(x, before)


def test_masking_keeps_the_distractors_and_zeroes_the_rest():
    x = np.arange(4 * 5, dtype=float).reshape(4, 5) + 1
    variables = ["x1", "x2", "x3", "x4", "x5"]
    mask_unused_variable_columns((x,), variables=variables, skeleton_tokens=["+", "x1", "x3"], padding="zero",
                                 keep=["x4"])
    assert np.all(x[:, [0, 2, 3]] != 0) and np.all(x[:, [1, 4]] == 0)
