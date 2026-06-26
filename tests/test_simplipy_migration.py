"""Behavioural-equivalence guard for the simplipy primitive migration.

Step-1 prep swapped flash-ansr's product code to import three expression/token primitives from
simplipy instead of from its own (forked) copies:

* ``codify``            (was ``flash_ansr.expressions.compilation.codify``)        -> ``simplipy.utils.codify``
* ``substitute_constants`` (was ``...token_ops.substitute_constants``)             -> ``simplipy.utils.substitude_constants``  [sic, typo upstream]
* ``identify_constants``   (was ``...token_ops.identify_constants``)               -> ``simplipy.utils.explicit_constant_placeholders``

The flash-ansr copies are intentionally left in place (research notebooks still import them; they
travel to the research fork at carve time). This module pins TWO things so the swap cannot silently
change behaviour or break on a simplipy upgrade:

1. The two name gotchas (§2): simplipy spells it ``substitude_constants`` (typo), and uses
   ``explicit_constant_placeholders`` / ``numbers_to_constant`` rather than ``identify_constants`` /
   ``numbers_to_num`` -- so the swaps must alias.
2. Behavioural equivalence between the flash fork and the simplipy function over the call shapes the
   product code actually uses (``constants`` defaulting to ``None``; the only shape any swapped call
   site passes -- verified in the discovery pass). The ``C_i``-naming divergence that appears only
   when a NON-empty ``constants`` is supplied is therefore out of scope and not exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

import simplipy.utils as sp
from flash_ansr.expressions.compilation import codify as flash_codify
from flash_ansr.expressions.token_ops import identify_constants as flash_identify_constants
from flash_ansr.expressions.token_ops import substitute_constants as flash_substitute_constants


class TestSimplipyNameGotchas:
    """The exact upstream spellings the import aliases depend on (§2)."""

    def test_substitude_constants_typo_is_the_only_spelling(self):
        assert hasattr(sp, "substitude_constants")  # the swap aliases this [sic]
        assert not hasattr(sp, "substitute_constants")  # correct spelling does NOT exist upstream

    def test_identify_constants_maps_to_explicit_constant_placeholders(self):
        assert hasattr(sp, "explicit_constant_placeholders")
        assert not hasattr(sp, "identify_constants")

    def test_numbers_to_num_maps_to_numbers_to_constant(self):
        assert hasattr(sp, "numbers_to_constant")
        assert not hasattr(sp, "numbers_to_num")


def _eval_codify(codify_fn, body, variables, args):
    """Compile ``body`` via ``codify_fn`` and evaluate the resulting lambda on ``args``."""
    fn = eval(codify_fn(body, variables), {"np": np, "numpy": np})  # noqa: S307 - trusted test input
    return fn(*args)


class TestCodifyEquivalence:
    @pytest.mark.parametrize(
        "body, variables, args",
        [
            ("x0 + x1", ["x0", "x1"], (1.0, 2.0)),
            ("np.sin(x0) * x1", ["x0", "x1"], (0.5, 3.0)),
            ("x0 ** 2 - 4.0", ["x0"], (2.5,)),
        ],
    )
    def test_flash_and_simplipy_agree(self, body, variables, args):
        assert _eval_codify(flash_codify, body, variables, args) == pytest.approx(
            _eval_codify(sp.codify, body, variables, args)
        )


class TestSubstituteConstantsEquivalence:
    @pytest.mark.parametrize(
        "prefix, values",
        [
            (["+", "<constant>", "x0"], [3.0]),
            (["*", "C_0", "sin", "C_1"], [2.0, 0.5]),
            (["+", "x0", "x1"], []),  # no placeholders
        ],
    )
    def test_flash_and_simplipy_agree(self, prefix, values):
        # The only call shape the product uses: values + inplace=False, no constants= arg.
        assert flash_substitute_constants(list(prefix), list(values)) == sp.substitude_constants(
            list(prefix), list(values)
        )


class TestIdentifyConstantsEquivalence:
    @pytest.mark.parametrize(
        "prefix",
        [
            ["+", "<constant>", "x0"],
            ["*", "<constant>", "sin", "<constant>"],
            ["+", "C_0", "x1"],  # already-named constant
            ["+", "3", "x0"],  # numeric literal (isnumeric)
            ["+", "x0", "x1"],  # nothing to rename
            ["*", "<constant>", "+", "<constant>", "<constant>"],  # multi-constant
        ],
    )
    def test_flash_and_simplipy_agree_at_constants_none(self, prefix):
        # constants defaults to None -- the only shape any swapped call site passes.
        flash_expr, flash_consts = flash_identify_constants(list(prefix))
        sp_expr, sp_consts = sp.explicit_constant_placeholders(list(prefix))
        assert flash_expr == sp_expr
        assert flash_consts == sp_consts
