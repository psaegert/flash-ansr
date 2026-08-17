"""Skeleton canonicalization: simplify + literal masking in the model's token dialect.

simplipy 0.12.0 deleted ``SimpliPyEngine.mask`` -- masking is downstream policy now and
lives in :mod:`simplipy.masking` as ``mask(tokens, engine, policy)``. These helpers are the
single home for flash-ansr's policy choice so every former ``engine.mask(engine.simplify(x))``
site keeps ONE shared behavior:

* ``policy = mask_all`` (every numeric literal, including ``np.pi``/``np.e``, becomes
  ``<constant>``): the legacy ``Engine.mask`` semantics these sites were written against,
  and the only choice the deployed tokenizer vocabulary supports -- it carries no numeric
  tokens beyond ``0``/``1``/``(-1)``/``np.pi``/``np.e``, so any literal left unmasked
  (e.g. a kept ``pow`` exponent under ``mask_fittable``) would fail to re-encode and
  silently drop the candidate.
* ``form='explicit'`` on simplify: simplipy 0.13's default token output is the tagged
  n-ary dialect (``<add> ... </add>``), which the tokenizer cannot encode. The explicit
  binary-chain form is the classic prefix dialect the models were trained on.
"""
from typing import TYPE_CHECKING, cast

from simplipy import masking

if TYPE_CHECKING:
    from simplipy import SimpliPyEngine


def mask_all_literals(engine: "SimpliPyEngine", expression: list[str]) -> list[str]:
    """Mask every numeric literal in ``expression`` to ``<constant>``.

    The ``simplipy.masking`` collect pass also enforces one ``<constant>`` per degree of
    freedom (``c1*c2`` collapses to one constant), which the deleted ``Engine.mask``
    approximated with its single sort pass.
    """
    return masking.mask(list(expression), engine, masking.mask_all)


def simplify_and_mask(engine: "SimpliPyEngine", expression: list[str]) -> list[str]:
    """Simplify ``expression`` and mask all literals: the skeleton the model vocabulary encodes.

    Drop-in successor of the pre-simplipy-0.12 ``engine.mask(engine.simplify(x))`` idiom.
    """
    # simplify mirrors its input container type; a list input returns a list
    simplified = cast(list[str], engine.simplify(list(expression), form='explicit'))
    return mask_all_literals(engine, simplified)
