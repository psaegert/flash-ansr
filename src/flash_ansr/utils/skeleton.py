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
import math
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
    # simplipy >= 0.14 simplify is dialect-preserving: an explicit binary-prefix
    # list in means an explicit binary-prefix list out (the form= escape is removed)
    simplified = cast(list[str], engine.simplify(list(expression)))
    return mask_all_literals(engine, simplified)


def _literal_value(token: str) -> float:
    """Numeric value of a literal token as classified by ``simplipy.masking.literal_sites``:
    a plain int/float spelling, a one-token exact rational (``1/3``), or ``np.pi``/``np.e``."""
    if token == 'np.pi':
        return math.pi
    if token == 'np.e':
        return math.e
    try:
        return float(token)
    except ValueError:
        numerator, _, denominator = token.partition('/')
        return float(numerator) / float(denominator)


def mask_literals_positional(engine: "SimpliPyEngine", expression: list[str]) -> tuple[list[str], list[float]]:
    """Mask every literal in a CONCRETE ``expression`` positionally and return the values.

    The training-data contract: symbolic-data >= 0.14 generative catalogs yield concrete
    expressions (literal values in the tokens, ``Problem.constants`` empty) and masking is
    downstream policy. ``collect=False`` keeps the strict 1:1 site<->token correspondence,
    so the i-th returned value belongs to the i-th ``<constant>`` in the returned skeleton
    -- the alignment the numeric head trains on. Reserved non-finite spellings
    (``float("inf")``/``float("nan")``) are not literal sites and stay as written, and
    pre-existing ``<constant>`` placeholders are likewise left alone (they carry no value).
    """
    tokens = list(expression)
    sites = masking.literal_sites(tokens, engine)
    skeleton = masking.mask(tokens, engine, masking.mask_all, collect=False)
    return skeleton, [_literal_value(value) for _, value, _ in sites]
