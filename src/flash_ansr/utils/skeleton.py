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
from simplipy.compat import RETIRED_OPERATOR_TOKENS

if TYPE_CHECKING:
    from simplipy import SimpliPyEngine


def has_retired_operators(expression: list[str]) -> bool:
    """True when ``expression`` contains a generation-1 hyper-operator token.

    The generation-2 vocabulary retires ``pow2``/``mult2``/``div2``/... . A retired token
    in OPERATOR position fails ``is_valid`` (arity mismatch), but in LEAF position the
    engine reads it as a VARIABLE, so a beam like ``['*', 'pow3', 'x1']`` validates and
    then NameErrors inside the fitted lambda. Beams from generation-1-era model
    vocabularies must therefore be rejected explicitly.
    """
    return any(token in RETIRED_OPERATOR_TOKENS for token in expression)


#: Exact generation-2 spellings of the generation-1 unary sugar operators:
#: ``token: (tokens_before_operand, tokens_after_operand)``. Every entry is definitional
#: (``pow2 x`` IS ``pow x 2``); the odd real roots map to ``rootn`` (defined on x < 0),
#: matching their generation-1 real-root semantics.
_GEN1_SUGAR: dict[str, tuple[list[str], list[str]]] = {
    'pow2': (['pow'], ['2']),
    'pow3': (['pow'], ['3']),
    'pow4': (['pow'], ['4']),
    'pow5': (['pow'], ['5']),
    'pow1_2': (['pow'], ['0.5']),
    'pow1_4': (['pow'], ['0.25']),
    'pow1_3': (['rootn'], ['3']),
    'pow1_5': (['rootn'], ['5']),
    'pow1': ([], []),
    'pow_1': (['inv'], []),
    'mult2': (['*', '2'], []),
    'mult3': (['*', '3'], []),
    'mult4': (['*', '4'], []),
    'mult5': (['*', '5'], []),
    'div2': (['/'], ['2']),
    'div3': (['/'], ['3']),
    'div4': (['/'], ['4']),
    'div5': (['/'], ['5']),
}


def desugar_gen1_operators(engine: "SimpliPyEngine", expression: list[str]) -> list[str]:
    """Rewrite generation-1 unary sugar operators into their generation-2 spellings.

    The published flash-ansr models (v23.x) were trained on the generation-1 vocabulary
    and emit ``pow2``/``mult2``/... tokens, which generation-2 engines retire outright.
    Each such operator is definitional sugar (``pow2 x == pow x 2``), so translating the
    decoded beam is exact -- this is what lets a generation-1-era model run on a released
    (>= 0.12) simplipy at all. Non-well-formed sequences (model garbage that a validity
    gate rejects later anyway) are returned unchanged.
    """
    if not any(token in _GEN1_SUGAR for token in expression):
        return list(expression)

    arity = getattr(engine, 'operator_arity_compat', engine.operator_arity)

    def rewrite(index: int) -> tuple[list[str], int]:
        token = expression[index]
        if token in _GEN1_SUGAR:
            before, after = _GEN1_SUGAR[token]
            inner, next_index = rewrite(index + 1)
            return before + inner + after, next_index
        out = [token]
        next_index = index + 1
        for _ in range(arity.get(token, 0)):
            operand, next_index = rewrite(next_index)
            out.extend(operand)
        return out, next_index

    try:
        result, end = rewrite(0)
    except (IndexError, RecursionError):
        return list(expression)
    if end != len(expression):
        return list(expression)
    return result


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
