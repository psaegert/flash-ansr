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
import threading
from typing import TYPE_CHECKING, Iterable, cast

from simplipy import masking

if TYPE_CHECKING:
    from simplipy import SimpliPyEngine


# The reserved spellings a simplification can fold a degenerate sub-expression to. Matches the
# forbidden set symbolic-data's generator rejects a sampled skeleton on
# (``LampleChartonCatalog._sympy_simplify_skeleton``); its SimpliPy branch checks the first three,
# the only spellings SimpliPy itself emits. The SymPy spellings are kept because this seam also
# canonicalizes expressions PARSED from external files (``flash_ansr.convert_data``), which no
# producer contract covers, and exact-token matching means they cannot false-positive: no
# vocabulary token, variable or literal spelling equals ``zoo`` / ``nan`` / ``oo``.
NON_FINITE_TOKENS = frozenset({'float("inf")', 'float("-inf")', 'float("nan")', 'zoo', 'nan', 'oo'})


class NonFiniteExpressionError(ValueError):
    """An expression contains a non-finite token (``float("inf")``, ``float("nan")``, ...).

    Raised by the seam, not by the policy: whether that is a rejectable candidate (drop it) or a
    broken producer contract (let it propagate) is the CALLER's decision, and the two directions
    of this package answer it differently. Raising rather than returning ``None`` keeps
    ``simplify_and_mask``'s return type a skeleton: an optional return would spread a silent-null
    contract to every call site, and a site that forgot the check would carry the ``None`` onward
    instead of failing.
    """

    def __init__(self, expression: Iterable[str]) -> None:
        self.expression = list(expression)
        self.tokens = find_non_finite(self.expression)
        super().__init__(
            f"Expression contains non-finite token(s) {self.tokens}: {self.expression}. "
            "A simplification folded a degenerate sub-expression (division by zero, log of zero, "
            "0/0) to a reserved non-finite spelling; these are encodable vocabulary tokens, so an "
            "unguarded result would re-enter the candidate stream as a valid expression."
        )


_drops_lock = threading.Lock()
_drops = 0


def find_non_finite(expression: Iterable[str]) -> list[str]:
    """The non-finite tokens present in ``expression``, in first-seen order (empty if finite).

    Exact token matching (never substring), matching symbolic-data's membership test.
    """
    seen: list[str] = []
    for token in expression:
        if token in NON_FINITE_TOKENS and token not in seen:
            seen.append(token)
    return seen


def record_non_finite_drop() -> None:
    """Count one candidate dropped for being non-finite.

    Dropping a candidate is silent by nature, so every drop site calls this: the count is the
    observable that says how much of a generation pool the guard removed. Process-local by
    design -- the forked simplify workers report a non-finite by omitting the entry so the
    parent's serial fallback does the dropping, and therefore the counting, here.
    """
    global _drops
    with _drops_lock:
        _drops += 1


def non_finite_drops() -> int:
    """Number of candidates dropped for being non-finite since the last reset."""
    with _drops_lock:
        return _drops


def reset_non_finite_drops() -> None:
    """Reset the non-finite drop count (per-run accounting)."""
    global _drops
    with _drops_lock:
        _drops = 0


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

    Raises :class:`NonFiniteExpressionError` when the result contains a non-finite token. Simplify
    is where those are MINTED -- a finite, grammar-valid candidate like ``['/', 'x1', '-', 'x2',
    'x2']`` folds to ``['*', 'float("inf")', 'x1']`` -- and this is the one seam every candidate
    producer shares, so the check belongs here rather than at any single producer's entry point.
    The check is on the RETURNED skeleton, which also covers a non-finite that was already in the
    input (masking leaves those spellings alone: they are not literal sites).
    """
    # simplify mirrors its input container type; a list input returns a list
    simplified = cast(list[str], engine.simplify(list(expression), form='explicit'))
    masked = mask_all_literals(engine, simplified)
    if find_non_finite(masked):
        raise NonFiniteExpressionError(masked)
    return masked


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
    -- the alignment the numeric head trains on. Pre-existing ``<constant>`` placeholders are
    left alone (they carry no value).

    Reserved non-finite spellings are not literal sites, so masking would leave them as written
    and the tokenizer would happily encode them into a training target. This raises
    :class:`NonFiniteExpressionError` instead, and does NOT drop the sample: on the ingest
    direction flash-ansr is the CONSUMER of a producer contract (symbolic-data rejects this exact
    token set before yielding), so one arriving here means that contract broke. Skipping the
    sample would hide the broken producer and silently reshape the training distribution; the
    candidate-stream direction drops because there flash-ansr mints the non-finite itself.
    """
    tokens = list(expression)
    sites = masking.literal_sites(tokens, engine)
    skeleton = masking.mask(tokens, engine, masking.mask_all, collect=False)
    if find_non_finite(skeleton):
        raise NonFiniteExpressionError(skeleton)
    return skeleton, [_literal_value(value) for _, value, _ in sites]
