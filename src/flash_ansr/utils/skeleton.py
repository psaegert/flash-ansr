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


_unencodable_lock = threading.Lock()
_unencodable = 0


def record_unencodable_drop() -> None:
    """Count one candidate dropped because its SIMPLIFIED spelling could not be re-encoded.

    simplipy's collect pass re-runs simplify after positional masking, so it can mint a bare
    numeral (`pow x1 2`) that a v24 vocabulary -- which carries no numeral tokens -- cannot encode.
    The candidate and its expression are both valid; only the spelling is unencodable. Counted for
    the same reason non-finite drops are: a silent drop is a loss nobody can size.
    """
    global _unencodable
    with _unencodable_lock:
        _unencodable += 1


def unencodable_drops() -> int:
    """Number of candidates dropped as unencodable since the last reset."""
    with _unencodable_lock:
        return _unencodable


def reset_unencodable_drops() -> None:
    """Reset the unencodable drop count (per-run accounting)."""
    global _unencodable
    with _unencodable_lock:
        _unencodable = 0


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
    # simplipy >= 0.14 simplify is dialect-preserving: an explicit binary-prefix
    # list in means an explicit binary-prefix list out (the form= escape is removed)
    simplified = cast(list[str], engine.simplify(list(expression)))
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


_SPECIAL_CONSTANT_TOKENS = ("np.pi", "np.e")


def _mask_numeric_keep_specials(value: str, role: "masking.Role") -> str | None:
    """The v24 target policy: every NUMERIC literal is extracted for ieee754
    serialization; the symbolic constants stay as written (contract A3: the AC engine
    keeps ``np.pi``/``np.e`` symbolic -- they have no decimal spelling to serialize and
    remain vocabulary tokens in the tagged canonical dialect)."""
    if value in _SPECIAL_CONSTANT_TOKENS:
        return None
    return "<constant>"


def mask_promptable(engine: "SimpliPyEngine", expression: "list[str]",
                    mode: str) -> "list[str]":
    """The promptable-mask target (owner ruling 2026-08-24): simplipy's policy
    semantics under the v24 dialect's specials doctrine.

    ``mode`` is ``'all'`` (every numeric literal masked, simplipy ``mask_all``) or
    ``'fittable'`` (fittable values masked, structural literals kept, simplipy
    ``mask_fittable``). ``np.pi``/``np.e`` stay as written in EITHER mode -- they are
    symbolic in the tagged canonical dialect (contract A3), exactly as the unmasked
    target path treats them. Runs with ``collect=True``: the degree-of-freedom
    collection is a ``simplify`` call and may restructure (``2*x0/3`` collects to
    ``<constant>*x0``) -- ruled fine, it is the canonical skeleton.
    """
    if mode not in ("all", "fittable"):
        raise ValueError(f"mask mode must be 'all' or 'fittable', got {mode!r}")
    base = masking.mask_all if mode == "all" else masking.mask_fittable

    def policy(value: str, role: "masking.Role") -> "str | None":
        if value in _SPECIAL_CONSTANT_TOKENS:
            return None
        return base(value, role)

    return masking.mask(list(expression), engine, policy, collect=True)


def fittable_slots(engine: "SimpliPyEngine", expression: "list[str]") -> "list[bool]":
    """Per non-special literal site of ``expression``, in positional order (the SAME
    slots ``mask_literals_positional`` returns values for): True where simplipy's
    ``mask_fittable`` policy masks the literal (a fittable value), False where it
    keeps it (a structural literal -- pow exponent, rootn index).

    The v24 promptable-mask worker decides placeholding PER SLOT with this, rather
    than re-running ``engine.mask`` with collection: the slots are exactly the ones
    the byte serialization fills, and per-slot decisions keep the placeheld values
    recoverable for the ``<predict_constants>`` block. Note the slot granularity is
    the LITERAL SITE, not the degree of freedom: the tagged canonical spells an exact
    rational structurally (``2.5`` is ``5 <div> 2``, two sites), so such a
    coefficient masks into two placeholders -- the serialization's own premise,
    inherited, not introduced here.
    """
    sites = masking.literal_sites(list(expression), engine)
    return [masking.mask_fittable(value, role) is not None
            for _, value, role in sites if value not in _SPECIAL_CONSTANT_TOKENS]


def nonspecial_site_positions(engine: "SimpliPyEngine", expression: "list[str]") -> "list[int]":
    """Token positions of the non-special literal sites, in positional order --
    aligned 1:1 with ``mask_literals_positional``'s returned values."""
    sites = masking.literal_sites(list(expression), engine)
    return [pos for pos, value, _ in sites if value not in _SPECIAL_CONSTANT_TOKENS]


def mask_selected_sites(engine: "SimpliPyEngine", expression: "list[str]",
                        placeheld: "list[bool]", *, collect: bool) -> "list[str]":
    """simplipy's ``masking.mask`` over exactly the selected non-special sites.

    ``placeheld`` is aligned with the non-special literal sites in positional
    order. With ``collect=True`` this is the engine's COLLECTED mask of the
    selected sites -- the reference the v24 stability check compares against
    plain substitution (owner ruling 2026-08-24: instances where collection
    restructures carry no ``<predict_constants>`` block, because the merged /
    sign-absorbed placeholder values are engine-internal until simplipy's
    value-carrying mask exists).
    """
    counter = {"i": 0}

    def policy(value: str, role: "masking.Role") -> "str | None":
        if value in _SPECIAL_CONSTANT_TOKENS:
            return None
        if counter["i"] >= len(placeheld):
            raise ValueError(f"site count mismatch: more sites than the {len(placeheld)} entries")
        selected = placeheld[counter["i"]]
        counter["i"] += 1
        return "<constant>" if selected else None

    out = masking.mask(list(expression), engine, policy, collect=collect)
    if counter["i"] != len(placeheld):
        raise ValueError(f"site count mismatch: policy saw {counter['i']}, expected {len(placeheld)}")
    return out


def mask_literals_positional(engine: "SimpliPyEngine", expression: list[str],
                             keep_specials: bool = False) -> tuple[list[str], list[float]]:
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

    ``keep_specials=True`` is the v24 target-path policy (contract A3): ``np.pi`` and
    ``np.e`` stay as written -- they are symbolic in the tagged canonical dialect and
    never serialize to an ieee754 span -- and are correspondingly EXCLUDED from the
    returned values, keeping the 1:1 site<->``<constant>`` alignment.
    """
    tokens = list(expression)
    sites = masking.literal_sites(tokens, engine)
    policy = _mask_numeric_keep_specials if keep_specials else masking.mask_all
    skeleton = masking.mask(tokens, engine, policy, collect=False)
    if find_non_finite(skeleton):
        raise NonFiniteExpressionError(skeleton)
    if keep_specials:
        sites = [site for site in sites if site[1] not in _SPECIAL_CONSTANT_TOKENS]
    return skeleton, [_literal_value(value) for _, value, _ in sites]
