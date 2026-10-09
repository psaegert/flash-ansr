"""Units augmentation of training instances: a change of units along the law's own dimensional symmetry.

Owner rulings 2026-10-08/09 (units augmentation, "route B"): the model must work at every scale without an
inference-time rescaling, so the TRAINING DATA carry changes of units. A physical law does not depend on the units
its quantities are measured in; changing units multiplies each input column and the output by a constant, and a
dimensionally consistent law keeps its form with its constants absorbing the factors (``sin(omega * t)``,
``exp(-E / (k * T))``). The generator's laws are mostly NOT dimensionally homogeneous (bare variables inside
``sin``, sums mixing degrees), so an arbitrary change of units is not a symmetry of the law and re-writing the
target for it adds constants everywhere. This module therefore only applies changes of units the law itself
admits:

* **Dimensions are inferred, never sampled.** Every used variable carries a log2 scale ``z_v``, the output
  ``z_y``, every fittable literal a free log2 shift ``w``. Summands share a dimension, arguments of transcendental
  functions are dimensionless, ``*`` / ``/`` add and subtract dimensions, ``pow`` / ``rootn`` with a numeric
  exponent multiply it, ``neg`` / ``abs`` are transparent. Pure numbers: ``np.pi`` / ``np.e``, structural
  (non-fittable) literals, and every INTEGER-valued literal below 2**53 (signs, counts from ``x + x -> 2 x``,
  rational-exponent spellings): the generator spells a sampled constant as an integer only when it is exactly
  integral, so integers are structure, not measurements -- except beyond float64's exact-integer limit, where every
  value is integral.
* **Scales are powers of two** on the integer solution lattice of that linear system, ``z ~ U{-z_max..z_max}`` on
  its free coordinates, applied with ``np.ldexp``: data, literals and residuals stay exact.
* **The target is re-valued, never re-written:** the literals of the already canonical target are multiplied by
  ``2**w`` in place; skeleton, length, complexity and fittable slots are unchanged by construction. The output
  follows the law (its scale is whatever the inputs imply, or free when an outer literal absorbs it).
* **Guards:** every re-valued literal stays inside ``2**+-log2_literal_max``, every data value finite and normal,
  and the re-valued target, evaluated in float64 on the scaled inputs, reproduces the scaled (clean) output to srbf's
  numeric-recovery precision (FVU <= float32 eps). The scaling is exact, but an intermediate product of large factors
  can overflow, or a transcendental of a large argument amplify the last bit, where the original units did not; such
  a target would not describe its data. A violating draw is redrawn up to ``max_redraws`` times, then the instance
  keeps identity units (counted, never dropped, so homogeneous laws are not selected against). A re-valued target is
  re-checked against the holdout pools; a hit keeps identity units.

A fraction ``p_identity`` of instances keeps identity units by construction: the generator is the "natural units"
hypothesis, the change of units the "arbitrary units" one, and the pilot measures both shares.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Sequence

import numpy as np
from simplipy import masking
from simplipy.utils import codify, safe_f

#: float64 represents every integer below this exactly; at and above it every value is integral.
EXACT_INTEGER_LIMIT = 2 ** 53
#: Literal tokens that are pure numbers by name (kept symbolic by the tagged canonical).
PURE_SPECIALS = frozenset({"np.pi", "np.e"})
#: Operators whose argument must be dimensionless (``log`` included: conservative).
TRANSCENDENTAL = frozenset({"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh",
                            "atanh", "exp", "log"})
#: Operators that pass their argument's dimension through unchanged.
TRANSPARENT = frozenset({"neg", "abs"})
#: Tagged-canonical bags: opener -> (closer, inverse-section marker).
BAGS = {"<add>": ("</add>", "<sub>"), "<mul>": ("</mul>", "<div>")}

UNITS_BLOCK_KEYS = ("p_identity", "log2_scale_max", "log2_literal_max", "max_redraws")

#: Outcomes of :func:`apply_units`, one counter each in the stream health counters.
STATUS_APPLIED = "applied"            # x (and y as the law implies) moved
STATUS_APPLIED_Y = "applied_y_only"   # only y moved: no input scaling is a symmetry, an outer literal absorbs y
STATUS_IDENTITY = "identity_draw"     # the p_identity share
STATUS_FIXED = "no_symmetry"          # the law admits no change of units (correct: it is never shifted)
STATUS_BOX = "fallback_box"           # no draw within max_redraws kept literals, data and the target's fit in range
STATUS_HOLDOUT = "fallback_holdout"   # the re-valued target hit a holdout family
STATUS_UNPARSED = "fallback_unparsed"  # an operator outside the dimension rules: identity units
UNITS_STATUSES = (STATUS_APPLIED, STATUS_APPLIED_Y, STATUS_IDENTITY, STATUS_FIXED, STATUS_BOX, STATUS_HOLDOUT,
                  STATUS_UNPARSED)
#: The worker's per-batch counter keys (summed by FlashANSRDataset.stream_counters, logged by the trainer).
UNITS_COUNTER_KEYS = tuple(f"n_units_{status}" for status in UNITS_STATUSES) + ("n_units_redraws",)


def validate_units_block(raw: "dict[str, Any] | None") -> "dict[str, Any] | None":
    """The ``units_block`` config: exact keys, pinned explicitly (never defaulted), like the task blocks."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"units_block must be a mapping, got {raw!r}")
    if set(raw) != set(UNITS_BLOCK_KEYS):
        raise ValueError(f"units_block must carry exactly {sorted(UNITS_BLOCK_KEYS)} (got {sorted(raw)}); "
                         f"priors are pinned explicitly, never defaulted")
    p_identity = float(raw["p_identity"])
    if not 0.0 <= p_identity <= 1.0:
        raise ValueError(f"units_block.p_identity must be a probability in [0, 1], got {raw['p_identity']!r}")
    out: dict[str, Any] = {"p_identity": p_identity}
    for key in ("log2_scale_max", "log2_literal_max", "max_redraws"):
        value = int(raw[key])
        if value < 1 or value != raw[key]:
            raise ValueError(f"units_block.{key} must be a positive integer, got {raw[key]!r}")
        out[key] = value
    if out["log2_scale_max"] > 1000 or out["log2_literal_max"] > 1000:
        raise ValueError("units_block: log2 bounds beyond 1000 leave float64's range")
    return out


class UnparsedLaw(ValueError):
    """The law contains an operator the dimension rules do not cover."""


@dataclass
class _Node:
    kind: str                 # "bag" | "op" | "leaf"
    tok: str
    kids: list["_Node"] = field(default_factory=list)
    inv: list[bool] = field(default_factory=list)
    pos: int = -1


def _parse(tokens: Sequence[str], arity: dict[str, int]) -> _Node:
    i = 0

    def rec() -> _Node:
        nonlocal i
        t = tokens[i]
        if t in BAGS:
            close, section = BAGS[t]
            i += 1
            kids: list[_Node] = []
            inv: list[bool] = []
            flip = False
            while tokens[i] != close:
                if tokens[i] == section:
                    flip = True
                    i += 1
                    continue
                kids.append(rec())
                inv.append(flip)
            i += 1
            return _Node("bag", t, kids, inv)
        if arity.get(t, 0) > 0:
            p = i
            i += 1
            return _Node("op", t, [rec() for _ in range(arity[t])], pos=p)
        p = i
        i += 1
        return _Node("leaf", t, pos=p)

    root = rec()
    if i != len(tokens):
        raise UnparsedLaw("trailing tokens after the expression")
    return root


def _fraction(tok: str) -> Fraction | None:
    if tok in PURE_SPECIALS:
        return None
    try:
        value = Fraction(tok)
    except (ValueError, ZeroDivisionError):
        return None
    return value


def _numeric(node: _Node) -> Fraction | None:
    """Exact value of a variable-free literal subtree (an exponent), else None."""
    if node.kind == "leaf":
        return _fraction(node.tok)
    vals = [_numeric(k) for k in node.kids]
    if any(v is None for v in vals):
        return None
    if node.kind == "bag":
        if node.tok == "<add>":
            total = Fraction(0)
            for v, f in zip(vals, node.inv):
                total = total - v if f else total + v  # type: ignore[operator]
            return total
        out = Fraction(1)
        for v, f in zip(vals, node.inv):
            if f:
                if v == 0:
                    return None
                out /= v  # type: ignore[operator]
            else:
                out *= v  # type: ignore[operator]
        return out
    a = vals
    if node.tok == "neg":
        return -a[0]  # type: ignore[operator]
    if node.tok == "inv":
        return None if a[0] == 0 else Fraction(1) / a[0]  # type: ignore[operator]
    if node.tok == "*":
        return a[0] * a[1]  # type: ignore[operator]
    if node.tok == "/":
        return None if a[1] == 0 else a[0] / a[1]  # type: ignore[operator]
    if node.tok == "+":
        return a[0] + a[1]  # type: ignore[operator]
    if node.tok == "-":
        return a[0] - a[1]  # type: ignore[operator]
    return None


def literal_slots(engine: Any, tokens: Sequence[str]) -> tuple[list[int], list[bool]]:
    """Token positions and fittability of the target's literal SLOTS, in the positional order
    ``mask_literals_positional(..., keep_specials=True)`` returns their values in (specials excluded)."""
    positions: list[int] = []
    fittable: list[bool] = []
    for pos, value, role in masking.literal_sites(list(tokens), engine):
        if value in PURE_SPECIALS:
            continue
        positions.append(int(pos))
        fittable.append(masking.mask_fittable(value, role) is not None)
    return positions, fittable


@dataclass(frozen=True)
class LawDimensions:
    """The law's change-of-units lattice: the solved linear system over the columns
    ``("w", slot)`` (absorbing literals), ``("z", variable)`` and ``("y",)``."""
    cols: tuple[tuple[Any, ...], ...]
    reduced: tuple[tuple[Fraction, ...], ...]
    pivots: tuple[int, ...]
    free: tuple[int, ...]
    x_movable: bool
    y_free_alone: bool


def _rref(rows: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """Exact reduced row echelon form; columns in order, so earlier columns become pivots first."""
    M = [list(r) for r in rows]
    pivots: list[int] = []
    r = 0
    n_cols = len(M[0]) if M else 0
    for c in range(n_cols):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        pivots.append(c)
        r += 1
        if r == len(M):
            break
    return M[:r], pivots


def law_dimensions(engine: Any, tokens: Sequence[str], variables: Sequence[str]) -> LawDimensions:
    """Infer the law's dimension classes from its tagged canonical ``tokens`` (literal values spelled inline)."""
    arity = dict(engine.operator_arity_compat)
    root = _parse(list(tokens), arity)
    positions, fittable = literal_slots(engine, tokens)
    slot_of = {pos: k for k, pos in enumerate(positions)}
    absorbing: set[int] = set()
    for k, (pos, fit) in enumerate(zip(positions, fittable)):
        value = _fraction(tokens[pos])
        # A sampled constant carries a free dimension unless it is an integer the generator spelled as one. Beyond
        # float64's exact-integer limit every value is integral, so integrality says nothing about structure there.
        if fit and value is not None and (value.denominator != 1 or abs(value) >= EXACT_INTEGER_LIMIT):
            absorbing.add(k)
    var_set = set(variables)
    rows: list[dict[tuple[Any, ...], Fraction]] = []

    def add(f: dict, g: dict, s: int = 1) -> dict:
        out = dict(f)
        for key, v in g.items():
            out[key] = out.get(key, 0) + s * v
            if out[key] == 0:
                del out[key]
        return out

    def scale(f: dict, c: Fraction) -> dict:
        return {key: v * c for key, v in f.items() if v * c != 0}

    def form(n: _Node) -> dict:
        if n.kind == "leaf":
            if n.tok in var_set:
                return {("z", n.tok): Fraction(1)}
            k = slot_of.get(n.pos)
            if k is not None and k in absorbing:
                return {("w", k): Fraction(1)}
            return {}
        if n.kind == "bag":
            forms = [form(k) for k in n.kids]
            if n.tok == "<mul>":
                out: dict = {}
                for f, inv in zip(forms, n.inv):
                    out = add(out, f, -1 if inv else 1)
                return out
            for f in forms[1:]:
                r = add(f, forms[0], -1)
                if r:
                    rows.append(r)
            return forms[0]
        t, kids = n.tok, n.kids
        if t in ("pow", "**", "rootn"):
            p = _numeric(kids[1])
            if p is None:
                for kid in kids:     # a non-numeric exponent: base and exponent dimensionless
                    r = form(kid)
                    if r:
                        rows.append(r)
                return {}
            if t == "rootn":
                if p == 0:
                    raise UnparsedLaw("rootn with index 0")
                p = 1 / p
            return scale(form(kids[0]), p)
        if t == "inv":
            return scale(form(kids[0]), Fraction(-1))
        if t in TRANSPARENT:
            return form(kids[0])
        if t == "*":
            return add(form(kids[0]), form(kids[1]))
        if t == "/":
            return add(form(kids[0]), form(kids[1]), -1)
        if t in ("+", "-"):
            f0, f1 = form(kids[0]), form(kids[1])
            r = add(f1, f0, -1)
            if r:
                rows.append(r)
            return f0
        if t in TRANSCENDENTAL:
            r = form(kids[0])
            if r:
                rows.append(r)
            return {}
        raise UnparsedLaw(f"operator {t!r} is outside the dimension rules")

    rows.append(add(form(root), {("y",): Fraction(1)}, -1))
    used = [v for v in variables if v in var_set and any(t == v for t in tokens)]
    cols: list[tuple[Any, ...]] = [("w", k) for k in sorted(absorbing)] + [("z", v) for v in used] + [("y",)]
    index = {c: j for j, c in enumerate(cols)}
    dense = []
    for r in rows:
        v = [Fraction(0)] * len(cols)
        for key, val in r.items():
            v[index[key]] = Fraction(val)
        dense.append(v)
    reduced, pivots = _rref(dense) if dense else ([], [])
    free = [c for c in range(len(cols)) if c not in pivots]
    xcols = [j for j, c in enumerate(cols) if c[0] == "z"]
    ycol = len(cols) - 1
    basis = []
    for f in free:
        b = [Fraction(0)] * len(cols)
        b[f] = Fraction(1)
        for i, p in enumerate(pivots):
            b[p] = -reduced[i][f]
        basis.append(b)
    x_movable = any(any(b[j] != 0 for j in xcols) for b in basis)
    # y alone: with every z_v pinned to 0, is z_y still free?
    pinned = dense + [[Fraction(1) if j == xc else Fraction(0) for j in range(len(cols))] for xc in xcols]
    reduced2, pivots2 = _rref(pinned) if pinned else ([], [])
    free2 = [c for c in range(len(cols)) if c not in pivots2]
    y_free_alone = ycol in free2 or (ycol in pivots2 and any(reduced2[pivots2.index(ycol)][f] != 0 for f in free2))
    return LawDimensions(cols=tuple(cols), reduced=tuple(tuple(r) for r in reduced), pivots=tuple(pivots),
                         free=tuple(free), x_movable=x_movable, y_free_alone=bool(y_free_alone))


@dataclass(frozen=True)
class ChangeOfUnits:
    """One admissible change of units: log2 scales of the used variables and of y, and the log2 shift of every
    absorbing literal slot (integers)."""
    z: dict[str, int]
    zy: int
    w: dict[int, int]


def draw_change(dims: LawDimensions, rng: np.random.Generator, *, log2_scale_max: int, need_x: bool,
                literal_log2: dict[int, float], log2_literal_max: int) -> ChangeOfUnits | None:
    """One draw on the lattice: free scale coordinates ~ U{-max..max}, free literal shifts 0, pivots solved.
    None when the draw leaves the integers, the scale bounds or the literal box, or moves nothing it should."""
    cols, reduced, pivots, free = dims.cols, dims.reduced, dims.pivots, dims.free
    val: dict[int, Fraction] = {}
    for f in free:
        kind = cols[f][0]
        val[f] = Fraction(int(rng.integers(-log2_scale_max, log2_scale_max + 1))) if kind in ("z", "y") else Fraction(0)
    for i, p in enumerate(pivots):
        val[p] = -sum((reduced[i][f] * val[f] for f in free), Fraction(0))
    z: dict[str, int] = {}
    zy = 0
    w: dict[int, int] = {}
    for j, c in enumerate(cols):
        v = val[j]
        if v.denominator != 1:
            return None              # off the integer lattice (a rootn share, or a non-integer literal shift)
        iv = int(v)
        if c[0] == "z":
            if abs(iv) > log2_scale_max:
                return None
            z[c[1]] = iv
        elif c[0] == "y":
            if abs(iv) > log2_scale_max:
                return None
            zy = iv
        else:
            if iv != 0 and abs(literal_log2.get(c[1], 0.0) + iv) > log2_literal_max:
                return None
            w[c[1]] = iv
    if need_x and not any(z.values()):
        return None
    if not need_x and zy == 0:
        return None
    return ChangeOfUnits(z=z, zy=zy, w=w)


#: The precision a re-valued target must keep on its data: srbf's numeric-recovery criterion.
REPRODUCE_FVU = float(np.finfo(np.float32).eps)


def reproduces(engine: Any, tokens: Sequence[str], x: np.ndarray, y: np.ndarray, variables: Sequence[str]) -> bool:
    """Whether the (tagged) expression ``tokens``, evaluated in float64 with the engine's own realizations on the rows
    of ``x``, reproduces ``y`` to FVU <= float32 eps (values normalized by max |y| first, so magnitudes near the float64
    limits do not overflow the squares). Non-finite predictions where ``y`` is finite fail."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    try:
        realized = engine.operators_to_realizations(list(engine.to_prefix(list(tokens))))
        function = engine.code_to_lambda(codify(engine.prefix_to_infix(realized, realization=True), list(variables)))
        with np.errstate(all="ignore"):
            pred = np.asarray(safe_f(function, np.asarray(x, dtype=np.float64)), dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - an expression that cannot be evaluated does not describe its data
        return False
    if pred.size == 1 and y.size > 1:
        pred = np.full(y.size, float(pred[0]))
    finite = np.isfinite(y)
    if not finite.any() or not np.all(np.isfinite(pred[finite])):
        return False
    scale = float(np.max(np.abs(y[finite]))) or 1.0
    yt, yp = y[finite] / scale, pred[finite] / scale
    variance = float(np.var(yt))
    error = float(np.mean((yp - yt) ** 2))
    if variance == 0.0:
        return bool(np.max(np.abs(yp - yt)) <= REPRODUCE_FVU)
    return bool(error / variance <= REPRODUCE_FVU)


def _finite_normal(a: np.ndarray) -> bool:
    a = np.asarray(a)
    if not np.all(np.isfinite(a)):
        return False
    nz = np.abs(a[a != 0])
    return bool(nz.size == 0 or nz.min() >= np.finfo(np.float64).tiny)


@dataclass
class UnitsResult:
    status: str
    x_support: np.ndarray
    y_support: np.ndarray
    y_encoder: np.ndarray
    literals: np.ndarray
    change: ChangeOfUnits | None = None
    revalued_tokens: list[str] | None = None
    literals_changed: bool = False
    redraws: int = 0


def apply_units(*, engine: Any, target_tokens: Sequence[str], literals: np.ndarray, x_support: np.ndarray,
                y_support: np.ndarray, y_encoder: np.ndarray, variables: Sequence[str], rng: np.random.Generator,
                cfg: dict[str, Any], is_held_out: Callable[[list[str]], bool] | None) -> UnitsResult:
    """Apply one change of units to a training instance, or keep identity units (and say why).

    ``target_tokens`` is the instance's tagged canonical target with its literal values inline, ``literals`` the
    values ``mask_literals_positional(..., keep_specials=True)`` extracted from it (aligned slot by slot). Inputs
    are never modified in place; the returned arrays are new whenever the units changed."""
    identity = UnitsResult(STATUS_IDENTITY, x_support, y_support, y_encoder, literals)
    if rng.random() < float(cfg["p_identity"]):
        return identity
    try:
        dims = law_dimensions(engine, target_tokens, variables)
    except UnparsedLaw:
        identity.status = STATUS_UNPARSED
        return identity
    if not dims.x_movable and not dims.y_free_alone:
        identity.status = STATUS_FIXED
        return identity
    positions, _ = literal_slots(engine, target_tokens)
    if len(positions) != len(literals):
        raise ValueError(f"literal slots ({len(positions)}) and literal values ({len(literals)}) are misaligned")
    literal_log2 = {k: (math.log2(abs(float(v))) if v != 0 else 0.0) for k, v in enumerate(literals)}
    column = {v: j for j, v in enumerate(variables)}
    # Inputs first; a law whose input scalings all leave the box (or that has none) may still move y alone
    # when an outer literal absorbs it.
    phases = ([True] if dims.x_movable else []) + ([False] if dims.y_free_alone else [])
    redraws = 0
    for need_x in phases:
        for _ in range(int(cfg["max_redraws"])):
            redraws += 1
            change = draw_change(dims, rng, log2_scale_max=int(cfg["log2_scale_max"]), need_x=need_x,
                                 literal_log2=literal_log2, log2_literal_max=int(cfg["log2_literal_max"]))
            if change is None:
                continue
            result = _realize(change, engine=engine, target_tokens=target_tokens, positions=positions,
                              literals=literals, x_support=x_support, y_support=y_support, y_encoder=y_encoder,
                              column=column, variables=variables)
            if result is None:
                continue
            result.redraws = redraws
            # A re-valued literal can move a law into a held-out family (a coefficient that becomes 1, say):
            # re-check, and keep identity units on a hit.
            if result.literals_changed and is_held_out is not None and is_held_out(list(result.revalued_tokens or [])):
                return UnitsResult(STATUS_HOLDOUT, x_support, y_support, y_encoder, literals, redraws=redraws)
            return result
    return UnitsResult(STATUS_BOX, x_support, y_support, y_encoder, literals, redraws=redraws)


def _realize(change: ChangeOfUnits, *, engine: Any, target_tokens: Sequence[str], positions: list[int],
             literals: np.ndarray, x_support: np.ndarray, y_support: np.ndarray, y_encoder: np.ndarray,
             column: dict[str, int], variables: Sequence[str]) -> UnitsResult | None:
    """Scale the data and re-value the literals for one drawn change; None when a value leaves the finite, normal
    float64 range or the re-valued target no longer reproduces the scaled clean output."""
    x_new = np.array(x_support, dtype=np.float64, copy=True)
    for var, zv in change.z.items():
        if zv:
            x_new[:, column[var]] = np.ldexp(x_new[:, column[var]], zv)
    y_new = np.ldexp(np.asarray(y_support, dtype=np.float64), change.zy)
    y_enc_new = np.ldexp(np.asarray(y_encoder, dtype=np.float64), change.zy)
    lit_new = np.array(literals, dtype=np.float64, copy=True)
    for k, wk in change.w.items():
        if wk:
            lit_new[k] = np.ldexp(lit_new[k], wk)
    used_cols = [column[v] for v in change.z]
    if not (_finite_normal(x_new[:, used_cols]) and _finite_normal(y_new) and _finite_normal(y_enc_new)
            and _finite_normal(lit_new)):
        return None
    revalued = list(target_tokens)
    for k, wk in change.w.items():
        if wk:
            revalued[positions[k]] = repr(float(lit_new[k]))
    if not reproduces(engine, revalued, x_new, y_new, variables):
        return None
    status = STATUS_APPLIED if any(change.z.values()) else STATUS_APPLIED_Y
    return UnitsResult(status, x_new, y_new, y_enc_new, lit_new, change=change, revalued_tokens=revalued,
                       literals_changed=any(change.w.values()))
