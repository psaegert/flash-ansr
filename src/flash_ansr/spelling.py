"""Constant re-spelling after refinement (the "ladder").

A fitted candidate carries its constants at full float precision, and the MDL ranking prices that
precision: a 17-digit float costs ~62 bits, an integer ~6. The pool never offered the ranking the
cheaper spelling, so it could not choose it. This module offers it.

For every fitted constant a menu of spellings is generated in closed form: the nearest integer,
the continued-fraction convergents up to a denominator bound, roundings to a few significant
digits, small rational multiples of pi and e (and their reciprocals), zero, and the float itself.
The fit's own curvature (the Gauss-Newton normal matrix of the residual at the optimum) predicts,
without any re-fit, how much the loss rises when one constant is moved to a spelling while the
others re-optimize. The best predicted spelling per constant is frozen as a literal, the remaining
constants are re-fitted once, and the re-spelled candidate is returned only if its real score
beats the parent's. Nothing here decides anything the ranking would not: the score is the
ranking's own, the spellings are only offered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

import numpy as np


__all__ = [
    'ConstantLadderConfig',
    'Spelling',
    'spelling_menu',
    'SpellingPricer',
    'FitCurvature',
    'choose_spellings',
    'splice_spellings',
    'slot_positions',
    'respell_fitted_candidate',
]

SPECIAL_CONSTANT_VALUES: dict[str, float] = {'np.pi': float(np.pi), 'np.e': float(np.e)}


@dataclass(frozen=True)
class ConstantLadderConfig:
    """Settings of the re-spelling step. ``None`` (the absence of the block) means off."""
    max_denominator: int = 1000          # continued-fraction convergents up to this denominator
    digits: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)   # significant-digit roundings offered
    special_constants: tuple[str, ...] = ('np.pi', 'np.e')
    special_max_denominator: int = 4     # (p/q) * s and p / (q * s) with q <= this
    special_max_numerator: int = 4
    max_relative_step: float = 0.5       # cap on the offered radius around a fitted value (relative)
    max_decades: float = 1.0             # the FVU rise (decades) the radius is derived from
    # The surprise rule for fractions: a convergent p/q is generically within 1/q^2 of the value; it is
    # offered only when it is closer than that by the factor ``fraction_surprise`` (the next partial
    # quotient of the continued fraction). 'denominator' = the factor is q itself (|x - p/q| <= 1/q^3,
    # no free constant, the default); a number = a fixed factor; None = every convergent.
    # Applied to the special-constant families through x / s as well. Integers are never filtered.
    fraction_surprise: str | float | None = 'denominator'
    # The pool bound (OFF by default): re-spell only the candidates that could still reach rank 0.
    # A candidate whose score with its MDL cut to nothing and its fit improved by ``max_decades``
    # still trails the running best cannot win by re-spelling; it is skipped (candidates are visited
    # in score order, the running best tightens as variants land). Measured on the stored 3M pools:
    # 23% of the candidates are tried, the returned answer differs on 1.6% of problems (a warm
    # re-fit that improved its parent by more than ``max_decades``), vNRR moves by 0.04 pp. It
    # prunes the fit-quality / MDL Pareto front below rank 0, which the owner wants intact and evenly
    # populated (2026-09-09), so the default re-spells EVERY candidate; True is for time-budgeted
    # runs where only the returned answer matters.
    pool_bound: bool = False
    confirm_decades: float = 0.1         # a verified score within this of its prediction ends the search
    n_restarts: int = 1                  # restarts of the verifying re-fit (warm-started)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | bool | None) -> 'ConstantLadderConfig | None':
        """``None``/``False`` -> off; ``True``/``{}`` -> defaults; a mapping -> overrides."""
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(f"constant_ladder must be a mapping, a bool or None; got {type(value).__name__}")
        kwargs: dict[str, Any] = {}
        for key, raw in value.items():
            if key == 'enabled':
                if not raw:
                    return None
                continue
            if key not in cls.__dataclass_fields__:
                raise ValueError(f"constant_ladder: unknown key {key!r}")
            if key in ('digits', 'special_constants'):
                raw = tuple(raw)
            kwargs[key] = raw
        config = cls(**kwargs)
        for name in config.special_constants:
            if name not in SPECIAL_CONSTANT_VALUES:
                raise ValueError(f"constant_ladder: unknown special constant {name!r}")
        fs = config.fraction_surprise
        if fs is not None and fs != 'denominator' and not (isinstance(fs, (int, float)) and not isinstance(fs, bool) and fs > 0):
            raise ValueError("constant_ladder: fraction_surprise must be None, 'denominator' or a positive number")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            'pool_bound': self.pool_bound,
            'fraction_surprise': self.fraction_surprise,
            'max_denominator': self.max_denominator,
            'digits': list(self.digits),
            'special_constants': list(self.special_constants),
            'special_max_denominator': self.special_max_denominator,
            'special_max_numerator': self.special_max_numerator,
            'max_relative_step': self.max_relative_step,
            'max_decades': self.max_decades,
            'confirm_decades': self.confirm_decades,
            'n_restarts': self.n_restarts,
        }


@dataclass(frozen=True)
class Spelling:
    """One way to write a constant: its prefix tokens, its exact value, and where it came from."""
    tokens: tuple[str, ...]
    value: float
    kind: str    # 'float' | 'integer' | 'rational' | 'digits' | 'special' | 'zero'

    @property
    def label(self) -> str:
        return ' '.join(self.tokens)


def _int_token(value: int) -> str:
    return str(int(value))


def _rational_tokens(fraction: Fraction) -> tuple[str, ...]:
    if fraction.denominator == 1:
        return (_int_token(fraction.numerator),)
    return ('/', _int_token(fraction.numerator), _int_token(fraction.denominator))


def _convergents(value: float, max_denominator: int) -> list[Fraction]:
    """Continued-fraction convergents of ``value`` with denominator <= ``max_denominator``.
    Each is the best rational approximation among all fractions with a denominator no larger
    than its own (Python's ``limit_denominator`` is one of them; this walks all of them)."""
    out: list[Fraction] = []
    if not math.isfinite(value):
        return out
    x = Fraction(value)
    # Euclid on the exact binary fraction; stop when the denominator bound is passed.
    h_prev, h = 0, 1   # numerators h[-2], h[-1]
    k_prev, k = 1, 0   # denominators k[-2], k[-1]
    a = math.floor(x)
    remainder = x - a
    h_prev, h = h, a * h + h_prev
    k_prev, k = k, a * k + k_prev
    out.append(Fraction(h, k))
    for _ in range(64):
        if remainder == 0 or k > max_denominator:
            break
        x = 1 / remainder
        a = math.floor(x)
        remainder = x - a
        h_prev, h = h, a * h + h_prev
        k_prev, k = k, a * k + k_prev
        if k > max_denominator:
            break
        out.append(Fraction(h, k))
    return out


def ladder_floor(entry: Mapping[str, Any], weights: Mapping[str, float], config: ConstantLadderConfig,
                 score_fn: Any) -> float:
    """The best score any re-spelling of ``entry`` could reach: its MDL cut to nothing, no constants,
    no nodes priced, and its fit error improved by ``config.max_decades``. ``score_fn(row, weights)``
    is the ranking's own score. A candidate whose floor trails the running best is skipped."""
    fvu = float(entry.get('fvu', float('nan')))
    if not math.isfinite(fvu):
        return math.inf
    row = {'fvu': fvu / (10.0 ** float(config.max_decades)), 'expression': [], 'constant_count': 0,
           'log_prob': entry.get('log_prob'), 'mdl': 0.0}
    return float(score_fn(row, weights))


def fraction_surprise(value: float, fraction: Fraction) -> float:
    """How much closer ``fraction`` = p/q is to ``value`` than a generic convergent of that size:
    1 / (|value - p/q| q^2), the next partial quotient of the continued fraction (inf when exact).
    An arbitrary real has a small one (Gauss-Kuzmin: P(a >= A) ~ 1 / (A ln 2)); a number that IS
    the fraction, fitted to precision sigma, has ~ 1 / (sigma q^2)."""
    error = abs(value - float(fraction))
    if error == 0.0:
        return math.inf
    return 1.0 / (error * fraction.denominator * fraction.denominator)


def _surprising_enough(value: float, fraction: Fraction, config: ConstantLadderConfig) -> bool:
    """The surprise rule: integers always pass; a fraction passes when its surprise reaches the
    configured bar ('denominator': q itself, i.e. |value - p/q| <= 1 / q^3; a number: that factor)."""
    if config.fraction_surprise is None or fraction.denominator == 1:
        return True
    bar = fraction.denominator if config.fraction_surprise == 'denominator' else float(config.fraction_surprise)
    return fraction_surprise(value, fraction) >= bar


def spelling_menu(value: float, config: ConstantLadderConfig, tolerance: float | None = None,
                  allow_zero: bool = True) -> list[Spelling]:
    """Every spelling offered for one fitted constant, the float itself first. Deduplicated on
    the exact value. Only spellings within ``tolerance`` (absolute; default the relative cap
    ``max_relative_step``) of the value are offered, zero included: a tiny coefficient is within
    reach of zero, a unit one is not. The caller derives the tolerance from the fit's curvature
    (see ``choose_spellings``): a precisely determined constant is offered only spellings the data
    cannot distinguish from it, a poorly determined one gets coarser spellings."""
    menu: list[Spelling] = [Spelling((repr(float(value)),), float(value), 'float')]
    if not math.isfinite(value):
        return menu
    seen: set[float] = set()   # the float itself does not block a cheaper spelling of the same value
    scale = max(1.0, abs(value))
    cap = config.max_relative_step * scale
    tolerance = cap if tolerance is None else min(float(tolerance), cap)

    def offer(tokens: Sequence[str], spelled: float, kind: str) -> None:
        spelled = float(spelled)
        if spelled in seen or not math.isfinite(spelled):
            return
        if spelled == 0.0 and not allow_zero:
            return   # a zero in a singular position, whichever generator produced it
        if spelled == float(value) and kind == 'digits':
            return   # a rounding that changes nothing is the float again
        if abs(spelled - value) > tolerance:
            return
        seen.add(spelled)
        menu.append(Spelling(tuple(tokens), spelled, kind))

    if allow_zero:
        offer(('0',), 0.0, 'zero')
    offer((_int_token(round(value)),), float(round(value)), 'integer')
    for fraction in _convergents(value, config.max_denominator):
        if not _surprising_enough(value, fraction, config):
            continue
        offer(_rational_tokens(fraction), float(fraction), 'rational' if fraction.denominator != 1 else 'integer')
    for digits in config.digits:
        rounded = float(f"{value:.{int(digits)}g}")
        offer((repr(rounded),), rounded, 'digits')
    for name in config.special_constants:
        s = SPECIAL_CONSTANT_VALUES[name]
        # value ~ (p/q) * s
        ratio = value / s
        fraction = Fraction(ratio).limit_denominator(config.special_max_denominator)
        if fraction != 0 and abs(fraction.numerator) <= config.special_max_numerator and _surprising_enough(ratio, fraction, config):
            tokens: tuple[str, ...]
            if fraction == 1:
                tokens = (name,)
            elif fraction == -1:
                tokens = ('neg', name)
            elif fraction.denominator == 1:
                tokens = ('*', _int_token(fraction.numerator), name)
            elif fraction.numerator == 1:
                tokens = ('/', name, _int_token(fraction.denominator))
            else:
                tokens = ('*', '/', _int_token(fraction.numerator), _int_token(fraction.denominator), name)
            offer(tokens, float(fraction) * s, 'special')
        # value ~ p / (q * s)
        ratio = value * s
        fraction = Fraction(ratio).limit_denominator(config.special_max_denominator)
        if fraction != 0 and abs(fraction.numerator) <= config.special_max_numerator and _surprising_enough(ratio, fraction, config):
            if fraction.denominator == 1:
                tokens = ('/', _int_token(fraction.numerator), name)
            else:
                tokens = ('/', _int_token(fraction.numerator), '*', _int_token(fraction.denominator), name)
            offer(tokens, float(fraction) / s, 'special')
    return menu


def slot_positions(expression: Sequence[str], arity: Mapping[str, int]) -> dict[int, tuple[str | None, int]]:
    """For every token index of a prefix expression: (parent operator, argument position), the root
    having ``(None, 0)``. A stack walk over the arities; tokens the arity table does not know are
    leaves."""
    out: dict[int, tuple[str | None, int]] = {}
    stack: list[list[Any]] = []   # [operator, remaining arguments, next argument index]
    for index, token in enumerate(expression):
        if stack:
            parent = stack[-1]
            out[index] = (parent[0], parent[2])
            parent[2] += 1
            parent[1] -= 1
        else:
            out[index] = (None, 0)
        n = int(arity.get(token, 0))
        if n > 0:
            stack.append([token, n, 0])
        while stack and stack[-1][1] == 0:
            stack.pop()
    return out


#: where a spelled zero is offered: sums, differences, products, and a pow EXPONENT. Anywhere else
#: (a pow base, a divisor, a log/inv/root argument) a literal 0 is a singularity, and a free partner
#: constant can then drift to where the fit's arithmetic and the pricer's folding disagree.
ZERO_POSITIONS: frozenset[tuple[str, int]] = frozenset({('+', 0), ('+', 1), ('-', 0), ('-', 1), ('*', 0), ('*', 1), ('pow', 1)})


class SpellingPricer:
    """Price (milli-bits) of a spelling on its own, through the engine's certified pricer, cached
    by token string. Used for the PREDICTED price of a variant (parent price minus the float's
    price plus the spelling's); the verifying step prices the whole realized expression."""

    def __init__(self, simplipy_engine: Any):
        self.engine = simplipy_engine
        self._cache: dict[tuple[str, ...], float] = {}

    def price(self, tokens: Sequence[str]) -> float:
        key = tuple(tokens)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        from simplipy.engine import Mode
        try:
            value = float(self.engine.complexity(list(key), certified=True, mode=Mode.f64, canon='default'))
        except Exception:
            value = float('inf')
        if len(self._cache) < 65536:
            self._cache[key] = value
        return value


class FitCurvature:
    """The Gauss-Newton normal matrix A = J^T J of the residual at the fitted optimum (central
    differences on ``predict``). ``stiffness(S)`` is the Schur complement of the block S: moving
    the constants in S by delta_S while every other constant re-optimizes raises the residual sum
    of squares by about delta_S^T stiffness(S) delta_S. A redundant constant (a null direction of
    A once the others may move) has stiffness ~0, which is exactly right: re-spelling it is free.
    ``None`` from ``build`` when the Jacobian is not finite."""

    def __init__(self, A: np.ndarray):
        self.A = A
        self.p = A.shape[0]
        self._scale = np.sqrt(np.maximum(np.diag(A), 1e-300))

    @classmethod
    def build(cls, predict: Callable[[np.ndarray], np.ndarray], constants: np.ndarray,
              relative_step: float = 1e-6) -> 'FitCurvature | None':
        constants = np.asarray(constants, dtype=float)
        p = constants.size
        if p == 0:
            return cls(np.zeros((0, 0)))
        columns = []
        for i in range(p):
            h = relative_step * max(1.0, abs(constants[i]))
            up = constants.copy()
            up[i] += h
            down = constants.copy()
            down[i] -= h
            with np.errstate(all='ignore'):
                column = (np.asarray(predict(up), dtype=float) - np.asarray(predict(down), dtype=float)) / (2.0 * h)
            if not np.all(np.isfinite(column)):
                return None
            columns.append(column)
        J = np.stack(columns, axis=1)
        return cls(J.T @ J)

    def stiffness(self, subset: Sequence[int]) -> np.ndarray:
        """Schur complement of the block ``subset`` against all other constants (pseudo-inverse on
        the complement, so a singular complement is handled), in the original units."""
        S = list(subset)
        if not S:
            return np.zeros((0, 0))
        T = [i for i in range(self.p) if i not in S]
        # work in column-scaled units for conditioning, undo the scaling at the end
        D = self._scale
        B = self.A / np.outer(D, D)
        B_SS = B[np.ix_(S, S)]
        if T:
            B_TT = B[np.ix_(T, T)]
            B_ST = B[np.ix_(S, T)]
            schur = B_SS - B_ST @ np.linalg.pinv(B_TT, rcond=1e-10, hermitian=True) @ B_ST.T
        else:
            schur = B_SS
        schur = np.where(np.abs(schur) < 1e-9, 0.0, schur)   # numerically null directions are free
        d = D[S]
        return schur * np.outer(d, d)


@dataclass
class SpellingChoice:
    """The spelling chosen for one fitted constant (slot ``index`` in the fitted order)."""
    index: int
    spelling: Spelling
    delta: float
    predicted_extra_rss: float
    predicted_score: float


def choose_spellings(constants: Sequence[float], curvature: FitCurvature, base_rss: float,
                     n_samples: int, y_variance: float, base_mdl: float,
                     score_fn: Callable[[float, float, int], float], pricer: SpellingPricer,
                     config: ConstantLadderConfig, parent_score: float,
                     zero_allowed: Sequence[bool] | None = None) -> list[SpellingChoice]:
    """Per constant, the menu entry with the best PREDICTED score, keeping only constants whose
    best entry is not the float itself and beats the parent's score. ``score_fn(fvu, bits,
    constant_count)`` is the ranking's own score; ``base_bits`` the parent's priced milli-bits."""
    choices: list[SpellingChoice] = []
    constants = [float(c) for c in constants]
    n_free = len(constants)
    # The radius offered around each constant: the move that would raise the residual by
    # ``max_decades`` of FVU (at least the float64 floor the scorer applies), under the fit's own
    # curvature. A constant the data pins down precisely gets a tiny radius; a redundant one
    # (stiffness ~0) gets the cap.
    rss_floor = float(n_samples) * float(y_variance) * float(np.finfo(np.float64).eps)
    allowed_extra = max(base_rss, rss_floor) * (10.0 ** config.max_decades - 1.0)
    for i, c in enumerate(constants):
        k_i = float(curvature.stiffness([i])[0, 0]) if curvature.p else 0.0
        radius = math.sqrt(allowed_extra / k_i) if np.isfinite(k_i) and k_i > 0 else float('inf')
        menu = spelling_menu(c, config, tolerance=radius,
                             allow_zero=True if zero_allowed is None else bool(zero_allowed[i]))
        float_price = pricer.price(menu[0].tokens)
        best: SpellingChoice | None = None
        for spelling in menu[1:]:
            delta = spelling.value - c
            extra_rss = delta * delta * k_i if np.isfinite(k_i) else float('inf')
            if not np.isfinite(extra_rss):
                continue
            fvu_pred = (base_rss + extra_rss) / max(n_samples, 1) / y_variance if y_variance > 0 else float('inf')
            mdl_pred = base_mdl - float_price + pricer.price(spelling.tokens)
            if not np.isfinite(mdl_pred):
                continue
            score_pred = score_fn(fvu_pred, mdl_pred, n_free - 1)
            if best is None or score_pred < best.predicted_score:
                best = SpellingChoice(i, spelling, delta, extra_rss, score_pred)
        if best is not None and best.predicted_score <= parent_score + 1e-12:
            choices.append(best)
    return choices


def joint_extra_rss(choices: Sequence[SpellingChoice], curvature: FitCurvature) -> float:
    """Predicted rise of the residual sum of squares when all ``choices`` are frozen together and
    the other constants re-optimize."""
    if not choices:
        return 0.0
    idx = [c.index for c in choices]
    delta = np.array([c.delta for c in choices], dtype=float)
    return float(delta @ curvature.stiffness(idx) @ delta)


def splice_spellings(expression: Sequence[str], slot_indices: Sequence[int],
                     choices: Mapping[int, Spelling]) -> tuple[list[str], list[int]]:
    """The variant's prefix expression: every fitted slot becomes ``<constant>`` unless a spelling
    was chosen for it, in which case the spelling's tokens are spliced in. Returns the tokens and
    the fitted-order indices of the slots that stay free."""
    slot_set = {int(i): k for k, i in enumerate(slot_indices)}
    out: list[str] = []
    free: list[int] = []
    for position, token in enumerate(expression):
        k = slot_set.get(position)
        if k is None:
            out.append(token)
        elif k in choices:
            out.extend(choices[k].tokens)
        else:
            out.append('<constant>')
            free.append(k)
    return out, free


def spelling_record(choices: Sequence[SpellingChoice]) -> str:
    """Compact provenance: ``c0=/ 3 2;c2=0``."""
    return ';'.join(f"c{c.index}={c.spelling.label}" for c in sorted(choices, key=lambda c: c.index))


# ----------------------------------------------------------------------------------------------
# The step itself: predict, choose, verify. Called by the refinement worker on a fitted candidate.
# ----------------------------------------------------------------------------------------------

def _is_variable(token: str) -> bool:
    return len(token) > 1 and token[0] == 'x' and token[1:].isdigit()


def _predict_with(refiner: Any, X: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """The refiner's prediction on ``X`` for the given constants, coerced like its fit does."""
    y_pred = refiner.expression_lambda(*X.T, *constants) if len(constants) else refiner.expression_lambda(*X.T)
    n_samples = X.shape[0]
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_pred, np.ndarray):
        if y_pred.ndim == 0:
            y_pred = np.full((n_samples,), float(y_pred))
        else:
            y_pred = y_pred.reshape(-1)
            if y_pred.size == 1 and n_samples > 1:
                y_pred = np.full((n_samples,), float(y_pred[0]))
            elif y_pred.size != n_samples:
                y_pred = np.resize(y_pred, n_samples)
    else:
        y_pred = np.full((n_samples,), y_pred)
    return np.asarray(y_pred, dtype=float)


def respell_fitted_candidate(
        *,
        refiner: Any,
        expression: Sequence[str],
        X: np.ndarray,
        y: np.ndarray,
        y_variance: float,
        parent_fvu: float,
        parent_mdl: float | None,
        parent_score: float,
        score_fn: Callable[[float, float, int], float],
        compute_fvu: Callable[[float, int, float], float],
        price_realized: Callable[[Any, Sequence[str]], float | None],
        simplipy_engine: Any,
        n_variables: int,
        method: str,
        full_fit: Mapping[str, Any],
        config: ConstantLadderConfig,
        pricer: SpellingPricer | None = None,
) -> dict[str, Any] | None:
    """Try to re-spell the fitted constants of one candidate. Returns the variant's fields
    (``expression`` with the surviving constants as ``<constant>``, ``refiner`` fitted under
    ``refine_scope='placeholders'``, ``constant_count``, ``complexity``, ``mdl``, ``fvu``,
    ``score``, ``spelling``) or ``None`` when no spelling beats the parent's score.

    ``score_fn(fvu, mdl_millibits, constant_count)`` is the ranking's score for this candidate,
    ``compute_fvu(loss, n_samples, y_variance)`` the harness's FVU, ``price_realized(refiner,
    expression)`` the harness's MDL price of a realized expression, ``full_fit`` the kwargs of a
    cold fit (``n_restarts``, ``p0_noise``, ``p0_noise_kwargs``) for the rare case where the
    canonical form of a variant changes its constant slots.
    """
    from flash_ansr.refine import Refiner   # local: refine imports nothing from here

    if parent_mdl is None or not np.isfinite(parent_mdl) or not np.isfinite(parent_score):
        return None
    fits = getattr(refiner, '_all_constants_values', None) or []
    if not fits:
        return None
    constants = np.asarray(fits[0][0], dtype=float).ravel()
    slots = list(getattr(refiner, 'slot_indices', []))
    if constants.size == 0 or len(slots) != constants.size or not np.all(np.isfinite(constants)):
        return None
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    base_rss = float(fits[0][2]) * n
    if not np.isfinite(base_rss):
        return None
    pricer = pricer or SpellingPricer(simplipy_engine)

    curvature = FitCurvature.build(lambda cc: _predict_with(refiner, X, cc), constants)
    if curvature is None:
        return None
    positions = slot_positions(list(expression), getattr(simplipy_engine, 'operator_arity', {}) or {})
    zero_allowed = [positions.get(int(slot), (None, 0)) in ZERO_POSITIONS or positions.get(int(slot), (None, 0))[0] is None
                    for slot in slots]
    choices = choose_spellings(constants, curvature, base_rss, n, float(y_variance), float(parent_mdl),
                               score_fn, pricer, config, float(parent_score), zero_allowed=zero_allowed)
    if not choices:
        return None

    # joint prediction, then the single best as the fallback
    def predicted_score(subset: Sequence[SpellingChoice]) -> float:
        extra = joint_extra_rss(subset, curvature)
        fvu_pred = (base_rss + extra) / max(n, 1) / float(y_variance) if y_variance > 0 else float('inf')
        mdl_pred = float(parent_mdl) - sum(pricer.price((repr(constants[c.index]),)) for c in subset) \
            + sum(pricer.price(c.spelling.tokens) for c in subset)
        return score_fn(fvu_pred, mdl_pred, constants.size - len(subset))

    # Verify in predicted order: the joint move first when it is predicted to pay, then the single
    # moves best-first. Stop at the first variant whose REAL score confirms its prediction (within
    # ``confirm_decades``); when a prediction fails -- a multiplicative constant sent to zero is
    # "free" under the local quadratic model and catastrophic in truth -- go on to the next. The
    # ranking therefore only ever sees real re-fits.
    attempts: list[tuple[list[SpellingChoice], float]] = []
    if len(choices) > 1:
        joint_pred = predicted_score(choices)
        if joint_pred <= parent_score + 1e-12:
            attempts.append((list(choices), joint_pred))
    attempts.extend(([c], c.predicted_score) for c in sorted(choices, key=lambda c: c.predicted_score))

    expression = list(expression)
    parent_reads = any(_is_variable(t) for t in expression)
    variable_price = pricer.price(('x1',))
    best: dict[str, Any] | None = None
    for subset, predicted in attempts:
        chosen = {c.index: c.spelling for c in subset}
        tokens, free = splice_spellings(expression, slots, chosen)
        p0 = np.asarray([constants[k] for k in free], dtype=float)
        verified = _fit_variant(Refiner, simplipy_engine, n_variables, tokens, X, y, p0, method, config, full_fit)
        if verified is None:
            continue
        fitted, loss = verified
        fvu = float(compute_fvu(loss, n, float(y_variance)))
        if not np.isfinite(fvu) or not _well_defined(fitted, tokens, expression, X, simplipy_engine):
            continue
        mdl = price_realized(fitted, tokens)
        if mdl is None or (parent_reads and float(mdl) < variable_price):
            continue   # a function of the data priced below a bare variable: the pricer folded a singularity
        score = score_fn(fvu, float(mdl), len(free))
        # a tie is accepted: the ranking is indifferent, and the variant is the canonical spelling
        # of an exactly fitted simple value (the pricer already priced `2.0` as `2`); it then
        # REPLACES its parent in the pool instead of standing beside it
        if score > parent_score + 1e-12 or (best is not None and score >= best['score']):
            continue
        best = {'tokens': tokens, 'refiner': fitted, 'fvu': fvu, 'mdl': mdl, 'score': score, 'p0': p0, 'subset': subset}
        if score <= predicted + config.confirm_decades:
            break
    if best is None:
        return None
    tokens, fitted, fvu, mdl, score, p0, subset = (best[k] for k in ('tokens', 'refiner', 'fvu', 'mdl', 'score', 'p0', 'subset'))
    # canonicalize for the record (the ranking saw the canonical price already)
    final_tokens, final_refiner, final_loss = _canonical_form(
        Refiner, simplipy_engine, n_variables, tokens, fitted, X, y, p0, method, config, full_fit)
    if final_refiner is not fitted:
        fvu_c = float(compute_fvu(final_loss, n, float(y_variance)))
        mdl_c = price_realized(final_refiner, final_tokens)
        if (mdl_c is not None and np.isfinite(fvu_c) and not (parent_reads and float(mdl_c) < variable_price)
                and _well_defined(final_refiner, final_tokens, expression, X, simplipy_engine)):
            score_c = score_fn(fvu_c, float(mdl_c), sum(t == '<constant>' for t in final_tokens))
            if score_c <= score + 1e-12:
                tokens, fitted, fvu, mdl, score = final_tokens, final_refiner, fvu_c, mdl_c, score_c
    if tokens == expression:
        return None
    # the ladder spells constants, it does not restructure: a parent that reads the data must not
    # collapse into a bare number (the ranking would take a 2-bit constant over a weak fit at
    # fvu ~1, which is its exchange rate at work on a candidate that was never going to win)
    if not any(_is_variable(t) for t in tokens) and any(_is_variable(t) for t in expression):
        return None
    return {
        'expression': list(tokens),
        'refiner': fitted,
        'constant_count': int(sum(t == '<constant>' for t in tokens)),
        'complexity': len(tokens),
        'mdl': float(mdl),
        'fvu': float(fvu),
        'score': float(score),
        'spelling': spelling_record(subset),
        'replaces_parent': bool(score >= parent_score - 1e-12),
    }


def _well_defined(fitted: Any, tokens: Sequence[str], parent_tokens: Sequence[str], X: np.ndarray, engine: Any) -> bool:
    """A variant must still be the kind of thing its parent was. A spelled zero can land as the base
    of a ``pow`` or under a ``log`` whose free partner then drifts to a singular value: the fit's
    arithmetic evaluates the dead subtree one way, the pricer's constant folding another, and the
    ranking would be handed a 2-bit price for a function that is not what it says. So: finite
    predictions that vary (when the parent read a variable), and a realized canonical form that
    keeps a variable and carries no non-finite literal."""
    try:
        constants = np.asarray(fitted._all_constants_values[0][0], dtype=float).ravel()
        y_pred = _predict_with(fitted, X, constants)
        if not np.all(np.isfinite(y_pred)):
            return False
        parent_reads = any(_is_variable(t) for t in parent_tokens)
        if parent_reads and not (np.std(y_pred) > 0.0):
            return False
        realized = list(fitted.transform(expression=list(tokens), return_prefix=True, variable_mapping=None))
        canon = list(engine.simplify(realized))
    except Exception:
        return False
    if parent_reads and not any(_is_variable(t) for t in canon):
        return False
    for t in canon:
        lowered = str(t).lower()
        if 'nan' in lowered or 'inf' in lowered:
            return False
    return True


def _fit_variant(Refiner: Any, engine: Any, n_variables: int, tokens: list[str], X: np.ndarray, y: np.ndarray,
                 p0: np.ndarray, method: str, config: ConstantLadderConfig,
                 full_fit: Mapping[str, Any]) -> tuple[Any, float] | None:
    """Warm-started re-fit of the free ``<constant>`` slots (the spelled literals are frozen under
    ``refine_scope='placeholders'``); a cold fit when the warm start does not converge."""
    n_free = sum(t == '<constant>' for t in tokens)
    try:
        fitted = Refiner(simplipy_engine=engine, n_variables=n_variables)
        if n_free == p0.size:
            fitted.fit(tokens, X, y, p0=p0 if p0.size else None, p0_noise=None, p0_noise_kwargs=None,
                       n_restarts=max(1, int(config.n_restarts)), method=method, converge_error='ignore',
                       refine_scope='placeholders')
        if n_free != p0.size or not fitted.valid_fit:
            fitted = Refiner(simplipy_engine=engine, n_variables=n_variables)
            fitted.fit(tokens, X, y, p0=None, p0_noise=full_fit.get('p0_noise', 'normal'),
                       p0_noise_kwargs=full_fit.get('p0_noise_kwargs'), n_restarts=int(full_fit.get('n_restarts', 8)),
                       method=method, converge_error='ignore', refine_scope='placeholders')
    except Exception:
        return None
    if not fitted.valid_fit or not fitted._all_constants_values:
        return None
    loss = float(fitted._all_constants_values[0][2])
    return (fitted, loss) if np.isfinite(loss) else None


def _canonical_form(Refiner: Any, engine: Any, n_variables: int, tokens: list[str], fitted: Any, X: np.ndarray,
                    y: np.ndarray, p0: np.ndarray, method: str, config: ConstantLadderConfig,
                    full_fit: Mapping[str, Any]) -> tuple[list[str], Any, float]:
    """simplipy's canonical form of the variant (``* 1 x`` -> ``x``, ``pow x 0.5`` -> ``rootn x 2``),
    re-fitted so its constants match its own slots. Falls back to the un-simplified variant."""
    try:
        simplified = list(engine.simplify(list(tokens)))
    except Exception:
        return tokens, fitted, float(fitted._all_constants_values[0][2])
    if simplified == tokens or not engine.is_valid(simplified):
        return tokens, fitted, float(fitted._all_constants_values[0][2])
    free_values = np.asarray(fitted._all_constants_values[0][0], dtype=float).ravel()
    # the canonical form usually keeps the free slots as they were (literal folding only): then the
    # verified constants carry over, and one evaluation confirms it instead of a re-fit
    if sum(t == '<constant>' for t in simplified) == free_values.size:
        try:
            carried = Refiner.from_serialized(simplipy_engine=engine, n_variables=n_variables, expression=simplified,
                                              n_inputs=X.shape[1], fits=list(fitted._all_constants_values), refine_scope='placeholders')
            if carried.valid_fit:
                y_old = _predict_with(fitted, X, free_values)
                y_new = _predict_with(carried, X, free_values)
                if np.allclose(y_old, y_new, rtol=1e-9, atol=1e-12, equal_nan=True):
                    return simplified, carried, float(fitted._all_constants_values[0][2])
        except Exception:
            pass
    verified = _fit_variant(Refiner, engine, n_variables, simplified, X, y, free_values, method, config, full_fit)
    if verified is None:
        return tokens, fitted, float(fitted._all_constants_values[0][2])
    return simplified, verified[0], verified[1]
