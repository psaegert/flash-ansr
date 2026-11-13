"""Real-valued unused input columns -- distractors -- in the training stream (owner rulings 2026-11-13).

The catalog draws every one of the pool's input columns from the same support prior, and the stream zeroes the
columns the expression does not use (``padding: 'zero'``). The model therefore never trains on an irrelevant column
that carries values, while real data has them (srsd-dummy's dummies, Korns' unused inputs, the data catalogs' extra
features). The distractor block keeps some of those columns as drawn:

* how many: none with probability ``1 - p_instance``; otherwise one, and each further one with probability
  ``p_continue`` -- a geometric chain whose only limit is the number of zeroed columns (owner: "more distractors
  get more unlikely but there is no cap"; ruled 20 % of instances, then a fair coin);
* which: uniformly among the zeroed columns (the expression's own variables are already scattered over the pool);
* values: the column's own draw from the support prior, so before any change of units a distractor looks exactly
  like a used column;
* units: on an instance whose units changed (:mod:`flash_ansr.data.units`), each distractor is scaled by the power
  of two of a randomly chosen variable of the expression, which has no unit change when the law left it out.
  Without this, magnitude alone separates used from unused columns: measured 2026-11-13 on 1,024 stream instances
  with units on, a used column is the larger of a used/unused pair 84 % of the time on unit-changed instances
  (52 % with the borrowed scale, 51 % on instances without a change).

The target, the holdout check and the validation stream are untouched: a distractor is data the expression ignores.
"""
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

DISTRACTOR_BLOCK_KEYS = ("p_instance", "p_continue")
DISTRACTOR_COUNTER_KEYS = ("n_distractor_instances", "n_distractor_columns", "n_distractor_scale_fallbacks")

_TINY = np.finfo(np.float64).tiny


def validate_distractor_block(raw: "dict[str, Any] | None") -> "dict[str, Any] | None":
    """The ``distractor_block`` config: exact keys, pinned explicitly (never defaulted), like the other blocks."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"distractor_block must be a mapping, got {raw!r}")
    if set(raw) != set(DISTRACTOR_BLOCK_KEYS):
        raise ValueError(f"distractor_block must carry exactly {sorted(DISTRACTOR_BLOCK_KEYS)} (got {sorted(raw)}); "
                         f"priors are pinned explicitly, never defaulted")
    out: dict[str, Any] = {}
    for key in DISTRACTOR_BLOCK_KEYS:
        value = float(raw[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"distractor_block.{key} must be a probability in [0, 1], got {raw[key]!r}")
        out[key] = value
    return out


@dataclass
class DistractorDraw:
    """The columns kept as distractors, the power of two each was scaled by (absent = unscaled), and how many
    borrowed scales were refused because the scaled column would leave float64's normal range."""
    columns: list[str] = field(default_factory=list)
    scales: dict[str, int] = field(default_factory=dict)
    scale_fallbacks: int = 0


def draw_distractors(*, x_support: np.ndarray, variables: Sequence[str], used: "set[str]", rng: np.random.Generator,
                     cfg: Mapping[str, Any], unit_z: "Mapping[str, int] | None" = None) -> DistractorDraw:
    """Choose this instance's distractors and scale them IN PLACE in ``x_support`` (columns follow ``variables``).

    ``used`` is the set of variables the expression uses; ``unit_z`` the instance's change of units (variable ->
    power of two), None when its units did not change. The caller zeroes every unused column NOT returned here.
    """
    n_columns = x_support.shape[1] if x_support.ndim == 2 else 0
    zeroed = [v for v in variables[:n_columns] if v not in used]
    if not zeroed or rng.random() >= cfg["p_instance"]:
        return DistractorDraw()
    count = 1
    while count < len(zeroed) and rng.random() < cfg["p_continue"]:
        count += 1
    picks = rng.choice(len(zeroed), size=count, replace=False)
    draw = DistractorDraw(columns=[zeroed[int(k)] for k in sorted(picks)])
    if unit_z is None:
        return draw
    # Every variable of the expression, with 0 where the law left it unscaled: the realized scales of used columns.
    borrowed = [int(unit_z.get(v, 0)) for v in variables[:n_columns] if v in used]
    if not borrowed:
        return draw
    index = {v: j for j, v in enumerate(variables[:n_columns])}
    for name in draw.columns:
        z = borrowed[int(rng.integers(len(borrowed)))]
        if z == 0:
            continue
        with np.errstate(over="ignore", under="ignore"):
            scaled = np.ldexp(np.asarray(x_support[:, index[name]], dtype=np.float64), z)
        if np.all(np.isfinite(scaled)) and np.all((scaled == 0) | (np.abs(scaled) >= _TINY)):
            x_support[:, index[name]] = scaled
            draw.scales[name] = z
        else:
            draw.scale_fallbacks += 1
    return draw
