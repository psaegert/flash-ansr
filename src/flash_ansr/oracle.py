"""The oracle: the problem's ground truth as the only candidate, through the unchanged refinement and
ranking.

The ceiling of the fitting stage. ``method: oracle`` replaces the decoder by the ground-truth expression
the harness hands over with the problem (``OracleConfig(expression=...)``), serialized the way the prior
sampler serializes a draw -- the model's emission format under the fittable flag: every fittable literal
is a ``<constant>`` placeholder, a structural literal (pow exponent, rootn index) stays spelled. The
refiner then fits the placeholders exactly as it fits a decoded candidate's, and its restarts are the
oracle's budget. What it measures: how often fitting alone recovers a law whose structure is known.

A ground truth the model's vocabulary cannot spell (a variable beyond the model's range, a token it lacks)
yields no candidate, so the problem has no prediction.
"""
from __future__ import annotations

from typing import Any, Sequence

from flash_ansr.prior import encode_candidate, wraps_expressions


def oracle_beams(expression: Sequence[str], *, engine: Any,
                 tokenizer: Any) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
    """The ground truth as one beam in the ``generate`` contract ``(beams, log_probs, completed,
    rewards)``; no beam when the vocabulary cannot spell it. It carries no log-probability (``nan``)."""
    encoded = encode_candidate(list(expression), engine=engine, tokenizer=tokenizer,
                               wrap=wraps_expressions(tokenizer))
    if encoded is None:
        return [], [], [], []
    return [encoded[0]], [float("nan")], [True], [float("nan")]


__all__ = ["oracle_beams"]
