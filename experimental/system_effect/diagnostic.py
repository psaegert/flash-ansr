"""Hypothesis-space coverage diagnostic.

A standardized, single-sourced definition of the candidate-distribution
fingerprint used across §8.1 (training-data ablation), §8.3
(inference-strategy ablation), and optionally §8.2 (architecture ablation).

For each system configuration, after running ``model.fit`` on a test problem,
we have a list of refined candidate hypotheses. We characterize the
distribution of those candidates with five dimensions:

1. **Coverage**: number of distinct canonical hypotheses found per
   refined candidate. Higher is broader exploration.
   ``coverage = n_canonical_unique / n_candidates``

2. **Quality**: fraction of distinct canonical hypotheses with at least one
   perfect-fit member (FVU ≤ ε₃₂). Higher is more often correct.
   ``quality = n_canonical_groups_with_perfect / n_canonical_unique``

3. **Concentration**: fraction of refined candidates that are syntactic
   rewrites of an already-encountered canonical hypothesis. Lower is more
   diverse use of compute.
   ``concentration = (n_candidates - n_canonical_unique) / n_candidates``

4. **Surface form**: mean ratio of raw token length to canonical length per
   candidate. 1.0 means every candidate is already canonical; >1.0 indicates
   bloat.
   ``surface_form = mean(len(raw_i) / len(canonical_i))``

5. **Targeting**: indicator of whether the GT canonical hypothesis is
   present in the candidate set. Mean across samples gives "GT canonical
   recall."
   ``targeting = 1 if gt_canonical ∈ canonical_groups else 0``

The canonical-form oracle is SimpliPy itself
(``engine.simplify(skel, max_pattern_length=4)``). We call this the
*self-application* of the simplifier as a measurement instrument; see App N
of the camera-ready for the methodological note.

Usage::

    import pickle
    from experimental.system_effect.diagnostic import (
        compute_sample_fingerprints,
        aggregate_fingerprint,
    )
    from simplipy import SimpliPyEngine

    engine = SimpliPyEngine.load("dev_7-3", install=False)
    payload = pickle.load(open("results/system_effect/cand_dist_*.pkl", "rb"))

    sample_fps = compute_sample_fingerprints(engine, payload["samples"])
    config_fp = aggregate_fingerprint(sample_fps, n_bootstrap=1000)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

EPS32 = float(np.finfo(np.float32).eps)


@dataclass
class SampleFingerprint:
    """Per-test-problem fingerprint."""
    eq_id: str
    sample_index: int
    n_candidates: int
    n_canonical_unique: int
    canonical_gt_len: int

    coverage: float
    quality: float
    concentration: float
    surface_form: float
    targeting: int

    # Auxiliary numbers we may want for stratified analysis or plots.
    max_canonical_group_size: int
    n_perfect_groups: int
    n_good_groups: int
    fit_time: float


@dataclass
class AggregateFingerprint:
    """Aggregated fingerprint over a configuration's samples (mean + 95% bootstrap CI)."""
    label: str
    n_samples: int

    coverage: tuple[float, float, float]
    quality: tuple[float, float, float]
    concentration: tuple[float, float, float]
    surface_form: tuple[float, float, float]
    targeting: tuple[float, float, float]

    # Extras: useful in figures even if not part of the headline 5-axis fingerprint.
    n_canonical_unique: tuple[float, float, float]
    fit_time: tuple[float, float, float]
    max_canonical_group_size: tuple[float, float, float]


# --- Internal helpers ---


# Module-level memoization for canonicalization. Same raw skeleton ->
# same canonical form regardless of which sample / which probe it came from,
# so a global cache cuts repeat work across cells / probes / runs.
_CANONICAL_CACHE: dict[tuple, tuple] = {}


def _safe_simplify(engine, skel):
    if not skel:
        return tuple(skel) if skel else tuple()
    key = tuple(skel)
    cached = _CANONICAL_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        out = engine.simplify(list(skel), max_pattern_length=4)
        result = tuple(out) if engine.is_valid(out) else key
    except Exception:
        result = key
    _CANONICAL_CACHE[key] = result
    return result


def safe_simplify(engine, skel):
    """Public wrapper around the cached canonicalization."""
    return _safe_simplify(engine, skel)


def normalize_skeleton(expr_tokens):
    """Replace numeric literals with the ``<constant>`` placeholder.

    Public wrapper used by notebooks and plotters so we share one definition.
    """
    out = []
    for t in expr_tokens:
        if t == "<constant>":
            out.append("<constant>")
        else:
            try:
                float(t)
                out.append("<constant>")
            except (TypeError, ValueError):
                out.append(t)
    return tuple(out)


# Internal alias kept for backward compatibility with existing call sites.
_normalize_skeleton = normalize_skeleton


def cache_stats() -> dict:
    """Snapshot of the canonicalization cache for telemetry / debugging."""
    return {"cache_size": len(_CANONICAL_CACHE)}


def _bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=0):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(a, size=a.size, replace=True).mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(a.mean()), float(lo), float(hi)


# --- Public API ---


def compute_sample_fingerprint(engine, sample) -> SampleFingerprint | None:
    """Compute the five-dimension fingerprint for a single probed sample.

    Returns ``None`` when the candidate set is empty (the fingerprint is
    undefined in that case).
    """
    cands = sample["candidates"]
    if not cands:
        return None

    raw_skels = [_normalize_skeleton(c["expression"]) for c in cands]
    canonical_skels = [_safe_simplify(engine, rs) for rs in raw_skels]

    n = len(cands)
    n_canonical_unique = len(set(canonical_skels))

    # Surface form: mean bloat over candidates with non-empty canonical.
    bloats = [
        len(rs) / max(len(cs), 1)
        for rs, cs in zip(raw_skels, canonical_skels)
        if len(cs) > 0
    ]
    surface_form = float(np.mean(bloats)) if bloats else float("nan")

    # Concentration: fraction of candidates that are rewrites of an
    # already-seen canonical hypothesis (i.e. all but the first occurrence
    # of each canonical class).
    seen = set()
    rewrites = 0
    for cs in canonical_skels:
        if cs in seen:
            rewrites += 1
        else:
            seen.add(cs)
    concentration = rewrites / n

    # Group-level quality.
    groups = defaultdict(list)
    for cs, c in zip(canonical_skels, cands):
        groups[cs].append(c)
    n_perfect_groups = sum(
        1 for g in groups.values()
        if any(np.isfinite(c["fvu"]) and c["fvu"] <= EPS32 for c in g)
    )
    n_good_groups = sum(
        1 for g in groups.values()
        if any(np.isfinite(c["fvu"]) and c["fvu"] <= 1e-3 for c in g)
    )
    quality = n_perfect_groups / max(n_canonical_unique, 1)

    coverage = n_canonical_unique / n

    gt_canonical = (
        tuple(sample["skeleton_canonical_gt"])
        if sample.get("skeleton_canonical_gt") else None
    )
    targeting = int(bool(gt_canonical) and gt_canonical in groups)

    group_sizes = [len(g) for g in groups.values()]
    max_group = max(group_sizes) if group_sizes else 0

    return SampleFingerprint(
        eq_id=sample["eq_id"],
        sample_index=sample.get("sample_index", -1),
        n_candidates=n,
        n_canonical_unique=n_canonical_unique,
        canonical_gt_len=len(gt_canonical) if gt_canonical else 0,
        coverage=coverage,
        quality=quality,
        concentration=concentration,
        surface_form=surface_form,
        targeting=targeting,
        max_canonical_group_size=max_group,
        n_perfect_groups=n_perfect_groups,
        n_good_groups=n_good_groups,
        fit_time=sample["fit_time"],
    )


def compute_sample_fingerprints(engine, samples) -> list[SampleFingerprint]:
    out = []
    for s in samples:
        fp = compute_sample_fingerprint(engine, s)
        if fp is not None:
            out.append(fp)
    return out


def compute_sample_fingerprints_cached(engine, samples, cache_path) -> list[SampleFingerprint]:
    """Compute fingerprints, persisting them to ``cache_path`` for re-use.

    The first call writes a pickle next to the dump; later calls load it.
    Cache is keyed only by ``cache_path`` --- the caller is responsible for
    using a stable path that captures (model, decoder, choices, n_samples).
    """
    import os
    import pickle as _pickle

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = _pickle.load(f)
            if isinstance(cached, list) and len(cached) == len(samples):
                return cached
        except Exception:
            pass  # fall through to recompute

    fps = compute_sample_fingerprints(engine, samples)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            _pickle.dump(fps, f)
    return fps


def aggregate_fingerprint(
    sample_fps: Sequence[SampleFingerprint],
    label: str = "",
    n_bootstrap: int = 1000,
) -> AggregateFingerprint:
    """Aggregate per-sample fingerprints with bootstrap 95% CIs."""
    def col(name):
        return [getattr(f, name) for f in sample_fps]

    return AggregateFingerprint(
        label=label,
        n_samples=len(sample_fps),
        coverage=_bootstrap_ci(col("coverage"), n_resamples=n_bootstrap),
        quality=_bootstrap_ci(col("quality"), n_resamples=n_bootstrap),
        concentration=_bootstrap_ci(col("concentration"), n_resamples=n_bootstrap),
        surface_form=_bootstrap_ci(col("surface_form"), n_resamples=n_bootstrap),
        targeting=_bootstrap_ci(col("targeting"), n_resamples=n_bootstrap),
        n_canonical_unique=_bootstrap_ci(col("n_canonical_unique"), n_resamples=n_bootstrap),
        fit_time=_bootstrap_ci(col("fit_time"), n_resamples=n_bootstrap),
        max_canonical_group_size=_bootstrap_ci(col("max_canonical_group_size"), n_resamples=n_bootstrap),
    )


# Convenience: which dimensions are "higher is better" vs "lower is better".
# Used by fingerprint_plot.py to flip directionality if desired.
DIMENSION_DIRECTION = {
    "coverage": +1,        # higher = broader hypothesis exploration
    "quality": +1,         # higher = more often correct
    "concentration": -1,   # higher = more compute on rewrites of same hypothesis
    "surface_form": -1,    # higher = more bloat per candidate
    "targeting": +1,       # higher = GT canonical present more often
}


FIVE_AXES = ("coverage", "quality", "concentration", "surface_form", "targeting")


def fingerprint_table(aggregates: Iterable[AggregateFingerprint]) -> str:
    """Format a side-by-side table of aggregate fingerprints for CLI/log use."""
    aggs = list(aggregates)
    if not aggs:
        return "(no fingerprints)"
    name_w = max(len("dimension"), max(len(a.label) for a in aggs))
    header = f"  {'dimension':<{max(20, name_w)}}" + "".join(
        f"{a.label:>22}" for a in aggs
    )
    lines = [header]
    for axis in FIVE_AXES:
        row = f"  {axis:<{max(20, name_w)}}"
        for a in aggs:
            m, lo, hi = getattr(a, axis)
            row += f"{m:>10.3f} [{lo:>5.3f},{hi:>5.3f}]"
        lines.append(row)
    return "\n".join(lines)
