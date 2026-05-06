"""Analyze one or more candidate-distribution dumps using the standardized
hypothesis-space coverage diagnostic (see ``diagnostic.py``).

For each dump, prints the five-dimension fingerprint with bootstrap 95% CIs.
With ≥2 dumps, also runs paired Wilcoxon signed-rank tests on each fingerprint
dimension across matched (eq_id, sample_index) pairs and reports the
top rewrite-cluster cases.

Usage::

    python experimental/system_effect/analyze_candidates.py \\
        results/system_effect/cand_dist_v23.0-20M-A-S100_choices02048_n300.pkl \\
        results/system_effect/cand_dist_v23.0-20M-A-U100_choices02048_n300.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simplipy import SimpliPyEngine  # noqa: E402

from diagnostic import (  # noqa: E402
    FIVE_AXES,
    aggregate_fingerprint,
    compute_sample_fingerprints,
    fingerprint_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dumps", nargs="+", help="One or more candidate-distribution pickle dumps.")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples for CI estimation (default: 1000).")
    return p.parse_args()


def label_for_payload(payload, dump_path):
    """Short, human-readable label for plots/tables."""
    model = Path(payload.get("model_path", "?")).name
    decoder = payload.get("decoder", "?")
    simp = "S" if payload.get("simplify") else "U"
    c = payload.get("choices", "?")
    return f"{model} {decoder[:5]} {simp} c={c}"


def stratified_summary(label, fps, n_bootstrap):
    """Print stratified breakdown by GT canonical length."""
    short = [f for f in fps if 0 < f.canonical_gt_len <= 7]
    long = [f for f in fps if f.canonical_gt_len > 7]
    if not (short and long):
        return
    print(f"\n  --- {label}: stratified by GT canonical length ---")
    for stratum_label, subset in [
        (f"short (<=7) n={len(short)}", short),
        (f"long  (>7)  n={len(long)}", long),
    ]:
        agg = aggregate_fingerprint(subset, label=stratum_label, n_bootstrap=n_bootstrap)
        for axis in FIVE_AXES:
            m, lo, hi = getattr(agg, axis)
            print(f"    {stratum_label:<24}  {axis:<14} mean={m:>8.3f}  95%CI=[{lo:>7.3f}, {hi:>7.3f}]")


def paired_wilcoxon(rows_by_label, n_bootstrap=0):
    """Paired Wilcoxon signed-rank test on each fingerprint axis."""
    if len(rows_by_label) < 2:
        return
    keys = list(rows_by_label.keys())

    def to_dict(rows):
        return {(r.eq_id, r.sample_index): r for r in rows}

    dicts = {k: to_dict(v) for k, v in rows_by_label.items()}
    common_keys = set.intersection(*[set(d.keys()) for d in dicts.values()])
    if not common_keys:
        print("\n(no overlapping (eq_id, sample_index) pairs across dumps)")
        return

    print(f"\n{'=' * 110}")
    print(f"Paired-by-sample Wilcoxon comparison ({len(common_keys)} matched samples)")
    print(f"{'=' * 110}")

    if len(keys) == 2:
        try:
            from scipy import stats as sps
        except ImportError:
            print("(scipy not available; skipping Wilcoxon)")
            return

        a, b = keys
        for axis in FIVE_AXES + ("max_canonical_group_size", "fit_time", "n_canonical_unique"):
            xs = np.array([getattr(dicts[a][k], axis) for k in common_keys], dtype=float)
            ys = np.array([getattr(dicts[b][k], axis) for k in common_keys], dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if mask.sum() < 5 or np.array_equal(xs[mask], ys[mask]):
                continue
            try:
                stat, p = sps.wilcoxon(xs[mask], ys[mask])
            except ValueError:
                continue
            print(f"  {axis:<28} W={stat:>10.1f}  p={p:.3g}  "
                  f"median(a-b)={np.median(xs[mask] - ys[mask]):>+.3f}")

    for label in keys:
        top = sorted(rows_by_label[label], key=lambda r: -r.max_canonical_group_size)[:5]
        print(f"\n  Top-5 rewrite-cluster cases in [{label}]:")
        for r in top:
            print(f"    eq={r.eq_id:>10} idx={r.sample_index}  "
                  f"max_group={r.max_canonical_group_size:>4}  "
                  f"concentration={r.concentration*100:>5.1f}%  "
                  f"GT_canon_len={r.canonical_gt_len}")


def main() -> None:
    args = parse_args()
    engine = SimpliPyEngine.load("dev_7-3", install=False)

    rows_by_label = {}
    aggregates = []
    for dump in args.dumps:
        if not os.path.exists(dump):
            print(f"WARNING: {dump} does not exist, skipping.")
            continue
        with open(dump, "rb") as f:
            payload = pickle.load(f)
        label = label_for_payload(payload, dump)
        fps = compute_sample_fingerprints(engine, payload["samples"])
        rows_by_label[label] = fps
        agg = aggregate_fingerprint(fps, label=label, n_bootstrap=args.bootstrap)
        aggregates.append(agg)
        stratified_summary(label, fps, n_bootstrap=args.bootstrap)

    if aggregates:
        print(f"\n{'=' * 110}")
        print("Aggregate fingerprints (mean and 95% bootstrap CI)")
        print(f"{'=' * 110}")
        print(fingerprint_table(aggregates))

    paired_wilcoxon(rows_by_label, n_bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
