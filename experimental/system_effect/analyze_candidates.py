"""Analyze a candidate-distribution dump produced by ``candidate_distribution_probe.py``.

For each probed test point, report:
- raw-form vs canonical-form distinct candidates,
- mean bloat factor (raw token length / canonical token length),
- canonical-group sizes (rewrite cluster size),
- hypothesis count under FVU thresholds,
- whether the GT canonical hypothesis appears in the candidate set,
- aggregate fit time and a worked example of the largest rewrite cluster.

Multiple dumps can be passed; each is summarized side-by-side and a paired
per-sample comparison is printed when at least two share their eq_ids.

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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from simplipy import SimpliPyEngine  # noqa: E402

EPS32 = float(np.finfo(np.float32).eps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dumps", nargs="+", help="One or more candidate-distribution pickle dumps.")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples for CI estimation (default: 1000).")
    return p.parse_args()


def normalize_skeleton(expr_tokens):
    """Replace numeric literals with the ``<constant>`` placeholder."""
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


def safe_simplify(engine, skel):
    if not skel:
        return tuple(skel) if skel else tuple()
    try:
        out = engine.simplify(list(skel), max_pattern_length=4)
        return tuple(out) if engine.is_valid(out) else tuple(skel)
    except Exception:
        return tuple(skel)


def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(a, size=a.size, replace=True).mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(a.mean()), float(lo), float(hi)


def per_sample_summary(engine, payload):
    samples = payload["samples"]
    label = f"{Path(payload['model_path']).name} simp={payload['simplify']} c={payload['choices']}"

    rows = []
    for s in samples:
        cands = s["candidates"]
        if not cands:
            continue

        raw_skels = [normalize_skeleton(c["expression"]) for c in cands]
        canonical_skels = [safe_simplify(engine, rs) for rs in raw_skels]

        n = len(cands)
        n_raw_unique = len(set(raw_skels))
        n_canonical_unique = len(set(canonical_skels))

        bloats = [len(rs) / max(len(cs), 1) for rs, cs in zip(raw_skels, canonical_skels) if len(cs) > 0]
        mean_bloat = float(np.mean(bloats)) if bloats else float("nan")

        groups = defaultdict(list)
        for cs, c in zip(canonical_skels, cands):
            groups[cs].append(c)
        group_sizes = [len(g) for g in groups.values()]
        max_group = max(group_sizes) if group_sizes else 0

        seen = set()
        rewrites = 0
        for cs in canonical_skels:
            if cs in seen:
                rewrites += 1
            else:
                seen.add(cs)

        groups_with_perfect = sum(
            1 for g in groups.values()
            if any(np.isfinite(c["fvu"]) and c["fvu"] <= EPS32 for c in g)
        )
        groups_with_good = sum(
            1 for g in groups.values()
            if any(np.isfinite(c["fvu"]) and c["fvu"] <= 1e-3 for c in g)
        )

        gt_canonical = tuple(s["skeleton_canonical_gt"]) if s["skeleton_canonical_gt"] else None
        gt_in_canonical_set = bool(gt_canonical) and (gt_canonical in groups)

        rows.append({
            "eq_id": s["eq_id"],
            "sample_index": s.get("sample_index", -1),
            "n_results": n,
            "n_raw_unique": n_raw_unique,
            "n_canonical_unique": n_canonical_unique,
            "rewrite_fraction": rewrites / n,
            "mean_bloat": mean_bloat,
            "max_group_size": max_group,
            "groups_with_perfect": groups_with_perfect,
            "groups_with_good": groups_with_good,
            "gt_in_canonical": int(gt_in_canonical_set),
            "fit_time": s["fit_time"],
            "canonical_gt_len": len(gt_canonical) if gt_canonical else 0,
        })
    return label, rows


def report(label, rows, n_bootstrap):
    print(f"\n{'=' * 110}")
    print(f"{label}   (n_samples={len(rows)})")
    print(f"{'=' * 110}")
    metrics = [
        ("n_results", "refined candidates"),
        ("n_canonical_unique", "canonical-unique hypotheses"),
        ("rewrite_fraction", "rewrite fraction"),
        ("mean_bloat", "mean bloat (raw / canonical)"),
        ("max_group_size", "largest canonical group"),
        ("groups_with_perfect", "hypotheses with FVU <= eps32"),
        ("groups_with_good", "hypotheses with FVU <= 1e-3"),
        ("gt_in_canonical", "GT canonical present (frac)"),
        ("fit_time", "fit_time per sample (s)"),
    ]
    for key, label_str in metrics:
        vals = [r[key] for r in rows]
        m, lo, hi = bootstrap_ci(vals, n_resamples=n_bootstrap)
        med = float(np.median(np.asarray(vals, dtype=float)))
        print(f"  {label_str:<32} mean={m:>9.3f}  95%CI=[{lo:>8.3f}, {hi:>8.3f}]   median={med:>8.3f}")

    # Stratified by canonical GT length
    short = [r for r in rows if r["canonical_gt_len"] > 0 and r["canonical_gt_len"] <= 7]
    long = [r for r in rows if r["canonical_gt_len"] > 7]
    if short and long:
        print(f"\n  --- Stratified by GT canonical length ---")
        for stratum_label, subset in [(f"short (<=7) [n={len(short)}]", short),
                                       (f"long  (>7)  [n={len(long)}]", long)]:
            for key in ("rewrite_fraction", "mean_bloat", "max_group_size"):
                m, lo, hi = bootstrap_ci([r[key] for r in subset], n_resamples=n_bootstrap)
                print(f"    {stratum_label:<22} {key:<22} mean={m:>9.3f}  95%CI=[{lo:>8.3f}, {hi:>8.3f}]")


def paired_comparison(rows_by_label):
    """Print per-(eq_id, sample_index) S/U comparison for matched dumps."""
    if len(rows_by_label) < 2:
        return
    keys = list(rows_by_label.keys())

    def to_dict(rows):
        return {(r["eq_id"], r["sample_index"]): r for r in rows}

    dicts = {k: to_dict(v) for k, v in rows_by_label.items()}
    common_keys = set.intersection(*[set(d.keys()) for d in dicts.values()])
    if not common_keys:
        return

    print(f"\n{'=' * 110}")
    print(f"Paired per-sample comparison ({len(common_keys)} matched samples)")
    print(f"{'=' * 110}")

    # paired Wilcoxon on the 2-pipeline case
    if len(keys) == 2:
        from scipy import stats as sps
        a, b = keys
        for metric in ("rewrite_fraction", "mean_bloat", "max_group_size", "groups_with_perfect", "fit_time"):
            xs = np.array([dicts[a][k][metric] for k in common_keys], dtype=float)
            ys = np.array([dicts[b][k][metric] for k in common_keys], dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if mask.sum() < 5:
                continue
            try:
                stat, p = sps.wilcoxon(xs[mask], ys[mask])
            except ValueError:
                continue
            print(f"  Wilcoxon {metric:<22}: {a} vs {b}  W={stat:>8.1f}  p={p:.3g}  "
                  f"(median diff = {np.median(xs[mask]-ys[mask]):>+.3f})")

    # Top rewrite-cluster cases for each label
    for label in keys:
        top = sorted(rows_by_label[label], key=lambda r: -r["max_group_size"])[:5]
        print(f"\n  Top-5 rewrite-cluster cases in [{label}]:")
        for r in top:
            print(f"    eq={r['eq_id']:>10} idx={r['sample_index']}  "
                  f"max_group={r['max_group_size']:>4}  rewrite_frac={r['rewrite_fraction']*100:>5.1f}%  "
                  f"GT_canon_len={r['canonical_gt_len']}")


def main() -> None:
    args = parse_args()
    engine = SimpliPyEngine.load("dev_7-3", install=False)

    rows_by_label = {}
    for dump in args.dumps:
        if not os.path.exists(dump):
            print(f"WARNING: {dump} does not exist, skipping.")
            continue
        with open(dump, "rb") as f:
            payload = pickle.load(f)
        label, rows = per_sample_summary(engine, payload)
        rows_by_label[label] = rows
        report(label, rows, args.bootstrap)

    paired_comparison(rows_by_label)


if __name__ == "__main__":
    main()
