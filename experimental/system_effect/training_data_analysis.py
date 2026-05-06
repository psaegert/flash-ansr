"""Compare training-data distributions under simplify=true vs simplify=false (and SymPy).

For each pool config, sample N skeletons, then report:
- Length distribution (raw and after canonicalization).
- Per-skeleton compression ratio (canonical / raw).
- Distinct canonical-form count (hypothesis diversity).
- Operator-frequency drift across pools.
- Per-skeleton sampling cost.

Usage::

    python experimental/system_effect/training_data_analysis.py \\
        --n-samples 5000 \\
        --pools S=configs/v23.0-20M-A-S1/skeleton_pool_train.yaml \\
        --pools U=configs/v23.0-20M-A-U1/skeleton_pool_train.yaml \\
        --output results/system_effect/training_data_summary.pkl

Each ``--pools`` entry is ``LABEL=path/to/skeleton_pool_*.yaml``. Run with
``--n-samples 50000`` for camera-ready statistics.
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flash_ansr.expressions.skeleton_pool import SkeletonPool  # noqa: E402
from simplipy import SimpliPyEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=5000,
                   help="Number of skeletons to sample per pool (default: 5000).")
    p.add_argument("--pools", action="append", required=True,
                   help="LABEL=path/to/skeleton_pool_*.yaml (repeat for multiple pools).")
    p.add_argument("--output", default=None,
                   help="Optional pickle path to dump the raw skeletons + statistics for later figures.")
    return p.parse_args()


def safe_simplify(engine, skel):
    if skel is None:
        return None
    try:
        out = engine.simplify(list(skel), max_pattern_length=4)
        return tuple(out) if engine.is_valid(out) else tuple(skel)
    except Exception:
        return tuple(skel)


def operator_counts(skel):
    return collections.Counter(t for t in skel if not (t.startswith("x") or t == "<constant>"))


def build_pool(cfg_path, label, n):
    print(f"\n--- Building pool: {label} ({cfg_path}) ---")
    t0 = time.time()
    pool = SkeletonPool.from_config(cfg_path)
    pool.create(size=n, verbose=False)
    elapsed = time.time() - t0
    print(f"  Built {len(pool.skeletons)} skeletons in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f}/s)")
    return list(pool.skeletons), elapsed


def summarize(engine, label, raw_skels, build_time):
    raw_lens = np.array([len(s) for s in raw_skels])
    canonical = [safe_simplify(engine, s) for s in raw_skels]
    canonical_lens = np.array([len(c) for c in canonical])
    distinct_canonical = len(set(canonical))
    reduction_ratio = canonical_lens / np.maximum(raw_lens, 1)

    print(f"\n=== {label} (n={len(raw_skels)}) ===")
    print(f"  Build time: {build_time:.1f}s ({build_time * 1000 / max(len(raw_skels), 1):.2f} ms/skeleton)")
    print(f"  Raw length:        mean={raw_lens.mean():.2f}  median={np.median(raw_lens):.0f}  "
          f"p90={np.percentile(raw_lens, 90):.0f}  max={raw_lens.max()}")
    print(f"  Canonical length:  mean={canonical_lens.mean():.2f}  median={np.median(canonical_lens):.0f}  "
          f"p90={np.percentile(canonical_lens, 90):.0f}")
    print(f"  Reduction ratio:   mean={reduction_ratio.mean():.3f}  median={np.median(reduction_ratio):.3f}  "
          f"frac_unchanged={np.mean(reduction_ratio == 1.0):.3f}")
    print(f"  Distinct canonical forms: {distinct_canonical} / {len(raw_skels)} = "
          f"{distinct_canonical / len(raw_skels):.4f}")

    return {
        "label": label,
        "raw_lens": raw_lens.tolist(),
        "canonical_lens": canonical_lens.tolist(),
        "canonical_forms": canonical,
        "build_time": build_time,
        "n_distinct_canonical": distinct_canonical,
    }


def main() -> None:
    args = parse_args()

    engine = SimpliPyEngine.load("dev_7-3", install=False)
    pools = []
    for spec in args.pools:
        if "=" not in spec:
            sys.exit(f"--pools expects LABEL=path, got {spec!r}")
        label, cfg = spec.split("=", 1)
        pools.append((label.strip(), cfg.strip()))

    summaries = []
    raw_data = {}
    for label, cfg in pools:
        raw, build_time = build_pool(cfg, label, args.n_samples)
        summaries.append(summarize(engine, label, raw, build_time))
        raw_data[label] = raw

    print(f"\n=== Operator frequency drift (per-token, normalized) ===")
    op_counters = {}
    for s in summaries:
        cnt = collections.Counter()
        for sk in raw_data[s["label"]]:
            cnt.update(operator_counts(sk))
        op_counters[s["label"]] = cnt

    all_ops = sorted(set().union(*[c.keys() for c in op_counters.values()]))
    headers = "  " + "{:>10}".format("op") + "".join("{:>14}".format(s["label"]) for s in summaries)
    print(headers)
    drift_rows = []
    for op in all_ops:
        row = [op]
        for s in summaries:
            total = sum(op_counters[s["label"]].values()) or 1
            row.append(op_counters[s["label"]][op] / total)
        drift_rows.append(row)

    base_label = summaries[0]["label"]
    drift_rows.sort(key=lambda r: -max(
        abs(np.log10(max(r[i], 1e-9) / max(r[1], 1e-9)))
        for i in range(2, len(summaries) + 1)
    ) if len(summaries) > 1 else -r[1])

    for row in drift_rows[:25]:
        op = row[0]
        line = "  " + "{:>10}".format(op) + "".join("{:>14.4f}".format(v) for v in row[1:])
        print(line)
    print(f"  (showing top 25 of {len(drift_rows)} operators by max log-ratio vs {base_label})")

    print(f"\n=== Length distribution (raw, fraction) ===")
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 35)]
    print("  " + "{:>10}".format("bucket") + "".join("{:>14}".format(s["label"]) for s in summaries))
    for lo, hi in buckets:
        line = "  " + "[{:>2},{:>3})".format(lo, hi)
        for s in summaries:
            lens = np.asarray(s["raw_lens"])
            line += "{:>14.4f}".format(np.mean((lens >= lo) & (lens < hi)))
        print(line)

    print(f"\n=== Cross-pool canonical-form overlap ===")
    canon_sets = {s["label"]: set(s["canonical_forms"]) for s in summaries}
    labels = list(canon_sets.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            inter = canon_sets[a] & canon_sets[b]
            print(f"  {a} ∩ {b}: {len(inter)}  "
                  f"({len(inter) / max(len(canon_sets[a]), 1) * 100:.2f}% of {a}, "
                  f"{len(inter) / max(len(canon_sets[b]), 1) * 100:.2f}% of {b})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        # Don't save canonical_forms (large); save lens + counters + meta only.
        for s in summaries:
            s.pop("canonical_forms", None)
        payload = {
            "summaries": summaries,
            "operator_counts": {k: dict(v) for k, v in op_counters.items()},
            "n_samples": args.n_samples,
        }
        with open(args.output, "wb") as f:
            pickle.dump(payload, f)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
