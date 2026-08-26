#!/usr/bin/env python
"""Build the v24 capability matrix from the per-arm row files, with provenance and statistics.

Replaces a hand-assembled `matrix_summary.json` that had no generator (`grep -rl matrix_summary`
over the harness returned nothing) and that mixed two different FVU denominators across its rows:
the baseline arms were scored in float64, the treatment arms in float32. On FastSRB targets
reaching |y| ~ 5e36 a float32 `np.var(y)` overflows to +inf, so `fvu = finite/inf = 0.0` passed the
float32-eps recovery bar as a FREE perfect fit -- in the direction that flattered the treatment.

Two things this refuses to do that the hand-built file did:

1. Quote an arm whose rows were scored against a denominator that could overflow, without either
   re-scoring it or explicitly excising the free rows. `--contaminated-set` names the affected
   problem indices; rows at those indices reporting exactly 0.0 are dropped and COUNTED, and the
   count is written into the output.
2. Publish an arm ordering with no uncertainty attached. Every arm carries a Wilson interval and
   every contrast against the baseline carries an exact McNemar test and a paired bootstrap CI.

Usage
-----
    python build_matrix.py --evals-dir <dir> [--out matrix_summary.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from srbf.metrics import mcnemar_exact, paired_difference_ci, wilson_interval

#: Problem indices whose float64 target variance is finite but whose FLOAT32 variance overflows to
#: +inf. Measured on fastsrb.v2, seed 20260825, the 110-problem realization: |y| from 6.06e19 to
#: 5.36e36. An arm scored in float32 books each of these as a free recovery.
FLOAT32_INF_VARIANCE_PROBLEMS = (2, 58, 60, 74, 78, 79, 80, 84, 85, 87, 94, 109)

#: The recovery bar, identical to srbf.metrics.is_perfect_fit.
F32_EPS = float(np.finfo(np.float32).eps)

#: arm label -> (row file, key within the file, fvu field). `key` indexes into a dict-of-lists file.
ARMS: dict[str, tuple[str, str | None, str]] = {
    "1. v23.0-3M baseline":                ("t16_vs_v23_fastsrb_rows.json", "v23.0-3M", "fvu"),
    "2. mask_all + random init":           ("cap1b_rows.json", None, "fvu"),
    "3. mask_fittable + random init":      ("cap6a_rows.json", None, "fvu"),
    "4. mask_all + infill init":           ("cap6_mask_all_infill_rows.json", None, "fvu"),
    "5. mask_fittable + infill init":      ("cap6_mask_fittable_infill_rows.json", None, "fvu"),
    "6. unflagged, as emitted":            ("cap1_rows.json", None, "fvu_emitted"),
    "7. compacted, as emitted":            ("cap6d_rows.json", None, "fvu_emitted"),
    "8. compacted + refine":               ("cap6d_rows.json", None, "fvu_refined"),
    "9. spans kept + refine (reference)":  ("cap1_rows.json", None, "fvu_refined"),
}

BASELINE = "1. v23.0-3M baseline"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_rows(path: Path, key: str | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if key is not None:
        if key not in data:
            raise KeyError(f"{path.name} has no arm '{key}' (has {sorted(data)})")
        data = data[key]
    if not isinstance(data, list):
        raise TypeError(f"{path.name} is not a row list")
    return data


def arm_outcomes(rows: list[dict[str, Any]], fvu_field: str,
                 contaminated: tuple[int, ...]) -> tuple[dict[int, bool], int]:
    """Per-problem recovery flags, with free (float32-overflow) recoveries excised.

    Returns ``(outcomes_by_problem_index, n_free_rows_dropped)``. A row is 'free' when it sits at a
    known inf-variance problem AND reports an FVU of exactly 0.0 -- genuine near-perfect fits bottom
    out around 1e-15 and are never exactly zero, so the test is specific.
    """
    outcomes: dict[int, bool] = {}
    n_free = 0
    missing = [row for row in rows if fvu_field not in row]
    if missing:
        # Never default a missing metric to +inf: it scores the arm as a silent, plausible-looking
        # ZERO. Caught during this script's own bring-up -- `fvu` does not exist in cap1_rows.json
        # (it is `fvu_emitted` / `fvu_refined`), and the first run reported two arms at 0.0%.
        raise KeyError(
            f"{len(missing)} of {len(rows)} rows have no '{fvu_field}' field "
            f"(available: {sorted(rows[0]) if rows else 'no rows'})")
    for row in rows:
        index = int(row["i"])
        fvu = float(row[fvu_field])
        if index in contaminated and fvu == 0.0:
            n_free += 1
            continue
        outcomes[index] = bool(row.get("status") == "ok" and fvu <= F32_EPS)
    return outcomes, n_free


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--contaminated-set", type=str,
                        default=",".join(str(i) for i in FLOAT32_INF_VARIANCE_PROBLEMS),
                        help="Problem indices whose float32 variance overflows; '' to disable.")
    args = parser.parse_args()

    contaminated = tuple(int(x) for x in args.contaminated_set.split(",") if x.strip())

    arms: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[int, bool]] = {}
    for label, (filename, key, fvu_field) in ARMS.items():
        path = args.evals_dir / filename
        if not path.exists():
            print(f"  SKIP {label}: {filename} not found", file=sys.stderr)
            continue
        rows = load_rows(path, key)
        arm_outcome, n_free = arm_outcomes(rows, fvu_field, contaminated)
        outcomes[label] = arm_outcome

        n_scored = len(arm_outcome)
        n_recovered = sum(arm_outcome.values())
        lo, hi = wilson_interval(n_recovered, n_scored)
        arms[label] = {
            "source": filename,
            "source_sha256": sha256(path),
            "fvu_field": fvu_field,
            "n_problems_scored": n_scored,
            "n_recovered": n_recovered,
            "fNRR_percent": round(100.0 * n_recovered / n_scored, 2) if n_scored else None,
            "fNRR_ci95_percent": [round(100 * lo, 2), round(100 * hi, 2)],
            "free_rows_excised": n_free,
        }

    def contrast(label_a: str, label_b: str) -> dict[str, Any] | None:
        """Paired contrast restricted to the problems BOTH arms scored validly."""
        arm_a, arm_b = outcomes[label_a], outcomes[label_b]
        shared = sorted(set(arm_a) & set(arm_b))
        if not shared:
            return None
        a = np.array([arm_a[i] for i in shared])
        b = np.array([arm_b[i] for i in shared])
        result = mcnemar_exact(a, b)
        point, lo, hi = paired_difference_ci(a, b)
        return {
            "n_paired_problems": len(shared),
            "a_only": result.n_a_only,
            "b_only": result.n_b_only,
            "both": result.n_both,
            "neither": result.n_neither,
            "mcnemar_exact_p": round(result.p_value, 6),
            "difference_percent": round(100 * point, 2),
            "difference_ci95_percent": [round(100 * lo, 2), round(100 * hi, 2)],
            "significant_at_0.05": bool(result.p_value < 0.05),
        }

    # Every arm pair, not only vs-baseline: the questions this matrix exists to answer ("does
    # infilling help?", "does compaction cost recoveries?") are arm-vs-arm. No multiplicity
    # correction is applied -- with 36 pairs at alpha=0.05 roughly two will be spurious, so a
    # single isolated p just under 0.05 here is not evidence on its own.
    labels = list(outcomes)
    pairwise: dict[str, Any] = {}
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            result_pair = contrast(label_a, label_b)
            if result_pair is not None:
                pairwise[f"{label_a} vs {label_b}"] = result_pair

    contrasts: dict[str, Any] = {}
    if BASELINE in outcomes:
        base = outcomes[BASELINE]
        for label, arm in outcomes.items():
            if label == BASELINE:
                continue
            shared = sorted(set(base) & set(arm))
            if not shared:
                continue
            a = np.array([base[i] for i in shared])
            b = np.array([arm[i] for i in shared])
            result = mcnemar_exact(a, b)
            point, lo, hi = paired_difference_ci(a, b)
            contrasts[f"{BASELINE} vs {label}"] = {
                "n_paired_problems": len(shared),
                "baseline_only": result.n_a_only,
                "arm_only": result.n_b_only,
                "both": result.n_both,
                "neither": result.n_neither,
                "mcnemar_exact_p": round(result.p_value, 6),
                "difference_percent": round(100 * point, 2),
                "difference_ci95_percent": [round(100 * lo, 2), round(100 * hi, 2)],
                "significant_at_0.05": bool(result.p_value < 0.05),
            }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "experiments/v24_capability_matrix/build_matrix.py",
        "flash_ansr_commit": git_commit(Path(__file__).resolve().parents[2]),
        "recovery_bar": "fvu <= float32 eps (srbf.metrics.is_perfect_fit)",
        "contaminated_problem_indices": list(contaminated),
        "contamination_note": (
            "Rows at these indices reporting fvu == 0.0 exactly were produced by a float32 variance "
            "that overflowed to +inf and are EXCISED, not counted as recoveries. Arms re-scored in "
            "float64 have no such rows and are unaffected."),
        "multiplicity_note": (
            "pairwise_contrasts holds every arm pair with no multiplicity correction; at 36 pairs "
            "and alpha=0.05 about two false positives are expected, so an isolated p just under "
            ".05 there is not evidence on its own."),
        "arms": arms,
        "contrasts_vs_baseline": contrasts,
        "pairwise_contrasts": pairwise,
    }

    out = args.out or (args.evals_dir / "matrix_summary.json")
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["arms"], indent=1))
    print(json.dumps(summary["contrasts_vs_baseline"], indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
