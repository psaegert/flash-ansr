"""Render hypothesis-space coverage fingerprints as parallel-coordinates or radar plots.

Used by §8.1 (training-data axis) and §8.3 (inference-strategy axis) for the
matched-axis comparison figures, and by App N for full-fingerprint reports.

Usage::

    python experimental/system_effect/fingerprint_plot.py \\
        --input "S100 c=2048=results/system_effect/cand_dist_v23.0-20M-A-S100_choices02048_n300.pkl" \\
        --input "U100 c=2048=results/system_effect/cand_dist_v23.0-20M-A-U100_choices02048_n300.pkl" \\
        --output results/figures/system_effect/fingerprint_a_track_c2048.svg \\
        --style parallel
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simplipy import SimpliPyEngine  # noqa: E402

from diagnostic import (  # noqa: E402
    DIMENSION_DIRECTION,
    FIVE_AXES,
    aggregate_fingerprint,
    compute_sample_fingerprints,
)


# Per-axis y-axis bounds for the parallel-coordinates view. The fingerprints
# have different natural scales; we let each axis breathe independently.
AXIS_LIMITS = {
    "coverage":     (0.0, 1.0),
    "quality":      (0.0, 1.0),
    "concentration":(0.0, 1.0),
    "surface_form": (1.0, 3.0),
    "targeting":    (0.0, 1.0),
}

AXIS_LABELS = {
    "coverage":     "Coverage\n(canonical-unique frac)",
    "quality":      "Quality\n(perfect-fit yield)",
    "concentration":"Concentration\n(rewrite frac)",
    "surface_form": "Surface form\n(mean bloat)",
    "targeting":    "Targeting\n(GT canonical present)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True,
                   help="LABEL=path/to/dump.pkl (repeat for multiple configurations).")
    p.add_argument("--output", required=True, help="Output figure path (svg or png).")
    p.add_argument("--style", default="parallel", choices=["parallel", "radar"])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--title", default=None)
    return p.parse_args()


def render_parallel(aggregates, output_path, title=None):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=200)
    n_axes = len(FIVE_AXES)
    x_positions = np.arange(n_axes)

    cmap = plt.get_cmap("tab10")

    for k, agg in enumerate(aggregates):
        means = []
        los = []
        his = []
        for axis in FIVE_AXES:
            m, lo, hi = getattr(agg, axis)
            means.append(m)
            los.append(lo)
            his.append(hi)
        ax.errorbar(
            x_positions,
            means,
            yerr=[np.asarray(means) - np.asarray(los), np.asarray(his) - np.asarray(means)],
            fmt="o-",
            color=cmap(k),
            label=agg.label,
            capsize=2,
            lw=1.4,
            ms=5,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([AXIS_LABELS[a] for a in FIVE_AXES], fontsize=8)
    ax.set_ylim(0.0, 3.0)
    ax.set_ylabel("Value (per-axis natural scale)", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


def render_radar(aggregates, output_path, title=None):
    """Radar plot. Each axis is normalized to [0, 1] using AXIS_LIMITS, with
    sign flipped if the dimension is "lower is better" (concentration,
    surface_form) so that "outward" always means "better."
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=200, subplot_kw={"projection": "polar"})

    angles = np.linspace(0, 2 * np.pi, len(FIVE_AXES), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    cmap = plt.get_cmap("tab10")

    for k, agg in enumerate(aggregates):
        normalized = []
        for axis in FIVE_AXES:
            m, _, _ = getattr(agg, axis)
            lo_lim, hi_lim = AXIS_LIMITS[axis]
            v = (m - lo_lim) / (hi_lim - lo_lim) if hi_lim > lo_lim else 0.0
            v = max(0.0, min(1.0, v))
            if DIMENSION_DIRECTION[axis] == -1:
                v = 1.0 - v  # flip so larger = better
            normalized.append(v)
        normalized += normalized[:1]
        ax.plot(angles, normalized, color=cmap(k), label=agg.label, lw=1.5)
        ax.fill(angles, normalized, color=cmap(k), alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([AXIS_LABELS[a] for a in FIVE_AXES], fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, pad=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()

    parsed = []
    for spec in args.input:
        if "=" not in spec:
            sys.exit(f"--input expects LABEL=path, got {spec!r}")
        label, dump = spec.split("=", 1)
        parsed.append((label.strip(), dump.strip()))

    engine = SimpliPyEngine.load("dev_7-3", install=False)
    aggregates = []
    for label, dump in parsed:
        with open(dump, "rb") as f:
            payload = pickle.load(f)
        fps = compute_sample_fingerprints(engine, payload["samples"])
        aggregates.append(aggregate_fingerprint(fps, label=label, n_bootstrap=args.bootstrap))

    if args.style == "parallel":
        render_parallel(aggregates, args.output, title=args.title)
    elif args.style == "radar":
        render_radar(aggregates, args.output, title=args.title)


if __name__ == "__main__":
    main()
