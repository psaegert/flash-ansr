"""Probe full candidate distribution for a given model + test-time pipeline.

For each sampled fastsrb test point, load the same X/y the original eval used,
run model.fit, and dump every candidate that survived refinement (the full
``model._results`` list). The dump is meant to be analyzed offline by
``analyze_candidates.py``.

Usage::

    python experimental/system_effect/candidate_distribution_probe.py \\
        --model-path models/ansr-models/v23.0-20M-A-S100 \\
        --simplify true \\
        --choices 2048 \\
        --n-samples 300 \\
        --source-pkl results/evaluation/scaling/v23.0-20M-A-S100/fastsrb/choices_00256.pkl \\
        --output results/system_effect/cand_dist_v23.0-20M-A-S100_choices02048_n300.pkl

The ``--source-pkl`` argument selects a previously-evaluated pickle to source
the (X, y) test points from. Any of the existing scaling result pickles works;
we keep choices=256 by default because every model has it.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from flash_ansr import FlashANSR  # noqa: E402
from flash_ansr.utils.generation import create_generation_config  # noqa: E402
from simplipy import SimpliPyEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True,
                   help="Path to a FlashANSR checkpoint dir (e.g. models/ansr-models/v23.0-20M-A-S100).")
    p.add_argument("--simplify", required=True, choices=["true", "false", "sympy"],
                   help="Test-time simplify flag for the generation config. "
                        "Use 'sympy' to match a SymPy-trained model (slow at large c but "
                        "required for matched train/inference comparison).")
    p.add_argument("--choices", type=int, required=True,
                   help="Number of candidates per sample. For decoder=softmax_sampling this is "
                        "the `choices` parameter; for decoder=beam_search it is `beam_width`.")
    p.add_argument("--decoder", default="softmax_sampling",
                   choices=["softmax_sampling", "beam_search"],
                   help="Decoding strategy (default: softmax_sampling).")
    p.add_argument("--n-samples", type=int, default=300,
                   help="Number of fastsrb samples to probe (stratified subsample of --source-pkl).")
    p.add_argument("--source-pkl", required=True,
                   help="An existing fastsrb eval pickle; only X/y/skeleton/eq_id/sample_index are read.")
    p.add_argument("--output", required=True,
                   help="Output pickle path for the candidate dump.")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda).")
    p.add_argument("--save-every", type=int, default=16,
                   help="Persist partial output every N completed samples (crash-safe; default 16). "
                        "Set to 0 to disable incremental saving.")
    return p.parse_args()


def stratified_indices(eq_ids, n_target):
    """Pick approximately n_target indices, evenly across distinct eq_ids."""
    if n_target >= len(eq_ids):
        return list(range(len(eq_ids)))
    stride = max(1, len(eq_ids) // n_target)
    return list(range(0, len(eq_ids), stride))[:n_target]


def _save_payload(samples, output_path, *, model_path, decoder, simplify, choices, n_samples, wall_time):
    """Atomic write of the probe payload (write to tmp, then os.replace)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "samples": samples,
        "model_path": model_path,
        "decoder": decoder,
        "simplify": simplify,
        "choices": choices,
        "n_samples": n_samples,
        "n_collected": len(samples),
        "wall_time": wall_time,
    }
    tmp = output_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, output_path)


def _load_resume_state(output_path, *, model_path, decoder, simplify, choices):
    """Load existing partial output if present and compatible.

    Returns (samples_list, already_done_idx_set, prev_wall_time). On any
    incompatibility (different model_path / decoder / simplify / choices),
    raises SystemExit so we never silently overwrite a probe of a different
    configuration.
    """
    if not os.path.exists(output_path):
        return [], set(), 0.0
    try:
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
    except Exception as exc:
        print(f"WARNING: could not load existing {output_path} ({exc}); starting fresh.")
        return [], set(), 0.0

    if not isinstance(existing, dict) or "samples" not in existing:
        print(f"WARNING: existing {output_path} is not a probe payload; starting fresh.")
        return [], set(), 0.0

    # Refuse to mix probes with different configurations.
    mismatches = []
    if existing.get("model_path") != model_path:
        mismatches.append(f"model_path ({existing.get('model_path')!r} vs {model_path!r})")
    if existing.get("decoder", "softmax_sampling") != decoder:
        mismatches.append(f"decoder ({existing.get('decoder')!r} vs {decoder!r})")
    if existing.get("simplify") != simplify:
        mismatches.append(f"simplify ({existing.get('simplify')!r} vs {simplify!r})")
    if existing.get("choices") != choices:
        mismatches.append(f"choices ({existing.get('choices')!r} vs {choices!r})")
    if mismatches:
        print(f"ERROR: existing {output_path} has different configuration: {', '.join(mismatches)}.")
        print("Move it aside or delete it before re-running with a different configuration.")
        sys.exit(1)

    samples = existing.get("samples", [])
    done = {s.get("idx") for s in samples if "idx" in s}
    prev_wall = float(existing.get("wall_time", 0.0))
    print(f"Resuming from {len(samples)} existing samples in {output_path} (prev wall_time {prev_wall:.1f}s).")
    return samples, done, prev_wall


def main() -> None:
    args = parse_args()

    if args.simplify == "true":
        simplify: bool | str = True
    elif args.simplify == "false":
        simplify = False
    else:
        # 'sympy': match a SymPy-trained model at inference. Generation config
        # accepts the literal string and dispatches to SymPy in the model's
        # post-generation simplification step.
        simplify = "sympy"

    print(f"Loading source samples from {args.source_pkl}")
    with open(args.source_pkl, "rb") as f:
        src = pickle.load(f)

    selected = stratified_indices(src["benchmark_eq_id"], args.n_samples)
    print(f"Selected {len(selected)} samples (target {args.n_samples}) "
          f"covering {len(set(src['benchmark_eq_id'][i] for i in selected))} unique eq_ids.")

    # SimpliPy engine for canonicalization of the GT skeleton (saved alongside each sample).
    engine = SimpliPyEngine.load("dev_7-3", install=False)

    def safe_simplify(skel):
        if not skel:
            return tuple(skel) if skel is not None else None
        try:
            out = engine.simplify(list(skel), max_pattern_length=4)
            return tuple(out) if engine.is_valid(out) else tuple(skel)
        except Exception:
            return tuple(skel)

    if args.decoder == "softmax_sampling":
        gen_config = create_generation_config(
            method="softmax_sampling",
            choices=args.choices,
            top_k=0,
            top_p=1.0,
            max_len=64,
            batch_size=128,
            temperature=1.0,
            valid_only=True,
            simplify=simplify,
            unique=True,
        )
    elif args.decoder == "beam_search":
        gen_config = create_generation_config(
            method="beam_search",
            beam_width=args.choices,
            max_len=64,
            batch_size=128,
            unique=True,
        )
    else:
        raise ValueError(f"Unsupported decoder: {args.decoder}")

    print(f"\nLoading model {args.model_path} "
          f"(decoder={args.decoder}, simplify={simplify}, choices/beam_width={args.choices})")
    model = FlashANSR.load(
        args.model_path,
        generation_config=gen_config,
        n_restarts=8,
        refiner_method="curve_fit_lm",
        refiner_p0_noise="normal",
        refiner_p0_noise_kwargs={"loc": 0.0, "scale": 5.0},
        length_penalty=0.05,
        constants_penalty=0.0,
        likelihood_penalty=0.0,
        device=args.device,
        prune_constant_budget=0,
    ).to(args.device).eval()

    # Resume existing partial output if it matches this configuration.
    out, already_done, prev_wall_time = _load_resume_state(
        args.output,
        model_path=args.model_path,
        decoder=args.decoder,
        simplify=simplify,
        choices=args.choices,
    )

    save_kwargs = dict(
        model_path=args.model_path,
        decoder=args.decoder,
        simplify=simplify,
        choices=args.choices,
        n_samples=args.n_samples,
    )
    save_every = max(0, int(args.save_every))
    n_skipped = 0
    n_session = 0
    t_total_start = time.time()

    for k, idx in enumerate(selected):
        if idx in already_done:
            n_skipped += 1
            continue

        X = np.asarray(src["x"][idx])
        y = np.asarray(src["y"][idx])
        skel_gt = src["skeleton"][idx]
        canonical_gt = safe_simplify(skel_gt)

        t0 = time.time()
        try:
            model.fit(X, y, verbose=False)
            elapsed = time.time() - t0
        except Exception as exc:
            print(f"  [{len(out)+n_skipped+1}/{len(selected)}] idx={idx} eq={src['benchmark_eq_id'][idx]} FAIL: {exc}")
            continue

        results = model._results
        rec = {
            "idx": idx,
            "eq_id": src["benchmark_eq_id"][idx],
            "sample_index": int(src["benchmark_sample_index"][idx]),
            "skeleton_raw_gt": list(skel_gt) if skel_gt is not None else None,
            "skeleton_canonical_gt": list(canonical_gt) if canonical_gt is not None else None,
            "fit_time": elapsed,
            "n_results": len(results),
            "candidates": [
                {
                    "expression": list(r.get("expression", [])),
                    "fvu": float(r.get("fvu", float("nan"))),
                    "log_prob": float(r.get("log_prob"))
                    if r.get("log_prob") is not None and np.isfinite(r.get("log_prob", float("nan")))
                    else float("nan"),
                    "score": float(r.get("score", float("nan"))),
                    "constant_count": int(r.get("constant_count", 0)),
                    "pruned_variant": bool(r.get("pruned_variant", False)),
                }
                for r in results
            ],
        }
        out.append(rec)
        already_done.add(idx)
        n_session += 1

        best_fvu = rec["candidates"][0]["fvu"] if rec["candidates"] else float("nan")
        print(f"  [{len(out)+n_skipped}/{len(selected)}] idx={idx:>4} eq={src['benchmark_eq_id'][idx]:>10} "
              f"time={elapsed:6.2f}s n_results={len(results):>5} best_fvu={best_fvu:.4g}")

        # Crash-safe incremental save.
        if save_every > 0 and n_session % save_every == 0:
            cumulative_wall = prev_wall_time + (time.time() - t_total_start)
            _save_payload(out, args.output, wall_time=cumulative_wall, **save_kwargs)

    cumulative_wall = prev_wall_time + (time.time() - t_total_start)
    _save_payload(out, args.output, wall_time=cumulative_wall, **save_kwargs)
    print(f"\nWall time (session): {time.time() - t_total_start:.1f}s   "
          f"Cumulative: {cumulative_wall:.1f}s   "
          f"Collected: {len(out)} / {len(selected)} ({n_session} new this session, {n_skipped} resumed)")
    print(f"Saved candidate dump to {args.output}")


if __name__ == "__main__":
    main()
