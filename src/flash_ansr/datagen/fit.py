"""Stage 2 (CPU nodes): fit the raw draws of each problem with the unchanged deployed pipeline.

Per problem: the deployed post-processing (extract, simplify, deduplicate; :meth:`FlashANSRModel._postprocess_sampled`,
the same call the chunked sampler makes), then :meth:`FlashANSR._fit_refine` (refit with the configured restarts,
constant ladder) and :meth:`FlashANSR._build_result` (two-part ranking). Every fitted ledger row is scored on the
problem's clean validation rows (fresh-point FVU). Rows whose fresh FVU is not below 1 are dropped: the constant
model dominates them, so they can never be front members. Each kept row records the token sequence of its family's
root as the ledger holds it, i.e. AFTER the deployed post-processing (simplified, deduplicated, re-encoded), not the
raw emission; ladder variants share their root's sequence. The raw draws stay in the raw shard (no raw-draw -> family
map is stored yet).

One estimator per worker process with a serial refiner (``ComputeConfig(workers=0)``): parallelism is across
problems, never nested.
"""
from __future__ import annotations

import glob
import os
import time
import zlib
from typing import Any

import numpy as np

from flash_ansr.datagen import shards

_W: dict[str, Any] = {}


def _init_worker(model_dir: str, n_restarts: int | None) -> None:
    import torch
    torch.set_num_threads(1)
    from flash_ansr import FlashANSR
    refine = {"n_restarts": int(n_restarts)} if n_restarts is not None else None
    w = FlashANSR.load(directory=model_dir, refine=refine, compute={"device": "cpu", "workers": 0})
    _W.update(w=w, pp=w._prepare_prompt_prefix(emission="fittable", complexity=None))


def _fresh_fvu(pred: np.ndarray, y_val: np.ndarray) -> float:
    pred = np.asarray(pred, float).reshape(-1)
    if pred.shape != y_val.shape or not np.isfinite(pred).all():
        return float("inf")
    var = float(np.var(y_val))
    return float(np.mean((pred - y_val) ** 2) / var) if var > 0 else float("inf")


def fit_problem(rec: dict, w: Any = None, prompt_prefix: Any = None) -> dict:
    """Fit one raw-shard problem record -> kept candidates and counts (see :mod:`flash_ansr.datagen.shards`)."""
    from flash_ansr.flash_ansr import Generation
    from flash_ansr.utils.tensor_ops import pad_input_set
    w = w if w is not None else _W["w"]
    prompt_prefix = prompt_prefix if prompt_prefix is not None else _W["pp"]
    m = w.flash_ansr_model
    t0 = time.time()
    seqs, scores, _valid = m._postprocess_sampled(rec["tokens"], rec["log_prob"], simplify=True, unique=True,
                                                  valid_only=True)
    out: dict[str, Any] = dict(n_raw=len(rec["tokens"]), n_distinct=len(seqs), n_fitted=0, cand=[], t_post=time.time() - t0)
    if not seqs:
        return out
    y = np.asarray(rec["y_noisy"], float).reshape(-1, 1)
    finite = np.isfinite(y).all(-1)
    seed = zlib.crc32(rec["problem_id"].encode())
    gs = Generation(raw_beams=seqs, log_probs=scores, X=pad_input_set(np.asarray(rec["x"], float), w.n_variables),
                    y=y, y_variance=float(y[finite].var()) if finite.sum() > 1 else float("nan"),
                    n_points=int(finite.sum()), prompt_prefix=prompt_prefix, memory=None, device="cpu",
                    variable_mapping={}, complexity=None, generation_time=0.0, draws=int(rec["k"]), seed=seed)
    before = np.geterr()
    np.seterr(all=w.numpy_errors)
    t1 = time.time()
    try:
        outcome = w._fit_refine(gs, converge_error="ignore", refine_seed=seed, verbose=False, allow_empty=True)
        result = w._build_result(gs, outcome)
    finally:
        np.seterr(**before)
    out["t_refine"] = time.time() - t1
    L = result.ledger
    parent = list(L.parent) if len(L.parent) == len(L) else [-1] * len(L)
    x_val, y_val = np.asarray(rec["x_val"], float), np.asarray(rec["y_val"], float).reshape(-1)
    root_of = {}
    for i in range(len(L)):
        j = i
        while parent[j] >= 0:
            j = parent[j]
        root_of[i] = j
    fam_ids: dict[int, int] = {}
    for i in range(len(L)):
        ri = L.result_index[i] if len(L.result_index) == len(L) else -1
        if ri is None or ri < 0:
            continue
        out["n_fitted"] += 1
        try:
            with np.errstate(all="ignore"):
                fv = _fresh_fvu(result.predict(x_val, rank=int(ri)), y_val)
        except Exception:  # noqa: BLE001  (a candidate that cannot be evaluated on fresh points is not kept)
            fv = float("inf")
        if not (np.isfinite(fv) and fv < 1.0):
            continue
        root = root_of[i]
        fam = fam_ids.setdefault(root, len(fam_ids))
        out["cand"].append(dict(fam=fam, tokens=list(map(int, L.token_lists[root])), fvu=float(L.fvu[i]), fvu_val=fv,
                                mdl=float(L.mdl[i]) if len(L.mdl) == len(L) else float("nan"),
                                score=float(L.score[i]) if len(L.score) == len(L) else float("nan"),
                                n_constants=int(L.n_constants[i]) if len(L.n_constants) == len(L) else -1,
                                is_variant=bool(parent[i] >= 0), log_prob=float(L.log_prob[root])))
    out["t_total"] = time.time() - t0
    return out


def _fit_shard(args: tuple[str, str]) -> tuple[str, int, int, float]:
    raw_path, out_path = args
    if os.path.exists(out_path):
        return out_path, -1, 0, 0.0
    t0 = time.time()
    recs = list(shards.iter_raw_shard(raw_path))
    fits = [fit_problem(r) for r in recs]
    cand = [c for f in fits for c in f["cand"]]
    fam_tok, fam_tok_off = shards.pack([np.asarray(c["tokens"], np.uint16) for c in cand], np.uint16)
    cand_off = np.zeros(len(recs) + 1, np.int64)
    cand_off[1:] = np.cumsum([len(f["cand"]) for f in fits])
    raw = shards.load(raw_path)
    keep = {k: raw[k] for k in ("problem_id", "k", "skeleton", "noise_level", "x", "x_off", "x_cols", "y_noisy", "y",
                                "y_off", "x_val", "x_val_off", "x_val_cols", "y_val", "y_val_off")}
    col = lambda key, dt: np.array([c[key] for c in cand], dt)  # noqa: E731
    shards.save(out_path, **keep, n_raw=np.array([f["n_raw"] for f in fits], np.int32),
                n_distinct=np.array([f["n_distinct"] for f in fits], np.int32),
                n_fitted=np.array([f["n_fitted"] for f in fits], np.int32),
                n_kept=np.array([len(f["cand"]) for f in fits], np.int32), cand_off=cand_off,
                fam_tok=fam_tok, fam_tok_off=fam_tok_off, fam_id=col("fam", np.int32), fvu=col("fvu", np.float64),
                fvu_val=col("fvu_val", np.float64), mdl=col("mdl", np.float64), score=col("score", np.float64),
                n_constants=col("n_constants", np.int16), is_variant=col("is_variant", bool),
                log_prob_draw=col("log_prob", np.float32),
                t_refine=np.array([f.get("t_refine", 0.0) for f in fits], np.float32),
                t_post=np.array([f.get("t_post", 0.0) for f in fits], np.float32))
    return out_path, len(recs), len(cand), time.time() - t0


def run(raw_dir: str, out_dir: str, model_dir: str, *, workers: int | None = None, n_restarts: int | None = None,
        part: tuple[int, int] = (0, 1)) -> None:
    """Fit every raw shard in ``raw_dir`` not yet fitted in ``out_dir``; ``part = (i, n)`` takes every n-th shard
    starting at i (one SLURM array task per part). Resumable: finished shards are skipped."""
    from multiprocessing import get_context
    from flash_ansr.utils.cpus import available_cores
    os.makedirs(out_dir, exist_ok=True)
    raws = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))[part[0]::part[1]]
    jobs = [(r, os.path.join(out_dir, os.path.basename(r))) for r in raws]
    jobs = [j for j in jobs if not os.path.exists(j[1])]
    workers = workers or available_cores()
    print(f"[fit] {len(jobs)} shards to fit with {workers} workers (part {part})", flush=True)
    t0, n_prob, n_cand = time.time(), 0, 0
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_dir, n_restarts)) as pool:
        for k, (path, n, c, dt) in enumerate(pool.imap_unordered(_fit_shard, jobs)):
            if n > 0:
                n_prob, n_cand = n_prob + n, n_cand + c
            el = time.time() - t0
            print(f"[fit] {k + 1}/{len(jobs)} {os.path.basename(path)}: {n} problems, {c} kept, {dt:.0f} s | "
                  f"total {n_prob} problems in {el / 60:.1f} min = {n_prob / max(el, 1e-9):.2f} problems/s", flush=True)
