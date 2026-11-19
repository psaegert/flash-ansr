"""Stage 2 (CPU nodes): fit the raw draws of each problem with the unchanged deployed pipeline.

Per problem: the deployed post-processing (extract, simplify, deduplicate; :meth:`FlashANSRModel._postprocess_sampled`,
the same call the chunked sampler makes), then :meth:`FlashANSR._fit_refine` (refit with the configured restarts,
constant ladder) and :meth:`FlashANSR._build_result` (two-part ranking). Every fitted ledger row is scored on the
problem's clean validation rows (fresh-point FVU). Rows whose fresh FVU is not below 1 are dropped (worse than the
mean on fresh points). A FAMILY is the set of ledger rows of one deduplicated draw: its fit, its constant-ladder
variants (``parent`` set) and its typed-thaw re-fits (same beam, no parent); they share the draw's log-prob, which is
the family key, exactly as the offline oracle (flash-ansr-rl ``target_check_v2``) groups them. Each kept row records
its beam, the raw emission of the first draw of its dedup class (post-processing deduplicates on the simplified
expression but hands on the sequence the model emitted).

Every raw draw is traced to its family exactly as the evaluation probe does (``trace_draws``): its status, its
ACTION (the index of its distinct raw expression within the problem; the unit of the front-member target) and the
kept family it collapsed into. These per-draw arrays are aligned with the raw shard's draws, so the training target
of a front member is read back from the raw shard as the model emitted it.

One estimator per worker process with a serial refiner (``ComputeConfig(workers=0)``): parallelism is across
problems, never nested.
"""
from __future__ import annotations

import glob
import os
import time
import zlib
from typing import Any, Iterator

import numpy as np

from flash_ansr.datagen import shards

_W: dict[str, Any] = {}

# per-draw status codes (fitted shard key ``draw_status``)
DRAW_OK, NO_EXPRESSION, MALFORMED_SPAN, INVALID, NONFINITE, NOT_IN_LEDGER = range(6)
STATUS_NAMES = ("ok", "no_expression", "malformed_span", "invalid", "nonfinite", "not_in_ledger")


def _dump_stacks_later() -> None:
    """``DG_FAULTHANDLER=<seconds>``: dump every thread's stack to stderr every that many seconds (hang diagnosis
    without ptrace)."""
    secs = os.environ.get("DG_FAULTHANDLER")
    if secs:
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(int(secs), repeat=True, file=sys.stderr)


def _init_worker(model_dir: str, n_restarts: int | None) -> None:
    _dump_stacks_later()
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


def trace_draws(m: Any, seqs: list[list[int]]) -> tuple[list[tuple[int, tuple | None, tuple | None, tuple | None]],
                                                        dict[tuple, list]]:
    """Per raw draw ``(status, raw expression, dedup key, expression body)``, computed exactly as
    :meth:`FlashANSRModel._postprocess_sampled` computes them (same extraction, span realization, validity test and
    simplification), plus the ``simplify_map`` that lets ``_postprocess_sampled`` reuse these simplifications (its
    output is byte-identical with the map). Each distinct raw expression is simplified once."""
    from flash_ansr.utils.skeleton import NonFiniteExpressionError, simplify_realized
    tok, engine = m.tokenizer, m.simplipy_engine
    cache: dict[tuple, tuple | None] = {}
    out: list[tuple[int, tuple | None, tuple | None, tuple | None]] = []
    for seq in seqs:
        try:
            enc, _before, _after = tok.extract_expression_from_beam(list(seq))
        except (ValueError, IndexError):
            out.append((NO_EXPRESSION, None, None, None))
            continue
        realized = m._realize_ieee754_spans(list(enc))
        if realized is None:
            out.append((MALFORMED_SPAN, None, None, None))
            continue
        expression = tok.constantify_expression(realized, exact=True)
        if not (engine.is_valid(expression) and len(expression) > 1):
            out.append((INVALID, None, None, None))
            continue
        raw = tuple(expression)
        if raw not in cache:
            try:
                cache[raw] = tuple(simplify_realized(engine, list(expression)))
            except NonFiniteExpressionError:
                cache[raw] = None
        key = cache[raw]
        out.append((NONFINITE, raw, None, None) if key is None else (DRAW_OK, raw, key, tuple(enc)))
    return out, {k: list(v) for k, v in cache.items() if v is not None}


def fit_problem(rec: dict, w: Any = None, prompt_prefix: Any = None) -> dict:
    """Fit one raw-shard problem record -> kept candidates, the per-draw map and counts (see
    :mod:`flash_ansr.datagen.shards`)."""
    from flash_ansr.flash_ansr import Generation
    from flash_ansr.utils.tensor_ops import pad_input_set
    w = w if w is not None else _W["w"]
    prompt_prefix = prompt_prefix if prompt_prefix is not None else _W["pp"]
    m = w.flash_ansr_model
    t0 = time.time()
    trace, smap = trace_draws(m, rec["tokens"])
    seqs, scores, _valid = m._postprocess_sampled(rec["tokens"], rec["log_prob"], simplify=True, unique=True,
                                                  valid_only=True, simplify_map=smap)
    n_raw = len(rec["tokens"])
    status = np.array([t[0] for t in trace], np.int8)
    act_of_raw: dict[tuple, int] = {}
    draw_act = np.array([act_of_raw.setdefault(t[1], len(act_of_raw)) if t[1] is not None else -1 for t in trace],
                        np.int32)
    out: dict[str, Any] = dict(n_raw=n_raw, n_distinct=len(seqs), n_fitted=0, cand=[], t_post=time.time() - t0,
                               draw_status=status, draw_act=draw_act, draw_fam=np.full(n_raw, -1, np.int32))
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
    fam_ids: dict[float, int] = {}                                  # family key: the draw's log-prob
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
        fam = fam_ids.setdefault(float(L.log_prob[i]), len(fam_ids))
        out["cand"].append(dict(fam=fam, tokens=list(map(int, L.token_lists[i])), fvu=float(L.fvu[i]), fvu_val=fv,
                                mdl=float(L.mdl[i]) if len(L.mdl) == len(L) else float("nan"),
                                score=float(L.score[i]) if len(L.score) == len(L) else float("nan"),
                                n_constants=int(L.n_constants[i]) if len(L.n_constants) == len(L) else -1,
                                is_variant=bool(parent[i] >= 0), log_prob=float(L.log_prob[i])))
    # raw draw -> its dedup class's representative (first draw with the key) -> the draw's ledger row (the first
    # parentless row with that body) -> its family (log-prob)
    tok = m.tokenizer
    row_of_body: dict[tuple, int] = {}
    for j in range(len(L)):
        if parent[j] < 0:
            row_of_body.setdefault(tuple(tok.extract_expression_from_beam(list(L.token_lists[j]))[0]), j)
    first_of_key: dict[tuple, tuple] = {}
    for d, (st, _raw, key, body) in enumerate(trace):
        if st != DRAW_OK:
            continue
        rep = first_of_key.setdefault(key, body)                    # type: ignore[arg-type]
        row = row_of_body.get(rep, -1)                              # type: ignore[arg-type]
        if row < 0:
            status[d] = NOT_IN_LEDGER
            continue
        out["draw_fam"][d] = fam_ids.get(float(L.log_prob[row]), -1)
    out["t_total"] = time.time() - t0
    return out


def _write_fitted(raw_path: str, out_path: str, fits: list[dict]) -> int:
    """Write one fitted shard from the per-problem fits (problem order of the raw shard). The problem data stay in
    the raw shard (aligned by position and ``draw_off``); only identifiers are repeated here."""
    cand = [c for f in fits for c in f["cand"]]
    fam_tok, fam_tok_off = shards.pack([np.asarray(c["tokens"], np.uint16) for c in cand], np.uint16)
    cand_off = np.zeros(len(fits) + 1, np.int64)
    cand_off[1:] = np.cumsum([len(f["cand"]) for f in fits])
    raw = shards.load(raw_path)
    keep = {k: raw[k] for k in ("problem_id", "k", "skeleton", "noise_level", "draw_off")}
    col = lambda key, dt: np.array([c[key] for c in cand], dt)  # noqa: E731
    cat = lambda key, dt: (np.concatenate([f[key] for f in fits]).astype(dt) if fits else np.zeros(0, dt))  # noqa: E731
    shards.save(out_path, **keep, n_raw=np.array([f["n_raw"] for f in fits], np.int32),
                n_distinct=np.array([f["n_distinct"] for f in fits], np.int32),
                n_fitted=np.array([f["n_fitted"] for f in fits], np.int32),
                n_kept=np.array([len(f["cand"]) for f in fits], np.int32), cand_off=cand_off,
                fam_tok=fam_tok, fam_tok_off=fam_tok_off, fam_id=col("fam", np.int32), fvu=col("fvu", np.float64),
                fvu_val=col("fvu_val", np.float64), mdl=col("mdl", np.float64), score=col("score", np.float64),
                n_constants=col("n_constants", np.int16), is_variant=col("is_variant", bool),
                log_prob_draw=col("log_prob", np.float32),
                draw_status=cat("draw_status", np.int8), draw_act=cat("draw_act", np.int32),
                draw_fam=cat("draw_fam", np.int32),
                t_refine=np.array([f.get("t_refine", 0.0) for f in fits], np.float32),
                t_post=np.array([f.get("t_post", 0.0) for f in fits], np.float32))
    return len(cand)


def _fit_shard(args: tuple[str, str]) -> tuple[str, int, int, float]:
    """Fit one raw shard serially in this process (tests, small runs)."""
    raw_path, out_path = args
    if os.path.exists(out_path):
        return out_path, -1, 0, 0.0
    t0 = time.time()
    fits = [fit_problem(r) for r in shards.iter_raw_shard(raw_path)]
    n_cand = _write_fitted(raw_path, out_path, fits)
    return out_path, len(fits), n_cand, time.time() - t0


def _fit_task(task: tuple[tuple[int, int], dict]) -> tuple[tuple[int, int], dict]:
    key, rec = task
    return key, fit_problem(rec)


def run(raw_dir: str, out_dir: str, model_dir: str, *, workers: int | None = None, n_restarts: int | None = None,
        part: tuple[int, int] = (0, 1), shards_in_flight: int = 4) -> None:
    """Fit every raw shard in ``raw_dir`` not yet fitted in ``out_dir``; ``part = (i, n)`` takes every n-th shard
    starting at i (one SLURM array task per part). Resumable: finished shards are skipped.

    Parallel over PROBLEMS (not shards), so a node never idles on a last, partial wave of shards; at most
    ``shards_in_flight`` shards are loaded at once, and a shard is written as soon as its last problem returns."""
    import threading
    from multiprocessing import get_context
    from flash_ansr.utils.cpus import available_cores
    os.makedirs(out_dir, exist_ok=True)
    raws = sorted(p for p in glob.glob(os.path.join(raw_dir, "*.npz")) if not p.endswith(".part.npz"))
    raws = raws[part[0]::part[1]]                                 # a .part.npz is a shard still being written
    jobs = [(r, os.path.join(out_dir, os.path.basename(r))) for r in raws]
    jobs = [j for j in jobs if not os.path.exists(j[1])]
    workers = workers or available_cores()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")     # one core per worker; inherited by the spawned workers before numpy loads
    _dump_stacks_later()
    print(f"[fit] {len(jobs)} shards to fit with {workers} workers, problem-level (part {part})", flush=True)
    slots = threading.BoundedSemaphore(max(1, shards_in_flight))
    stop = threading.Event()               # set on any exit, so the feeder never blocks Pool.terminate()
    n_of: dict[int, int] = {}

    def tasks() -> Iterator[tuple[tuple[int, int], dict]]:     # runs in the pool's feeder thread
        for si, (raw_path, _) in enumerate(jobs):
            while not slots.acquire(timeout=1.0):
                if stop.is_set():
                    return
            recs = list(shards.iter_raw_shard(raw_path))
            n_of[si] = len(recs)
            if not recs:                                          # nothing will come back for an empty shard
                _write_fitted(*jobs[si], [])
                slots.release()
                continue
            for p, rec in enumerate(recs):
                yield (si, p), rec

    t0, n_prob, n_cand, done = time.time(), 0, 0, 0
    pending: dict[int, dict[int, dict]] = {}
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_dir, n_restarts)) as pool:
        try:
            for (si, p), res in pool.imap_unordered(_fit_task, tasks(), chunksize=1):
                pending.setdefault(si, {})[p] = res
                if len(pending[si]) == n_of[si]:
                    got = pending.pop(si)
                    fits = [got[q] for q in range(n_of[si])]
                    c = _write_fitted(*jobs[si], fits)
                    slots.release()
                    done, n_prob, n_cand = done + 1, n_prob + len(fits), n_cand + c
                    el = time.time() - t0
                    print(f"[fit] {done}/{len(jobs)} {os.path.basename(jobs[si][1])}: {len(fits)} problems, {c} kept | "
                          f"total {n_prob} problems in {el / 60:.1f} min = {n_prob / max(el, 1e-9):.2f} problems/s",
                          flush=True)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            stop.set()
