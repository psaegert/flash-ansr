"""Stage 3 (CPU): training targets from fitted shards (front-member targets and the self-distillation control).

The front of one problem's pool, as the offline oracle defines it (flash-ansr-rl ``kfront_check.py`` /
``target_check_v2.py``):
- MEMBERS are the pool's distinct raw expressions ("actions"): two raw forms that collapse into one family are two
  members. An action's ROWS are the fitted candidates of its family (the root and its constant-ladder variants).
- A row is the point ``b = mdl / 1000`` (bits) and ``y = max(log10(max(fresh FVU, eps)), floor)``, the floor being
  -7 for a clean problem and ``max(-9, log10(level**2 / 100))`` for noise level ``level > 0``.
- Action z dominates action x iff every row of x is dominated (weakly in both, strictly in one) by some row of z; an
  action without rows is dominated by every action with rows. The constant model, the point (8 bits, 0), dominates
  an action whose every row it dominates.
- The front is the set of actions dominated neither by another action of the pool nor by the constant model.

Targets per problem (only problems with a non-empty front are written):
- FRONT: every front member gets 1/|front|, split evenly over its distinct token sequences (the raw emissions; for
  one raw expression they almost always coincide). This is the exact expectation of "a uniformly random front member".
- CONTROL: m = |front| actions drawn uniformly without replacement (seeded by the problem id) from the actions with
  a kept family, 1/m each: the expectation of "a uniformly random kept action" with the same number of sequences.

Target shard keys: problem_id, k, noise_level, x, x_off, x_cols, y_noisy, y_off (the model's input data);
n_actions, n_kept_actions, n_front [P]; for each of ``front`` and ``ctrl``: ``{name}_off [P + 1]`` (sequences of
problem p), ``{name}_tok, {name}_tok_off`` (uint16 sequences, the sampling prompt prefix included), ``{name}_w``.
"""
from __future__ import annotations

import glob
import os
import time
import zlib
from typing import Any

import numpy as np

from flash_ansr.datagen import shards

B_NULL, Y_NULL = 8.0, 0.0          # the constant model: ~8 bits at FVU 1, a virtual member of every pool
EPS = 2.220446049250313e-16


def y_floor(noise_level: float) -> float:
    return -7.0 if not noise_level > 0 else max(-9.0, float(np.log10(noise_level ** 2 / 100.0)))


def pool_front(draw_act: np.ndarray, draw_fam: np.ndarray, cand_fam: np.ndarray, cand_b: np.ndarray,
               cand_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(actions, action family, front mask)`` of one pool.

    ``draw_act`` / ``draw_fam``: per raw draw, its action and family (-1: none); ``cand_fam`` / ``cand_b`` / ``cand_y``:
    per usable fitted row, its family and point. Actions are returned in increasing id order."""
    acts = np.unique(draw_act[draw_act >= 0])
    afam = np.array([int(draw_fam[np.flatnonzero(draw_act == a)[0]]) for a in acts], np.int64)
    rows = [np.flatnonzero(cand_fam == f) if f >= 0 else np.zeros(0, np.int64) for f in afam]
    b, y = np.asarray(cand_b, float), np.asarray(cand_y, float)
    A, R = len(acts), len(b)
    if A == 0:
        return acts, afam, np.zeros(0, bool)
    Dp = (b[:, None] <= b[None, :]) & (y[:, None] <= y[None, :]) & ((b[:, None] < b[None, :]) | (y[:, None] < y[None, :]))
    Ind = np.zeros((A, R))
    for i, r in enumerate(rows):
        Ind[i, r] = 1.0
    size = Ind.sum(1)
    M = (Ind @ Dp.astype(float)) > 0                      # M[a, r]: some row of action a dominates row r
    hit = M.astype(float) @ Ind.T                         # hit[z, x]: rows of x dominated by some row of z
    FD = (hit >= size[None, :]) & (size[None, :] > 0)
    FD |= (size[None, :] == 0) & (size[:, None] > 0)
    np.fill_diagonal(FD, False)
    nd_row = (B_NULL <= b) & (Y_NULL <= y) & ((B_NULL < b) | (Y_NULL < y))
    nul = np.array([bool(nd_row[r].all()) if len(r) else True for r in rows])
    return acts, afam, ~FD.any(0) & ~nul


def problem_targets(problem_id: str, tokens: list[list[int]], draw_act: np.ndarray, draw_fam: np.ndarray,
                    cand_fam: np.ndarray, cand_mdl: np.ndarray, cand_fvu_val: np.ndarray, cand_score: np.ndarray,
                    noise_level: float) -> dict[str, Any]:
    """Front and control targets of one problem: ``{'front': [(seq, w)], 'ctrl': [(seq, w)], counts}``."""
    use = np.isfinite(cand_mdl) & np.isfinite(cand_fvu_val) & np.isfinite(cand_score)
    fl = y_floor(noise_level)
    cy = np.maximum(np.log10(np.maximum(cand_fvu_val[use], EPS)), fl)
    acts, afam, front = pool_front(draw_act, draw_fam, cand_fam[use], cand_mdl[use] / 1000.0, cy)

    def seqs_of(a: int) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        for d in np.flatnonzero(draw_act == a):
            s = tuple(tokens[d])
            if s not in out:
                out.append(s)
        return out

    def weighted(members: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
        res: list[tuple[tuple[int, ...], float]] = []
        for a in members:
            ss = seqs_of(int(a))
            res.extend((s, 1.0 / len(members) / len(ss)) for s in ss)
        return res

    f_acts = acts[front]
    kept = acts[afam >= 0]
    rng = np.random.default_rng(zlib.crc32(problem_id.encode()))
    c_acts = rng.choice(kept, size=min(len(f_acts), len(kept)), replace=False) if len(f_acts) else kept[:0]
    return dict(front=weighted(f_acts), ctrl=weighted(np.sort(c_acts)), n_actions=len(acts), n_kept_actions=len(kept),
                n_front=int(front.sum()))


def _targets_shard(args: tuple[str, str, str]) -> tuple[str, int, int, float]:
    raw_path, fit_path, out_path = args
    if os.path.exists(out_path):
        return out_path, -1, 0, 0.0
    t0 = time.time()
    raw, fit = shards.load(raw_path), shards.load(fit_path)
    assert (raw["problem_id"] == fit["problem_id"]).all() and (raw["draw_off"] == fit["draw_off"]).all()
    keep, res = [], []
    for p in range(len(raw["problem_id"])):
        d0, d1 = int(raw["draw_off"][p]), int(raw["draw_off"][p + 1])
        c0, c1 = int(fit["cand_off"][p]), int(fit["cand_off"][p + 1])
        toks = [raw["tok"][raw["tok_off"][j]:raw["tok_off"][j + 1]].astype(np.int64).tolist() for j in range(d0, d1)]
        r = problem_targets(str(raw["problem_id"][p]), toks, fit["draw_act"][d0:d1], fit["draw_fam"][d0:d1],
                            fit["fam_id"][c0:c1], fit["mdl"][c0:c1], fit["fvu_val"][c0:c1], fit["score"][c0:c1],
                            float(raw["noise_level"][p]))
        if r["n_front"] > 0:
            keep.append(p)
            res.append(r)
    sel = lambda flat, off, cols=None: ((shards.pack2d([shards.unpack2d(flat, off, cols, p) for p in keep]))  # noqa: E731
                                        if cols is not None else shards.pack([shards.unpack(flat, off, p) for p in keep],
                                                                             flat.dtype))
    x, x_off, x_cols = sel(raw["x"], raw["x_off"], raw["x_cols"])
    y_noisy, y_off = sel(raw["y_noisy"], raw["y_off"])
    out: dict[str, np.ndarray] = dict(problem_id=raw["problem_id"][keep], k=raw["k"][keep],
                                      noise_level=raw["noise_level"][keep], x=x, x_off=x_off, x_cols=x_cols,
                                      y_noisy=y_noisy, y_off=y_off,
                                      n_actions=np.array([r["n_actions"] for r in res], np.int32),
                                      n_kept_actions=np.array([r["n_kept_actions"] for r in res], np.int32),
                                      n_front=np.array([r["n_front"] for r in res], np.int32))
    for name in ("front", "ctrl"):
        per = [r[name] for r in res]
        off = np.zeros(len(per) + 1, np.int64)
        off[1:] = np.cumsum([len(t) for t in per])
        tok, tok_off = shards.pack([np.asarray(s, np.uint16) for t in per for s, _ in t], np.uint16)
        out.update({f"{name}_off": off, f"{name}_tok": tok, f"{name}_tok_off": tok_off,
                    f"{name}_w": np.array([w for t in per for _, w in t], np.float64)})
    shards.save(out_path, **out)
    return out_path, len(raw["problem_id"]), len(keep), time.time() - t0


def run(raw_dir: str, fit_dir: str, out_dir: str, *, workers: int | None = None, part: tuple[int, int] = (0, 1)) -> None:
    """Build target shards for every fitted shard in ``fit_dir`` (its raw shard of the same name in ``raw_dir``)
    not yet built in ``out_dir``. Resumable; ``part = (i, n)`` as in :func:`flash_ansr.datagen.fit.run`."""
    from multiprocessing import get_context
    from flash_ansr.utils.cpus import available_cores
    os.makedirs(out_dir, exist_ok=True)
    fits = sorted(p for p in glob.glob(os.path.join(fit_dir, "*.npz")) if not p.endswith(".part.npz"))[part[0]::part[1]]
    jobs = [(os.path.join(raw_dir, os.path.basename(f)), f, os.path.join(out_dir, os.path.basename(f))) for f in fits]
    jobs = [j for j in jobs if not os.path.exists(j[2])]
    workers = workers or available_cores()
    print(f"[targets] {len(jobs)} shards with {workers} workers (part {part})", flush=True)
    t0, n_in, n_out = time.time(), 0, 0
    with get_context("spawn").Pool(workers) as pool:
        for k, (path, n, c, dt) in enumerate(pool.imap_unordered(_targets_shard, jobs)):
            if n > 0:
                n_in, n_out = n_in + n, n_out + c
            print(f"[targets] {k + 1}/{len(jobs)} {os.path.basename(path)}: {c}/{n} problems with a front, {dt:.1f} s "
                  f"| total {n_out}/{n_in} in {(time.time() - t0) / 60:.1f} min", flush=True)
