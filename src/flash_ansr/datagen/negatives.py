"""Stage 3b (CPU): T23 negatives, the owner-approved design v5 (2026-11-25; flash-ansr-rl t23, the ladder session's
``negatives_plan.md``): pairs (y+, y-) for a dominance preference term next to the front targets.

Per problem and its training pool size K (assigned exactly as the front targets, :func:`targets.assign_k`):
- POINTS as the fronts (:mod:`flash_ansr.datagen.targets`): a row is ``b = mdl / 1000`` bits and
  ``y = max(log10(max(FVU_val, eps)), floor)``, unclipped above (FVU_val >= 1 gives y >= 0, inf stays inf).
- An ACTION (distinct raw expression) has the rows of its family: those with FVU_val < 1 if it has any, else its rows
  with FVU_val >= 1 (stored only by ``fit --keep-all-rows``). Fronts ignore FVU >= 1 rows, and a family's worse ladder
  variants must not change a front.
- Action z DOMINATES action x iff every row of x is dominated (weakly in both, strictly in one) by some row of z. The
  virtual point V = (-inf bits, y = 0), "the mean predictor is always available", dominates every action whose rows
  all have FVU_val >= 1. Dominance between two actions does not depend on the pool, so it is built once per problem.
- F_K: the actions among the first K draws dominated neither by another of them nor by V. This is the front targets'
  front (``targets.pool_front``); the tests check that.
- LAYERS: non-dominated sorting of the problem's FULL pool, V in layer 1 (V is never a positive). A negative's weight
  is read from its full-pool layer (v5: comparable across instances; a K-pool layer grows with K). The trainer turns
  it into w = log2(layer) or the raw layer.
- PAIRS: y- draw-uniform without replacement (up to ``pairs``) over the eligible draws among the first K: status ok,
  a stored row, not an F_K member, not a draw of the true law (fit shard ``draw_gt``), and dominated by at least one
  F_K member. y+ uniform over the F_K members with a row that dominates y-'s representative row (its best-FVU row,
  ties to the shorter); y+'s sequence uniform over its distinct raw emissions among the first K draws.
- Counted, not paired: invalid emissions (status not ok), ok draws without a stored row (no fit at all; on shards
  without ``rows_all`` also every FVU >= 1 draw), true-law draws, and draws only V dominates (an FVU >= 1 draw
  shorter than every member).

Negative shard keys (no problem data; same name as the raw shard): problem_id, k, k_spec (0-d), rows_all (0-d), and per
problem: n_pairs, pair_off [P + 1], n_front_k, n_elig, n_drop_vonly, n_invalid, n_unfitted, n_gt, n_nm_fit, n_nm_junk
(K-pool non-member draws with FVU < 1 rows / with FVU >= 1 rows only), n_fam_full, n_layers_8, n_layers_64,
n_layers_full, nm_layer_med, nm_layer_p90, nm_layer_max (full-pool layers of the full pool's non-front actions); per
pair: pos_tok / pos_tok_off, neg_tok / neg_tok_off (uint16, prompt prefix included, as the front targets), layer,
neg_kind (0: y- has FVU < 1 rows; 1: FVU >= 1 rows only), pos_b / pos_y (y+'s representative row), pos_dom_b /
pos_dom_y (the longest row of y+ that dominates y-'s representative), neg_b / neg_y (y-'s representative row).
"""
from __future__ import annotations

import glob
import os
import time
import zlib
from typing import Any, Callable

import numpy as np

from flash_ansr.datagen import problems as PR
from flash_ansr.datagen import shards
from flash_ansr.datagen.fit import DRAW_OK
from flash_ansr.datagen.targets import EPS, assign_k, y_floor


def action_rows(draw_status: np.ndarray, draw_act: np.ndarray, draw_fam: np.ndarray, cand_fam: np.ndarray,
                cand_fvu_val: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(actions, row indices per action)`` over the ok draws: FVU < 1 rows if any, else the action's other usable
    rows. An action whose family has no usable row gets an empty row set."""
    ok = (draw_status == DRAW_OK) & (draw_act >= 0)
    acts = np.unique(draw_act[ok])
    rows = []
    for a in acts:
        f = int(draw_fam[np.flatnonzero(ok & (draw_act == a))[0]])
        r = np.flatnonzero((cand_fam == f) & usable) if f >= 0 else np.zeros(0, np.int64)
        lt1 = r[cand_fvu_val[r] < 1.0]
        rows.append(lt1 if len(lt1) else r)
    return acts, rows


def dominance(rows: list[np.ndarray], b: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(FD, M, vdom)`` over actions with rows: FD[z, x] action z dominates action x; M[a, r] some row of action a
    dominates row r (r indexes ``b`` / ``y``); vdom[x] V dominates x (every row of x has y >= 0)."""
    A = len(rows)
    Dp = (b[:, None] <= b[None, :]) & (y[:, None] <= y[None, :]) & ((b[:, None] < b[None, :]) | (y[:, None] < y[None, :]))
    Ind = np.zeros((A, len(b)), np.float32)
    for i, r in enumerate(rows):
        Ind[i, r] = 1.0
    size = Ind.sum(1)
    M = (Ind @ Dp.astype(np.float32)) > 0
    hit = M.astype(np.float32) @ Ind.T
    FD = (hit >= size[None, :] - 0.5) & (size[None, :] > 0)
    np.fill_diagonal(FD, False)
    vdom = np.array([bool((y[r] >= 0).all()) for r in rows])
    return FD, M, vdom


def layers(FD: np.ndarray, vdom: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """Non-dominated sorting depth of the selected actions (all if ``sel`` is None), V peeled first in layer 1.
    Returns the layer per action (0 for actions outside ``sel``). Dominance is a strict partial order, so this is the
    longest-chain depth (Kahn layering)."""
    A = len(vdom)
    sel = np.ones(A, bool) if sel is None else sel
    F = FD & sel[:, None] & sel[None, :]
    indeg = F.sum(0).astype(np.int64) + (vdom & sel)
    layer = np.zeros(A, np.int64)
    remaining = sel.copy()
    cur = remaining & (indeg == 0)
    L = 1
    while True:
        layer[cur] = L
        remaining &= ~cur
        dec = F[cur].sum(0)
        if L == 1:
            dec = dec + (vdom & sel)                       # V leaves with layer 1
        indeg = indeg - dec
        cur = remaining & (indeg == 0)
        if not cur.any():
            break
        L += 1
    assert not remaining.any(), "dominance cycle"
    return layer


def problem_negatives(problem_id: str, k: int, k_spec: str, tok_of: Callable[[int], tuple[int, ...]],
                      draw_status: np.ndarray, draw_act: np.ndarray, draw_fam: np.ndarray, draw_gt: np.ndarray,
                      cand_fam: np.ndarray, cand_mdl: np.ndarray, cand_fvu_val: np.ndarray, cand_score: np.ndarray,
                      noise_level: float, pairs: int = 8) -> dict[str, Any]:
    """The negatives of one problem at pool size ``k`` (see the module docstring)."""
    usable = np.isfinite(cand_mdl) & np.isfinite(cand_score) & ~np.isnan(cand_fvu_val)
    b = cand_mdl / 1000.0
    y = np.maximum(np.log10(np.maximum(np.where(np.isnan(cand_fvu_val), np.inf, cand_fvu_val), EPS)),
                   y_floor(noise_level))
    acts, rows = action_rows(draw_status, draw_act, draw_fam, cand_fam, cand_fvu_val, usable)
    has = np.array([len(r) > 0 for r in rows], bool)
    out: dict[str, Any] = dict(pairs=[], n_front_k=0, n_elig=0, n_drop_vonly=0, n_invalid=0, n_unfitted=0, n_gt=0,
                               n_nm_fit=0, n_nm_junk=0, n_fam_full=int(has.sum()), n_layers_8=0, n_layers_64=0,
                               n_layers_full=0, nm_layer_med=0.0, nm_layer_p90=0.0, nm_layer_max=0)
    kk = min(k, len(draw_status))
    out["n_invalid"] = int((draw_status[:kk] != DRAW_OK).sum())
    if not has.any():
        out["n_unfitted"] = int(((draw_status[:kk] == DRAW_OK) & (draw_act[:kk] >= 0)).sum())
        return out
    ai = np.flatnonzero(has)                              # actions with rows, in a compact index
    A = acts[ai]
    R = [rows[i] for i in ai]
    used = np.unique(np.concatenate(R))
    remap = {int(r): j for j, r in enumerate(used)}
    Rc = [np.array([remap[int(r)] for r in rr], np.int64) for rr in R]
    FD, M, vdom = dominance(Rc, b[used], y[used])
    pos_of = {int(a): j for j, a in enumerate(A)}
    rep = np.array([rr[np.lexsort((b[used][rr], y[used][rr]))[0]] for rr in Rc], np.int64)  # best y, then shortest

    def present(n: int) -> np.ndarray:
        m = np.zeros(len(A), bool)
        for a in np.unique(draw_act[:n][(draw_status[:n] == DRAW_OK) & (draw_act[:n] >= 0)]):
            if int(a) in pos_of:
                m[pos_of[int(a)]] = True
        return m

    full = layers(FD, vdom)
    out["n_layers_full"] = int(full.max())
    for n, key in ((8, "n_layers_8"), (64, "n_layers_64")):
        sel = present(n)
        out[key] = int(layers(FD, vdom, sel).max()) if sel.any() else 0
    nm = full[full > 1]
    if len(nm):
        out.update(nm_layer_med=float(np.median(nm)), nm_layer_p90=float(np.quantile(nm, 0.9)), nm_layer_max=int(nm.max()))

    # F_K: the K-pool's actions dominated neither by another K-pool action nor by V
    inK = present(kk)
    front = inK & ~vdom & ~(FD & inK[:, None]).any(0)
    out["n_front_k"] = int(front.sum())
    members = np.flatnonzero(front)
    rng = np.random.default_rng([zlib.crc32(problem_id.encode()), zlib.crc32(f"neg:{k_spec}".encode())])
    elig: list[tuple[int, np.ndarray]] = []
    for d in range(kk):
        if draw_status[d] != DRAW_OK or draw_act[d] < 0:
            continue
        j = pos_of.get(int(draw_act[d]))
        if j is None:
            out["n_unfitted"] += 1
            continue
        if front[j]:
            continue
        if bool(draw_gt[d]):
            out["n_gt"] += 1
            continue
        junk = bool(vdom[j])
        out["n_nm_junk" if junk else "n_nm_fit"] += 1
        dom = members[M[members, rep[j]]]
        if not len(dom):
            out["n_drop_vonly"] += 1
            continue
        elig.append((d, dom))
    out["n_elig"] = len(elig)
    if not elig or not len(members):
        return out
    pick = rng.choice(len(elig), size=min(pairs, len(elig)), replace=False)
    bu, yu = b[used], y[used]
    for i in np.sort(pick):
        d, dom = elig[i]
        jn = pos_of[int(draw_act[d])]
        jp = int(dom[rng.integers(len(dom))])
        draws_p = [q for q in range(kk) if draw_status[q] == DRAW_OK and pos_of.get(int(draw_act[q])) == jp]
        seqs: list[tuple[int, ...]] = []
        for q in draws_p:
            s = tok_of(q)
            if s not in seqs:
                seqs.append(s)
        r = rep[jn]                                        # y+'s rows dominating y-'s representative: the longest
        dr = Rc[jp][(bu[Rc[jp]] <= bu[r]) & (yu[Rc[jp]] <= yu[r]) & ((bu[Rc[jp]] < bu[r]) | (yu[Rc[jp]] < yu[r]))]
        ds = dr[np.lexsort((yu[dr], -bu[dr]))[0]]
        out["pairs"].append(dict(pos=seqs[rng.integers(len(seqs))], neg=tok_of(d), layer=int(full[jn]),
                                 neg_kind=int(vdom[jn]), pos_b=float(bu[rep[jp]]), pos_y=float(yu[rep[jp]]),
                                 pos_dom_b=float(bu[ds]), pos_dom_y=float(yu[ds]),
                                 neg_b=float(bu[r]), neg_y=float(yu[r])))
    return out


_PROBLEM_KEYS = ("n_front_k", "n_elig", "n_drop_vonly", "n_invalid", "n_unfitted", "n_gt", "n_nm_fit", "n_nm_junk",
                 "n_fam_full", "n_layers_8", "n_layers_64", "n_layers_full", "nm_layer_max")


def _negatives_shard(args: tuple[str, str, str, str, int]) -> tuple[str, int, int, float]:
    raw_path, fit_path, out_path, k_spec, pairs = args
    if os.path.exists(out_path):
        return out_path, -1, 0, 0.0
    t0 = time.time()
    raw = shards.load(raw_path, ("problem_id", "draw_off", "tok", "tok_off", "noise_level"))
    fit = shards.load(fit_path)
    assert (raw["problem_id"] == fit["problem_id"]).all() and (raw["draw_off"] == fit["draw_off"]).all()
    tok, tok_off = raw["tok"], raw["tok_off"]
    gt_all = fit["draw_gt"] if "draw_gt" in fit else np.zeros(len(fit["draw_status"]), bool)
    res, ks = [], []
    for p in range(len(raw["problem_id"])):
        pid = str(raw["problem_id"][p])
        d0, d1 = int(raw["draw_off"][p]), int(raw["draw_off"][p + 1])
        c0, c1 = int(fit["cand_off"][p]), int(fit["cand_off"][p + 1])
        k = assign_k(pid, k_spec)
        ks.append(k)

        def tok_of(d: int, d0: int = d0) -> tuple[int, ...]:
            return tuple(tok[tok_off[d0 + d]:tok_off[d0 + d + 1]].tolist())
        res.append(problem_negatives(pid, k, k_spec, tok_of, fit["draw_status"][d0:d1], fit["draw_act"][d0:d1],
                                     fit["draw_fam"][d0:d1], gt_all[d0:d1], fit["fam_id"][c0:c1], fit["mdl"][c0:c1],
                                     fit["fvu_val"][c0:c1], fit["score"][c0:c1], float(raw["noise_level"][p]),
                                     pairs=pairs))
    prs = [pr for r in res for pr in r["pairs"]]
    off = np.zeros(len(res) + 1, np.int64)
    off[1:] = np.cumsum([len(r["pairs"]) for r in res])
    pos, pos_off = shards.pack([np.asarray(pr["pos"], np.uint16) for pr in prs], np.uint16)
    neg, neg_off = shards.pack([np.asarray(pr["neg"], np.uint16) for pr in prs], np.uint16)
    out = dict(problem_id=raw["problem_id"], k=np.array(ks, np.int32), k_spec=np.array(k_spec),
               rows_all=np.array(bool(fit["rows_all"]) if "rows_all" in fit else False),
               n_pairs=np.array([len(r["pairs"]) for r in res], np.int32), pair_off=off,
               **{key: np.array([r[key] for r in res], np.int32) for key in _PROBLEM_KEYS},
               nm_layer_med=np.array([r["nm_layer_med"] for r in res], np.float32),
               nm_layer_p90=np.array([r["nm_layer_p90"] for r in res], np.float32),
               pos_tok=pos, pos_tok_off=pos_off, neg_tok=neg, neg_tok_off=neg_off,
               layer=np.array([pr["layer"] for pr in prs], np.int32),
               neg_kind=np.array([pr["neg_kind"] for pr in prs], np.int8),
               **{key: np.array([pr[key] for pr in prs], np.float32)
                  for key in ("pos_b", "pos_y", "pos_dom_b", "pos_dom_y", "neg_b", "neg_y")})
    shards.save(out_path, **out)
    return out_path, len(res), len(prs), time.time() - t0


def run(raw_dir: str, fit_dir: str, out_dir: str, *, k_spec: str, pairs: int = 8, workers: int | None = None,
        part: tuple[int, int] = (0, 1)) -> None:
    """Build the negative shards of one K spec for every fitted shard in ``fit_dir`` not yet built in ``out_dir``.
    Resumable; ``part = (i, n)`` as in :func:`flash_ansr.datagen.fit.run`."""
    from multiprocessing import get_context
    from flash_ansr.utils.cpus import available_cores
    PR.k_sampler(k_spec)
    os.makedirs(out_dir, exist_ok=True)
    fits = sorted(p for p in glob.glob(os.path.join(fit_dir, "*.npz")) if not p.endswith(".part.npz"))[part[0]::part[1]]
    jobs = [(os.path.join(raw_dir, os.path.basename(f)), f, os.path.join(out_dir, os.path.basename(f)), k_spec, pairs)
            for f in fits]
    jobs = [j for j in jobs if not os.path.exists(j[2])]
    workers = workers or available_cores()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    print(f"[negatives] {k_spec}: {len(jobs)} shards with {workers} workers (part {part})", flush=True)
    t0, n_in, n_pairs = time.time(), 0, 0
    with get_context("spawn").Pool(workers) as pool:
        for i, (path, n, c, dt) in enumerate(pool.imap_unordered(_negatives_shard, jobs)):
            if n > 0:
                n_in, n_pairs = n_in + n, n_pairs + c
            print(f"[negatives] {i + 1}/{len(jobs)} {os.path.basename(path)}: {c} pairs from {n} problems, {dt:.1f} s "
                  f"| total {n_pairs} pairs / {n_in} problems in {(time.time() - t0) / 60:.1f} min", flush=True)
