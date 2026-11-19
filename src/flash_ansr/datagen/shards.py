"""File formats of the split pipeline: one compressed ``.npz`` per shard of problems, written atomically.

Ragged per-problem arrays are stored CSR-style (a flat array plus ``*_off`` offsets of length ``n + 1``); 2-D
per-problem arrays additionally store their column count. Strings are fixed-width numpy unicode arrays.

Raw shard (stage 1 -> stage 2), keys:
    problem_id [P], k [P], skeleton [P] (space-joined prefix), noise_level [P]
    x, x_off, x_cols          support inputs (rows of all problems concatenated, float64)
    y_noisy, y, y_off         support targets
    x_val, x_val_off, x_val_cols, y_val, y_val_off   clean validation rows
    draw_off [P + 1]          problem p's draws are draws draw_off[p]:draw_off[p+1]
    tok, tok_off              raw token sequences (uint16, prompt prefix included, trimmed after <eos>)
    log_prob [n_draws]        summed token log-likelihood of each draw (float32)

Fitted shard (stage 2 -> targets): the raw shard's identifiers (problem_id, k, skeleton, noise_level, draw_off; the
problem data stay in the raw shard, aligned by position) and, per problem, the kept candidates (fresh-point FVU < 1):
    n_raw [P], n_distinct [P], n_fitted [P], n_kept [P], cand_off [P + 1]
    fam_tok, fam_tok_off      per kept candidate: its beam, i.e. the raw emission of the first draw of its dedup
                              class, prompt prefix included
    fam_id [n_cand]           kept-family index within the problem (the rows sharing a draw's log-prob: its fit,
                              ladder variants and typed-thaw re-fits)
    fvu, fvu_val, mdl, score  support FVU, fresh FVU (clean validation rows), price (milli-bits), ranking score
    n_constants, is_variant, log_prob_draw
and, per raw draw (aligned with the raw shard's draws through ``draw_off``):
    draw_status               0 ok, 1 no expression, 2 malformed span, 3 invalid, 4 non-finite simplification,
                              5 representative not in the ledger (``fit.STATUS_NAMES``)
    draw_act                  index of the draw's distinct raw expression within the problem (-1: none)
    draw_fam                  the kept family the draw collapsed into (-1: none kept)

Target shard (stage 3 -> training): see :mod:`flash_ansr.datagen.targets`; no problem data either.
"""
from __future__ import annotations

import os
from typing import Any, Iterator, Sequence

import numpy as np


def pack(arrays: Sequence[np.ndarray], dtype: Any) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ragged arrays along axis 0 -> (flat, offsets)."""
    off = np.zeros(len(arrays) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(a) for a in arrays])
    flat = np.concatenate([np.asarray(a, dtype=dtype) for a in arrays]) if len(arrays) and off[-1] else np.zeros((0,), dtype)
    return flat, off


def pack2d(arrays: Sequence[np.ndarray], dtype: Any = np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-concatenate 2-D arrays of varying width -> (flat, row offsets, column counts)."""
    cols = np.array([a.shape[1] if a.ndim == 2 else 1 for a in arrays], dtype=np.int32)
    flat_rows = [np.asarray(a, dtype=dtype).reshape(-1) for a in arrays]
    flat, off = pack(flat_rows, dtype)
    return flat, off, cols


def unpack(flat: np.ndarray, off: np.ndarray, i: int) -> np.ndarray:
    return flat[off[i]:off[i + 1]]


def unpack2d(flat: np.ndarray, off: np.ndarray, cols: np.ndarray, i: int) -> np.ndarray:
    a = flat[off[i]:off[i + 1]]
    return a.reshape(-1, int(cols[i]))


def save(path: str, **arrays: np.ndarray) -> None:
    """Atomic ``np.savez_compressed``: write to a temporary file, then rename (a crash never leaves a torn shard)."""
    tmp = f"{path}.part.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_raw_shard(path: str, problems: list, draws: list[tuple[list[list[int]], list[float]]]) -> None:
    """``problems``: :class:`flash_ansr.datagen.problems.Problem`; ``draws[p]``: (token lists, log-probs)."""
    x, x_off, x_cols = pack2d([p.x for p in problems])
    x_val, x_val_off, x_val_cols = pack2d([p.x_val for p in problems])
    y_noisy, y_off = pack([p.y_noisy for p in problems], np.float64)
    y, _ = pack([p.y for p in problems], np.float64)
    y_val, y_val_off = pack([p.y_val for p in problems], np.float64)
    seqs = [np.asarray(s, np.uint16) for toks, _ in draws for s in toks]
    tok, tok_off = pack(seqs, np.uint16)
    log_prob = np.concatenate([np.asarray(lp, np.float32) for _, lp in draws]) if seqs else np.zeros(0, np.float32)
    draw_off = np.zeros(len(problems) + 1, np.int64)
    draw_off[1:] = np.cumsum([len(toks) for toks, _ in draws])
    save(path, problem_id=np.array([p.problem_id for p in problems]), k=np.array([p.k for p in problems], np.int32),
         skeleton=np.array([" ".join(p.skeleton) for p in problems]),
         noise_level=np.array([float((p.noise or {}).get("level", 0.0) or 0.0) for p in problems]),
         x=x, x_off=x_off, x_cols=x_cols, y_noisy=y_noisy, y=y, y_off=y_off,
         x_val=x_val, x_val_off=x_val_off, x_val_cols=x_val_cols, y_val=y_val, y_val_off=y_val_off,
         draw_off=draw_off, tok=tok, tok_off=tok_off, log_prob=log_prob)


def iter_raw_shard(path: str) -> Iterator[dict[str, Any]]:
    """Yield one dict per problem of a raw shard."""
    z = load(path)
    for p in range(len(z["problem_id"])):
        d0, d1 = int(z["draw_off"][p]), int(z["draw_off"][p + 1])
        yield dict(problem_id=str(z["problem_id"][p]), k=int(z["k"][p]), skeleton=str(z["skeleton"][p]),
                   noise_level=float(z["noise_level"][p]),
                   x=unpack2d(z["x"], z["x_off"], z["x_cols"], p), y_noisy=unpack(z["y_noisy"], z["y_off"], p),
                   y=unpack(z["y"], z["y_off"], p), x_val=unpack2d(z["x_val"], z["x_val_off"], z["x_val_cols"], p),
                   y_val=unpack(z["y_val"], z["y_val_off"], p),
                   tokens=[z["tok"][z["tok_off"][j]:z["tok_off"][j + 1]].astype(np.int64).tolist() for j in range(d0, d1)],
                   log_prob=z["log_prob"][d0:d1].astype(float).tolist())
