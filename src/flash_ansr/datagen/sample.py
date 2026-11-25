"""Stage 1 (GPU nodes): raw draws for many problems at once.

Packing: one decode holds ``G`` groups of ``R`` rows each (``R`` = ``chunk_rows``). A problem with pool size K fills
ceil(K / R) groups: its K draws are the first K rows of those groups, and the unused rows of its last group are
sampled and discarded (rows are i.i.d., so dropping them leaves the K draws exact; the waste is under R rows per
problem, ~6% of rows at R = 32 for K uniform on 1..512). Groups of different problems pack together, so any K
distribution fills the GPU. For the T23 data every problem gets K = 512 (the training K is assigned later, as the
first K draws, in :mod:`flash_ansr.datagen.targets`), so groups are always full; small groups (R = 8) finish early
and are dropped by ``decode_pack``'s compaction. With ``chunk_rows=0`` the older mode applies: equal-K problems only, R = K (no waste, but
packs only when K values repeat, e.g. powers of two). The encoder runs once per problem; the prefix is prefilled once
per group and expanded to its rows; the cross-attention reads each group's memory through the grouped path of
:meth:`Attention.forward_static_cross` (no per-row copy of the memory K/V); self-attention reads only the written
slots (``StaticKVCache.live_slice``). Sampling is plain softmax sampling at temperature 1 (the deployed
``top_k=0, top_p=1``), scored with the raw log-softmax, exactly as the static sampler scores. No simplification here:
raw sequences go to the CPU stage (:mod:`flash_ansr.datagen.fit`).

``verify`` checks a pack against the library: the accumulated score of every sampled row must equal the
teacher-forced log-likelihood of the same row under :meth:`FlashANSRModel.forward` (dynamic, per problem).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from flash_ansr.datagen import problems as PR
from flash_ansr.datagen import shards


@torch.no_grad()
def decode_pack(m: Any, memory: torch.Tensor, rows: int, prompt_prefix: Any, max_len: int,
                generator: torch.Generator | None = None, check_every: int = 8, compact: bool = True,
                compact_frac: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``rows`` sequences for each of the ``G = memory.shape[0]`` groups (one memory per group).

    Returns ``(sequences (G*rows, max_len) long, scores (G*rows,) float)``; rows are group-major. Rows after
    ``<eos>`` hold ``<pad>``.

    Compaction (``compact``): every ``check_every`` steps, once at least ``compact_frac`` of the active groups have
    finished every row, those groups are written out and dropped. The kept rows move to the front of the existing
    cache buffers (in place, one layer's temporary at a time), so later steps compute only unfinished groups. A
    group is the unit because the grouped cross-attention needs equal rows per group."""
    from flash_ansr.model.decoders.static_kv import StaticKVCache
    from flash_ansr.model.flash_ansr_model import NUMERIC_DTYPE
    device = memory.device
    G = int(memory.shape[0])
    B = G * rows
    tokens0, num0 = m._resolve_generation_prefix(prompt_prefix=prompt_prefix, initial_tokens=None, input_num=None)
    L0 = len(tokens0)
    pad, eos = int(m.tokenizer["<pad>"]), int(m.tokenizer["<eos>"])
    numeric = None
    if num0 is not None:
        numeric = torch.full((max_len,), float("nan"), device=device, dtype=NUMERIC_DTYPE)
        numeric[:len(num0)] = torch.tensor(num0, device=device, dtype=NUMERIC_DTYPE)
    prefix = torch.tensor(tokens0, device=device, dtype=torch.long).unsqueeze(0).expand(G, -1).contiguous()
    pre_num = None if numeric is None else numeric[:L0].view(1, L0, 1).expand(G, -1, -1).contiguous()
    logits_pre, cache = m.forward(prefix, None, input_num=pre_num, memory=memory, use_cache=True)
    block0 = m.decoder.layers[0]
    # The cache takes the dtype the layers produce K/V in (bf16 under autocast), not the logits' dtype: the output head
    # returns fp32 logits even under autocast, and an fp32 buffer would store the same bf16 values at twice the memory.
    sc = StaticKVCache(n_layers=len(m.decoder.layers), batch=B, n_heads=block0.self_attention.n_heads,
                       head_dim=block0.self_attention.head_dim, max_len=max_len + m.decoder.prefix_len,
                       device=device, dtype=cache[0][0][0].dtype)
    for i, ((sk, sv), (ck, cv)) in enumerate(cache):
        n = sk.shape[2]
        sc.sa_k[i][:, :, :n] = sk.repeat_interleave(rows, dim=0)
        sc.sa_v[i][:, :, :n] = sv.repeat_interleave(rows, dim=0)
        sc.ca[i][0] = (ck, cv)                      # grouped: one K/V per group
    sc.position = cache[0][0][0].shape[2]
    sc.live_slice = True
    del cache
    out_seqs = torch.full((B, max_len), pad, device=device, dtype=torch.long)
    out_scores = torch.zeros(B, device=device, dtype=torch.float32)
    seqs = torch.full((B, max_len), pad, device=device, dtype=torch.long)
    seqs[:, :L0] = prefix[0]
    scores = torch.zeros(B, device=device, dtype=torch.float32)
    finished = torch.zeros(B, device=device, dtype=torch.bool)
    row_id = torch.arange(B, device=device)          # original row of each active row
    next_logits = logits_pre[:, -1, :].float().repeat_interleave(rows, dim=0)
    for t in range(L0, max_len):
        logp = torch.log_softmax(next_logits.float(), dim=-1)
        sampled = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        sampled = torch.where(finished, torch.full_like(sampled, pad), sampled)
        seqs[:, t] = sampled
        scores += torch.where(finished, torch.zeros_like(scores), logp.gather(1, sampled.unsqueeze(-1)).squeeze(-1))
        finished |= sampled == eos
        if t == max_len - 1:
            break
        if (t - L0) % check_every == 0:
            if bool(finished.all()):
                break
            if compact:
                gdone = finished.view(-1, rows).all(1)
                n_done = int(gdone.sum())
                if n_done and n_done >= compact_frac * gdone.numel():
                    done_rows = gdone.repeat_interleave(rows)
                    out_seqs[row_id[done_rows]] = seqs[done_rows]
                    out_scores[row_id[done_rows]] = scores[done_rows]
                    kg = (~gdone).nonzero().squeeze(1)
                    kr = (~done_rows).nonzero().squeeze(1)
                    nb = int(kr.numel())
                    w = t + m.decoder.prefix_len + 1                # slots written so far (data prefix in front)
                    for i in range(len(sc.sa_k)):
                        sc.sa_k[i][:nb, :, :w] = sc.sa_k[i][kr, :, :w]
                        sc.sa_v[i][:nb, :, :w] = sc.sa_v[i][kr, :, :w]
                        sc.sa_k[i], sc.sa_v[i] = sc.sa_k[i][:nb], sc.sa_v[i][:nb]
                        ck, cv = sc.ca[i][0]
                        if ck.shape[0] > 1:
                            sc.ca[i][0] = (ck[kg], cv[kg])
                    if memory.shape[0] > 1:
                        memory = memory[kg]
                    seqs, scores, finished, row_id = seqs[kr], scores[kr], finished[kr], row_id[kr]
        b = int(seqs.shape[0])
        num_step = None if numeric is None else numeric[t].expand(b).reshape(b, 1, 1)
        next_logits = m.forward_static(seqs[:, t:t + 1], num_step, memory, sc, position=t)[:, -1, :]
    out_seqs[row_id] = seqs
    out_scores[row_id] = scores
    return out_seqs, out_scores


def trim(row: list[int], eos: int, pad: int) -> list[int]:
    return row[:row.index(eos) + 1] if eos in row else [t for t in row if t != pad]


@torch.no_grad()
def _teacher_forced(m: Any, memory_1: torch.Tensor, rows: list[list[int]], L0: int, num0: Any) -> np.ndarray:
    """Summed log-likelihood of each row's tokens after the prompt under the plain dynamic forward."""
    from flash_ansr.model.flash_ansr_model import NUMERIC_DTYPE
    device = memory_1.device
    pad = int(m.tokenizer["<pad>"])
    L = max(len(r) for r in rows)
    tok = torch.full((len(rows), L), pad, device=device, dtype=torch.long)
    for i, r in enumerate(rows):
        tok[i, :len(r)] = torch.tensor(r, device=device)
    lengths = torch.tensor([len(r) for r in rows], device=device)
    num = None
    if num0 is not None:
        num = torch.full((len(rows), L, 1), float("nan"), device=device, dtype=NUMERIC_DTYPE)
        num[:, :len(num0), 0] = torch.tensor(num0, device=device, dtype=NUMERIC_DTYPE)
    logits = m.forward(tok, None, input_num=num, memory=memory_1.expand(len(rows), -1, -1).contiguous())
    logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(2, tok[:, 1:, None]).squeeze(-1)
    pos = torch.arange(1, L, device=device)[None, :]
    mask = (pos >= L0) & (pos < lengths[:, None])
    return (logp * mask).sum(1).cpu().numpy()


@torch.no_grad()
def verify(m: Any, memory: torch.Tensor, seqs: torch.Tensor, scores: torch.Tensor, rows: int, prompt_prefix: Any,
           n_check: int = 32) -> float:
    """Max |packed score - teacher-forced log p| over the first ``n_check`` rows of every problem (nats)."""
    tokens0, num0 = m._resolve_generation_prefix(prompt_prefix=prompt_prefix, initial_tokens=None, input_num=None)
    eos, pad = int(m.tokenizer["<eos>"]), int(m.tokenizer["<pad>"])
    worst = 0.0
    S = seqs.cpu().tolist()
    for p in range(memory.shape[0]):
        idx = list(range(p * rows, p * rows + min(rows, n_check)))
        rr = [trim(S[i], eos, pad) for i in idx]
        tf = _teacher_forced(m, memory[p:p + 1], rr, len(tokens0), num0)
        worst = max(worst, float(np.max(np.abs(tf - scores[idx].cpu().numpy()))))
    return worst


def data_tensor(m_wrapper: Any, x: np.ndarray, y: np.ndarray, device: str) -> torch.Tensor:
    from flash_ansr.model.flash_ansr_model import NUMERIC_DTYPE
    from flash_ansr.utils.tensor_ops import pad_input_set
    X = torch.tensor(x, dtype=NUMERIC_DTYPE, device=device)
    Y = torch.tensor(y, dtype=NUMERIC_DTYPE, device=device).reshape(-1, 1)
    return torch.cat([pad_input_set(X, m_wrapper.n_variables), Y], -1)


def max_rows_for(k: int, bytes_free: float, per_row: float, per_problem: float) -> int:
    """Largest multiple of ``k`` rows whose self-KV (per row) and grouped cross K/V (per problem) fit."""
    p = int(bytes_free // (k * per_row + per_problem))
    return max(1, p) * k


def _chunks(ks: list[int], chunk_rows: int) -> list[tuple[int, int]]:
    """``(problem index, used rows)`` per group: problem i with pool size k fills ceil(k / R) groups of R rows."""
    return [(i, min(chunk_rows, k - c * chunk_rows)) for i, k in enumerate(ks) for c in range(-(-k // chunk_rows))]


def _produce(prior_dir: str, seed: int, k_spec: str, n_problems: int, shard_problems: int, q: Any) -> None:
    """Problem producer (a spawned process, no CUDA): the serial path's problem stream, in shard-sized batches."""
    try:
        stream = PR.problem_stream(prior_dir, seed, k_spec=k_spec)
        for s in range(0, n_problems, shard_problems):
            q.put([next(stream) for _ in range(min(shard_problems, n_problems - s))])
    except BaseException:
        q.put(traceback.format_exc())
        raise


class _Batches:
    """Shard-sized problem batches in stream order. ``overlap``: a spawned producer process runs ``prefetch`` batches
    ahead of the GPU (a process, not a thread: the decode loop is Python-bound and would contend for the GIL).
    Otherwise inline. Both give identical problems: the same stream, consumed in the same order. The producer is
    spawned, so a calling script must guard its entry point (``if __name__ == "__main__":``), as for any spawn-based
    multiprocessing; the CLI does."""

    def __init__(self, prior_dir: str, seed: int, k_spec: str, n_problems: int, shard_problems: int, overlap: bool,
                 prefetch: int = 2):
        self.proc = None
        if overlap:
            ctx = mp.get_context("spawn")
            self.q = ctx.Queue(maxsize=prefetch)
            self.proc = ctx.Process(target=_produce, args=(prior_dir, seed, k_spec, n_problems, shard_problems, self.q),
                                    daemon=True)
            self.proc.start()
        else:
            self.stream = PR.problem_stream(prior_dir, seed, k_spec=k_spec)

    def next(self, size: int) -> list:
        if self.proc is None:
            return [next(self.stream) for _ in range(size)]
        while True:
            try:
                b = self.q.get(timeout=60)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"problem producer died (exit code {self.proc.exitcode}); a script calling "
                                       f"sample.run must guard its entry point with `if __name__ == '__main__':`")
                continue
            if isinstance(b, str):
                raise RuntimeError(f"problem producer failed:\n{b}")
            assert len(b) == size, (len(b), size)
            return b

    def close(self) -> None:
        if self.proc is not None:
            self.proc.join(timeout=60)
            if self.proc.is_alive():
                self.proc.terminate()


def run(model_dir: str, out_dir: str, *, n_problems: int, seed: int, k_spec: str = "fixed:512",
        max_len: int = 128, shard_problems: int = 256, rows_target: int = 4096, chunk_rows: int = 8,
        device: str = "cuda", amp: str = "tf32", verify_every: int = 0, prior_dir: str | None = None,
        overlap: bool = True) -> None:
    """Draw ``n_problems`` problems (seed ``seed``), sample their pools packed in groups of ``chunk_rows`` rows (0:
    equal-K packing), write raw shards to ``out_dir``.

    ``prior_dir``: the bundle whose training prior (``catalog_train.yaml``, ``dataset_train.yaml``) the problems come
    from; default ``model_dir``. A released bundle ships without those files, so the sampler and the prior can come
    from different directories of the same run.

    ``overlap``: keep the GPU busy while the CPU works: the next shards' problems are generated in a separate process,
    and each finished shard is packed and written in a background thread while the next one decodes. The output is
    identical to the serial path's (same problems, same draws, same files).

    Resumable at shard granularity: shards already on disk are skipped (the problem stream is replayed). Each shard
    samples with its own generator, seeded from (``seed``, shard index), so a resumed run writes the same shards as an
    uninterrupted one. (Until 2026-11-21 one generator served the whole run; a resumed run then reused the random
    numbers of the shards it skipped: valid draws, not reproducible ones.)"""
    from flash_ansr import FlashANSR
    os.makedirs(out_dir, exist_ok=True)
    # started first: the producer's start-up (imports, catalog, first problems; ~50 s on Helix) overlaps model loading
    batches = _Batches(prior_dir or model_dir, seed, k_spec, n_problems, shard_problems, overlap)
    if amp in ("tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    w = FlashANSR.load(directory=model_dir).to(device)
    m = w.flash_ansr_model.eval()
    pp = w._prepare_prompt_prefix(emission="fittable", complexity=None)
    eos, pad = int(m.tokenizer["<eos>"]), int(m.tokenizer["<pad>"])
    gen = torch.Generator(device=device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" else nullcontext()
    if str(device).startswith("cuda") and chunk_rows > 0:
        # cap rows per decode by free memory: the static self-attention cache dominates (fp32 K and V per layer)
        blk = m.decoder.layers[0].self_attention
        elem = 2 if amp == "bf16" else 4                    # the cache takes the K/V dtype (bf16 under autocast)
        per_row = len(m.decoder.layers) * 2 * blk.n_heads * blk.head_dim * (max_len + m.decoder.prefix_len) * elem
        fit_rows = int(0.6 * torch.cuda.mem_get_info()[0] // per_row) // chunk_rows * chunk_rows
        if fit_rows < rows_target:
            print(f"[sample] rows per decode capped {rows_target} -> {fit_rows} by free GPU memory "
                  f"({per_row / 2 ** 20:.1f} MiB of cache per row)", flush=True)
            rows_target = max(chunk_rows, fit_rows)
    writer = ThreadPoolExecutor(max_workers=1) if overlap else None
    pending = None
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
    t0, done, n_rows, n_used = time.time(), 0, 0, 0
    for s in range(0, n_problems, shard_problems):
        path = os.path.join(out_dir, f"raw_s{seed}_{s // shard_problems:06d}.npz")
        ph = dict.fromkeys(("problems", "encode", "decode", "host", "write"), 0.0)   # seconds the GPU loop spends
        tp = time.time()
        batch = batches.next(min(shard_problems, n_problems - s))
        ph["problems"] = time.time() - tp
        if os.path.exists(path):
            continue
        gen.manual_seed(int(np.random.SeedSequence([int(seed), s // shard_problems]).generate_state(1, np.uint64)[0]))
        toks: list[list[list[int]]] = [[] for _ in batch]
        lps: list[list[float]] = [[] for _ in batch]
        if chunk_rows > 0:
            tp = time.time()
            with torch.no_grad(), autocast:
                mems = [m._create_memory(data_tensor(w, p.x, p.y_noisy, device)) for p in batch]
            sync()
            ph["encode"] = time.time() - tp
            plan = [(_chunks([p.k for p in batch], chunk_rows), chunk_rows)]
        else:
            by_k: dict[int, list[int]] = {}
            for i, p in enumerate(batch):
                by_k.setdefault(p.k, []).append(i)
            mems = None
            plan = [([(i, k) for i in idx], k) for k, idx in by_k.items()]
        for groups, rows in plan:
            per_pack = max(1, rows_target // rows)
            for j in range(0, len(groups), per_pack):
                grp = groups[j:j + per_pack]
                tp = time.time()
                with torch.no_grad(), autocast:
                    mem = torch.cat([mems[i] if mems is not None else
                                     m._create_memory(data_tensor(w, batch[i].x, batch[i].y_noisy, device))
                                     for i, _ in grp], 0)
                    seqs, sc = decode_pack(m, mem, rows, pp, max_len, generator=gen)
                if verify_every and (s // shard_problems) % verify_every == 0 and j == 0:
                    err = verify(m, mem, seqs, sc, rows, pp)
                    print(f"[sample] verify R={rows} G={len(grp)}: max |score - teacher-forced log p| = {err:.2e} nats "
                          f"(sampled under {amp}, teacher-forced under TF32)", flush=True)
                seqs, sc = seqs.cpu(), sc.cpu()
                ph["decode"] += time.time() - tp
                tp = time.time()
                S, C = seqs.tolist(), sc.tolist()
                for g, (i, used) in enumerate(grp):
                    toks[i].extend(trim(S[r], eos, pad) for r in range(g * rows, g * rows + used))
                    lps[i].extend(C[r] for r in range(g * rows, g * rows + used))
                ph["host"] += time.time() - tp
                n_rows += len(grp) * rows
                n_used += sum(u for _, u in grp)
        assert all(len(t) == p.k for t, p in zip(toks, batch))
        tp = time.time()
        if writer is None:
            shards.write_raw_shard(path, batch, list(zip(toks, lps)))
        else:                                       # at most one shard in flight; its errors surface here
            if pending is not None:
                pending.result()
            pending = writer.submit(shards.write_raw_shard, path, batch, list(zip(toks, lps)))
        ph["write"] = time.time() - tp
        done += len(batch)
        el = time.time() - t0
        print(f"[sample] {os.path.basename(path)}: {len(batch)} problems, {sum(len(t) for t in toks)} draws | "
              f"{done} in {el / 60:.1f} min = {done / max(el, 1e-9):.2f} problems/s, "
              f"{n_used / max(el, 1e-9):.0f} draws/s, row use {n_used / max(n_rows, 1):.1%} | GPU loop s: "
              + ", ".join(f"{k} {v:.1f}" for k, v in ph.items()), flush=True)
    if pending is not None:
        pending.result()
    if writer is not None:
        writer.shutdown()
    batches.close()
