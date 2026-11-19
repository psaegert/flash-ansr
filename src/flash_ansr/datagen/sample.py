"""Stage 1 (GPU nodes): raw draws for many problems at once.

Packing: problems with the same pool size K are decoded together, ``P`` problems x ``K`` rows in one batch. The
encoder runs once per problem; the prefix is prefilled once per problem (batch ``P``) and expanded to the rows; the
cross-attention reads each problem's own memory through the grouped path of
:meth:`MultiHeadAttention.forward_static_cross` (no per-row copy of the memory K/V); self-attention reads only the
written slots (``StaticKVCache.live_slice``). Sampling is plain softmax sampling at temperature 1 (the deployed
``top_k=0, top_p=1``), scored with the raw log-softmax, exactly as the static sampler scores. No simplification here:
raw sequences go to the CPU stage (:mod:`flash_ansr.datagen.fit`).

``verify`` checks a pack against the library: the accumulated score of every sampled row must equal the
teacher-forced log-likelihood of the same row under :meth:`FlashANSRModel.forward` (dynamic, per problem).
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from flash_ansr.datagen import problems as PR
from flash_ansr.datagen import shards


@torch.no_grad()
def decode_pack(m: Any, memory: torch.Tensor, rows: int, prompt_prefix: Any, max_len: int,
                generator: torch.Generator | None = None, check_every: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``rows`` sequences for each of the ``P = memory.shape[0]`` problems.

    Returns ``(sequences (P*rows, max_len) long, scores (P*rows,) float)``; rows are problem-major. Rows after
    ``<eos>`` hold ``<pad>``."""
    from flash_ansr.model.decoders.static_kv import StaticKVCache
    from flash_ansr.model.flash_ansr_model import NUMERIC_DTYPE
    device = memory.device
    P = int(memory.shape[0])
    B = P * rows
    tokens0, num0 = m._resolve_generation_prefix(prompt_prefix=prompt_prefix, initial_tokens=None, input_num=None)
    L0 = len(tokens0)
    pad, eos = int(m.tokenizer["<pad>"]), int(m.tokenizer["<eos>"])
    numeric = None
    if num0 is not None:
        numeric = torch.full((max_len,), float("nan"), device=device, dtype=NUMERIC_DTYPE)
        numeric[:len(num0)] = torch.tensor(num0, device=device, dtype=NUMERIC_DTYPE)
    prefix = torch.tensor(tokens0, device=device, dtype=torch.long).unsqueeze(0).expand(P, -1).contiguous()
    pre_num = None if numeric is None else numeric[:L0].view(1, L0, 1).expand(P, -1, -1).contiguous()
    logits_pre, cache = m.forward(prefix, None, input_num=pre_num, memory=memory, use_cache=True)
    block0 = m.decoder.layers[0]
    sc = StaticKVCache(n_layers=len(m.decoder.layers), batch=B, n_heads=block0.self_attention.n_heads,
                       head_dim=block0.self_attention.head_dim, max_len=max_len + m.decoder.prefix_len,
                       device=device, dtype=logits_pre.dtype)
    for i, ((sk, sv), (ck, cv)) in enumerate(cache):
        n = sk.shape[2]
        sc.sa_k[i][:, :, :n] = sk.repeat_interleave(rows, dim=0)
        sc.sa_v[i][:, :, :n] = sv.repeat_interleave(rows, dim=0)
        sc.ca[i][0] = (ck, cv)                      # grouped: one K/V per problem
    sc.position = cache[0][0][0].shape[2]
    sc.live_slice = True
    del cache
    seqs = torch.full((B, max_len), pad, device=device, dtype=torch.long)
    seqs[:, :L0] = prefix[0]
    scores = torch.zeros(B, device=device, dtype=torch.float32)
    finished = torch.zeros(B, device=device, dtype=torch.bool)
    next_logits = logits_pre[:, -1, :].float().repeat_interleave(rows, dim=0)
    for t in range(L0, max_len):
        logp = torch.log_softmax(next_logits.float(), dim=-1)
        sampled = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        sampled = torch.where(finished, torch.full_like(sampled, pad), sampled)
        seqs[:, t] = sampled
        scores += torch.where(finished, torch.zeros_like(scores), logp.gather(1, sampled.unsqueeze(-1)).squeeze(-1))
        finished |= sampled == eos
        if t == max_len - 1 or ((t - L0) % check_every == 0 and bool(finished.all())):
            break
        num_step = None if numeric is None else numeric[t].expand(B).reshape(B, 1, 1)
        next_logits = m.forward_static(seqs[:, t:t + 1], num_step, memory, sc, position=t)[:, -1, :]
    return seqs, scores


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


def run(model_dir: str, out_dir: str, *, n_problems: int, seed: int, k_spec: str = "log2uniform:1:512",
        max_len: int = 128, shard_problems: int = 256, rows_target: int = 4096, device: str = "cuda",
        amp: str = "tf32", verify_every: int = 0) -> None:
    """Draw ``n_problems`` problems (seed ``seed``), sample their pools packed by K, write raw shards to ``out_dir``.

    Resumable at shard granularity: shards already on disk are skipped (the problem stream is replayed)."""
    from flash_ansr import FlashANSR
    os.makedirs(out_dir, exist_ok=True)
    if amp in ("tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    w = FlashANSR.load(directory=model_dir).to(device)
    m = w.flash_ansr_model.eval()
    pp = w._prepare_prompt_prefix(emission="fittable", complexity=None)
    eos, pad = int(m.tokenizer["<eos>"]), int(m.tokenizer["<pad>"])
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" else nullcontext()
    stream = PR.problem_stream(model_dir, seed, k_spec=k_spec)
    t0, done = time.time(), 0
    for s in range(0, n_problems, shard_problems):
        path = os.path.join(out_dir, f"raw_s{seed}_{s // shard_problems:06d}.npz")
        batch = [next(stream) for _ in range(min(shard_problems, n_problems - s))]
        if os.path.exists(path):
            continue
        draws: dict[int, tuple[list, list]] = {}
        by_k: dict[int, list[int]] = {}
        for i, p in enumerate(batch):
            by_k.setdefault(p.k, []).append(i)
        for k, idx in by_k.items():
            per_pack = max(1, rows_target // k)
            for j in range(0, len(idx), per_pack):
                grp = idx[j:j + per_pack]
                with torch.no_grad(), autocast:
                    mem = torch.cat([m._create_memory(data_tensor(w, batch[i].x, batch[i].y_noisy, device)) for i in grp], 0)
                    seqs, sc = decode_pack(m, mem, k, pp, max_len, generator=gen)
                if verify_every and (s // shard_problems) % verify_every == 0 and j == 0:
                    err = verify(m, mem, seqs, sc, k, pp)
                    print(f"[sample] verify K={k} P={len(grp)}: max |score - teacher-forced log p| = {err:.2e} nats", flush=True)
                S, C = seqs.cpu().tolist(), sc.cpu().tolist()
                for g, i in enumerate(grp):
                    rr = range(g * k, (g + 1) * k)
                    draws[i] = ([trim(S[r], eos, pad) for r in rr], [C[r] for r in rr])
        shards.write_raw_shard(path, batch, [draws[i] for i in range(len(batch))])
        done += len(batch)
        el = time.time() - t0
        print(f"[sample] {os.path.basename(path)}: {len(batch)} problems, {sum(len(d[0]) for d in draws.values())} draws | "
              f"{done} in {el / 60:.1f} min = {done / max(el, 1e-9):.2f} problems/s", flush=True)
