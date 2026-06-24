"""Static-shape, position-indexed KV cache for graph-capturable decode.

The default incremental decode (`use_cache=True`) GROWS the self-attention cache by `torch.cat` every
step and SHRINKS the batch to active rows by `index` gather -- both are dynamic shapes that (a) cost
~22% of the 1B decode wall (gather 1.54s + cat 0.31s, measured 2026-06-19) and (b) defeat CUDA-graph
capture (the graph re-records per decode length; `torch.compile(reduce-overhead)` measured 0.92x).

This module replaces that with a FULL-WIDTH, POSITION-INDEXED cache: per-layer self-attention K/V live
in a preallocated `(batch, n_heads, max_len, head_dim)` buffer, the new token's K/V are written
IN-PLACE at slot `position`, and attention reads the full buffer under an explicit causal mask (slots
> position masked). Cross-attention K/V (from the static encoder memory) are computed once and reused.
Shapes are then identical every step -> one CUDA graph captures the per-step forward (Stage 2).

Scope: the deployed v23.0 decoder path only -- pre-norm, RoPE on self-attention (XSA supported). The
static decode is gated by the model-config capability check (see the build
plan `experimental/eval/quantization/DECODE_STATIC_GRAPH_PLAN.md`). Quality bar = logits-allclose to
the dynamic cat-grow path at atol 1e-5 (Stage-1 gate).
"""
from __future__ import annotations

import torch


class StaticKVCache:
    """Preallocated position-indexed KV cache for one decode chunk (fixed `batch` width).

    Self-attention: `sa_k[i]`, `sa_v[i]` are `(batch, n_heads, max_len, head_dim)`, written in-place at
    `position`. Cross-attention: `ca[i]` is a 1-element holder `[None]` populated on the first forward
    with `(k, v)` of shape `(1 or batch, n_heads, mem_len, head_dim)` and reused thereafter. `position`
    is the slot for the NEXT write (advanced by the decode loop, not here).
    """

    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.n_layers = n_layers
        self.batch = batch
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.sa_k = [torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype) for _ in range(n_layers)]
        self.sa_v = [torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype) for _ in range(n_layers)]
        # cross-attn K/V holders: [None] until the first forward computes them from encoder memory.
        self.ca: list[list] = [[None] for _ in range(n_layers)]
        self.position = 0

    def reset(self) -> None:
        """Zero the self-attn buffers + cross-attn holders + position (reuse across chunks; a CUDA
        graph records fixed buffer ADDRESSES, so we reset-in-place rather than realloc)."""
        for k, v in zip(self.sa_k, self.sa_v):
            k.zero_()
            v.zero_()
        for holder in self.ca:
            holder[0] = None
        self.position = 0

    def seed_from_dynamic(self, past_key_values: list) -> None:
        """Copy a dynamic prefill cache (list of `((sa_k, sa_v), (ca_k, ca_v))` per layer) into the
        static buffers. `sa_*` of length L are written to slots [0:L]; `position` is set to L (the next
        free slot). Cross-attn K/V are stored as-is (reused, never written again)."""
        prefill_len = past_key_values[0][0][0].shape[2]
        for i, ((sa_k, sa_v), (ca_k, ca_v)) in enumerate(past_key_values):
            self.sa_k[i][:, :, :prefill_len, :] = sa_k
            self.sa_v[i][:, :, :prefill_len, :] = sa_v
            self.ca[i][0] = (ca_k, ca_v)
        self.position = prefill_len

    def attend_mask(self, position: int) -> torch.Tensor:
        """Boolean self-attn mask `(1, 1, 1, max_len)` for a single query at `position`: True (attend)
        for key slots [0:position+1], False beyond. SDPA boolean mask: True = participate."""
        m = torch.zeros(1, 1, 1, self.max_len, device=self.device, dtype=torch.bool)
        m[:, :, :, : position + 1] = True
        return m
