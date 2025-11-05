"""Decoder-specific building blocks, including attention and positional encodings."""
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_ansr.model.common import FeedForward, get_norm_layer


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) cache."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        device = inv_freq.device
        t = torch.arange(self.max_seq_len, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        cos_cached = self.cos_cached
        sin_cached = self.sin_cached
        if not isinstance(cos_cached, torch.Tensor) or not isinstance(sin_cached, torch.Tensor):
            raise RuntimeError("RotaryEmbedding buffers not initialised as tensors")
        return (
            cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention layer with optional rotary embeddings."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        rope_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size_q, seq_len_q, _ = query.shape
        batch_size_kv, seq_len_kv, _ = key_value.shape

        q = self.w_q(query)
        k = self.w_k(key_value)
        v = self.w_v(key_value)

        q = q.view(batch_size_q, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size_kv, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size_kv, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = rope_emb
            q, k = apply_rotary_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size_q, seq_len_q, -1)
        return self.w_o(attn_output)


class TransformerDecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with optional RoPE and checkpointing."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
        use_rope_self_attn: bool = False,
        use_rope_cross_attn: bool = False,
        self_attn_norm_type: str = "rms",
        cross_attn_norm_type: str = "rms",
        ffn_norm_type: str = "rms",
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.self_attn_norm = get_norm_layer(self_attn_norm_type, dim)
        self.self_attention = Attention(dim=dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope_self_attn)

        self.cross_attn_norm = get_norm_layer(cross_attn_norm_type, dim)
        self.cross_attention = Attention(dim=dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope_cross_attn)

        self.ffn_norm = get_norm_layer(ffn_norm_type, dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def _forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        rope_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normed_x = self.self_attn_norm(x)
        x = x + self.self_attention(normed_x, normed_x, rope_emb=rope_emb, is_causal=True)
        x = x + self.cross_attention(self.cross_attn_norm(x), encoder_memory, rope_emb=rope_emb)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        rope_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.training and self.use_checkpointing:
            cos, sin = rope_emb

            def ckpt_fn(
                x: torch.Tensor,
                encoder_memory: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
            ) -> torch.Tensor:
                return self._forward(x, encoder_memory, (cos, sin))

            return checkpoint(ckpt_fn, x, encoder_memory, cos, sin, use_reentrant=False)

        return self._forward(x, encoder_memory, rope_emb)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding utility for experiments and ablations."""

    def __init__(self) -> None:
        super().__init__()
        self.seq_len: int | None = None
        self.input_size: int | None = None
        self.encoding: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor | None = None,
        shape: tuple[int, int] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if shape is not None and device is not None:
            T, E = shape
        elif x is not None:
            if len(x.shape) < 3:
                x = x.unsqueeze(0)
            _, T, E = x.shape
            device = x.device
        else:
            raise ValueError("Either X or shape and device must be provided")

        E_compat = E + E % 2

        current_encoding = self.encoding
        current_device = current_encoding.device if current_encoding is not None else None

        if self.seq_len is None or (
            T,
            E_compat,
            device,
        ) != (
            self.seq_len,
            self.input_size,
            current_device,
        ):
            self.seq_len = T
            self.input_size = E_compat
            self.encoding = torch.zeros((T, E_compat), device=device)

            t = 1 / 10000 ** (torch.arange(0, E_compat, 2) / E_compat)
            k = torch.arange(T, device=device)
            v = torch.outer(k, t)

            self.encoding[:, 0::2] = v.sin()
            self.encoding[:, 1::2] = v.cos()

        if self.encoding is None:
            raise RuntimeError("Positional encoding buffer not initialised")

        return self.encoding[:, :E]
