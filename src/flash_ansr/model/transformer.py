from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_ansr.model.generic import get_norm_layer, FeedForward, PositionalEncoding


class RotaryEmbedding(nn.Module):
    """Implements Rotary Positional Embeddings (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len, device=self.inv_freq.device)  # type: ignore
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),  # type: ignore
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),  # type: ignore
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary embeddings to query and key tensors."""
    # cos, sin have shape (1, 1, seq_len, head_dim)
    # xq, xk have shape (batch, n_heads, seq_len, head_dim)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Decoder attention with RoPE and causal masking support."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, use_rope: bool = False):
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
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Get batch sizes for query and key_value independently
        batch_size_q, seq_len_q, _ = query.shape
        batch_size_kv, seq_len_kv, _ = key_value.shape

        # Project query, key, and value
        q = self.w_q(query)
        k = self.w_k(key_value)
        v = self.w_v(key_value)

        # Reshape for multi-head attention using their respective batch sizes
        q = q.view(batch_size_q, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size_kv, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size_kv, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if rope_emb is None:
                raise ValueError("rope_emb must be provided when use_rope is True")
            cos, sin = rope_emb
            q, k = apply_rotary_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size_q, seq_len_q, -1)
        return self.w_o(attn_output)


class TransformerDecoderBlock(nn.Module):
    """A single block of the Transformer Decoder with configurable norms.

    Parameters
    ----------
    dim : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    ffn_hidden_dim : int | None
        Hidden dimension of FFN (defaults to 4 * dim if None).
    dropout : float
        Dropout probability.
    use_checkpointing : bool
        Whether to use gradient checkpointing.
    self_attn_norm_type : str
        Norm type for self-attention input ("rms", "layer", "none").
    cross_attn_norm_type : str
        Norm type for cross-attention input.
    ffn_norm_type : str
        Norm type for FFN input.
    """
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
        rope_emb: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        normed_x = self.self_attn_norm(x)
        x = x + self.self_attention(
            normed_x, normed_x,
            rope_emb=rope_emb, is_causal=True
        )
        x = x + self.cross_attention(
            self.cross_attn_norm(x), encoder_memory
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.training and self.use_checkpointing:
            # Re-define ckpt_fn to be compatible with checkpoint's argument handling
            def ckpt_fn(x: torch.Tensor, encoder_memory: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None) -> torch.Tensor:
                if cos is None or sin is None:
                    rope_emb = None
                else:
                    rope_emb = (cos, sin)
                return self._forward(x, encoder_memory, rope_emb)
            if rope_emb is None:
                cos, sin = None, None
            else:
                cos, sin = rope_emb
            return checkpoint(ckpt_fn, x, encoder_memory, cos, sin, use_reentrant=False)
        else:
            return self._forward(x, encoder_memory, rope_emb)


class TransformerDecoder(nn.Module):
    """State-of-the-art Transformer Decoder with configurable norms.

    Added parameters:
    cross_attn_kv_norm_type : str
        Norm type applied to encoder memory prior to cross-attention ("rms", "layer", "none").
    output_norm_type : str
        Norm type applied before the output projection.
    block_self_attn_norm_type / block_cross_attn_norm_type / block_ffn_norm_type : str
        Passed to each TransformerDecoderBlock.
    """
    def __init__(
        self,
        vocab_size: int,
        input_dim: int | None,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int = 4096,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        block_self_attn_norm_type: str = "rms",
        block_cross_attn_norm_type: str = "rms",
        block_ffn_norm_type: str = "rms",
        cross_attn_kv_norm_type: str = "rms",
        output_norm_type: str = "rms",
        use_checkpointing: bool = False,
        use_rope_self_attn: bool = False,
        use_rope_cross_attn: bool = False,
        use_sinusoidal_pos_emb: bool = True,
    ):
        super().__init__()
        head_dim = model_dim // n_heads
        self.tok_embeddings = nn.Embedding(vocab_size, model_dim)

        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len) if (use_rope_self_attn or use_rope_cross_attn) else None
        self.pos_encoding = PositionalEncoding() if use_sinusoidal_pos_emb else nn.Identity()

        self.cross_attn_kv_proj: nn.Module
        if input_dim is not None and input_dim != model_dim:
            self.cross_attn_kv_proj = nn.Linear(input_dim, model_dim)
        else:
            self.cross_attn_kv_proj = nn.Identity()

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                dim=model_dim,
                n_heads=n_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                use_checkpointing=use_checkpointing,
                use_rope_self_attn=use_rope_self_attn,
                use_rope_cross_attn=use_rope_cross_attn,
                self_attn_norm_type=block_self_attn_norm_type,
                cross_attn_norm_type=block_cross_attn_norm_type,
                ffn_norm_type=block_ffn_norm_type,
            ) for _ in range(n_layers)
        ])

        self.cross_attn_kv_norm = get_norm_layer(cross_attn_kv_norm_type, model_dim)
        self.output_norm = get_norm_layer(output_norm_type, model_dim)
        self.output = nn.Linear(model_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, encoder_memory: torch.Tensor, extra_parallel_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = tokens.shape[1]
        h = self.tok_embeddings(tokens)
        h = h + self.pos_encoding(h)

        if extra_parallel_embeddings is not None:
            h = h + extra_parallel_embeddings

        if self.rope is not None:
            rope_emb = self.rope(h, seq_len=seq_len)
        else:
            rope_emb = None
        encoder_memory = self.cross_attn_kv_proj(encoder_memory)
        encoder_memory = self.cross_attn_kv_norm(encoder_memory)

        for layer in self.layers:
            h = layer(h, encoder_memory, rope_emb)

        h = self.output_norm(h)
        return self.output(h)
