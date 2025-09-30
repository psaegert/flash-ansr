"""Transformer decoder components with rotary positional embeddings support."""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_ansr.model.generic import get_norm_layer, FeedForward


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) cache.

    Parameters
    ----------
    dim : int
        Size of the rotary embedding dimension. Must be divisible by two.
    max_seq_len : int, optional
        Maximum sequence length to precompute, by default ``4096``.
    base : int, optional
        Base used for frequency computation, by default ``10000``.
    """
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
        """Return sine and cosine caches truncated to ``seq_len``.

        Parameters
        ----------
        x : torch.Tensor
            Reference tensor whose dtype/device determine the returned caches.
        seq_len : int
            Sequence length to slice from the cached rotary embeddings.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Pair of cosine and sine tensors shaped ``(1, 1, seq_len, dim)``.

        Raises
        ------
        ValueError
            If ``seq_len`` exceeds the configured ``max_seq_len``.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),  # type: ignore
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),  # type: ignore
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the final dimension by exchanging its two halves.

    Parameters
    ----------
    x : torch.Tensor
        Tensor whose last dimension is split evenly and rotated.

    Returns
    -------
    torch.Tensor
        Tensor where the first half of the last dimension is replaced with the
        negated second half and vice versa.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Parameters
    ----------
    xq : torch.Tensor
        Query tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
    xk : torch.Tensor
        Key tensor shaped like ``xq``.
    cos : torch.Tensor
        Cosine cache of shape ``(1, 1, seq_len, head_dim)``.
    sin : torch.Tensor
        Sine cache matching ``cos``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors with the same shape as the inputs.
    """
    # cos, sin have shape (1, 1, seq_len, head_dim)
    # xq, xk have shape (batch, n_heads, seq_len, head_dim)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention layer with optional rotary embeddings.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    n_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability applied to attention weights, by default ``0.0``.
    use_rope : bool, optional
        Whether to apply rotary positional embeddings to queries and keys.
    """
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
        rope_emb: tuple[torch.Tensor, torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Compute attention outputs for decoder queries.

        Parameters
        ----------
        query : torch.Tensor
            Decoder query tensor of shape ``(batch_q, seq_q, dim)``.
        key_value : torch.Tensor
            Source tensor providing keys and values, shape ``(batch_kv, seq_kv, dim)``.
        rope_emb : tuple[torch.Tensor, torch.Tensor]
            Precomputed rotary cosine and sine caches.
        is_causal : bool, optional
            If ``True``, apply a causal mask to prevent attending to future tokens.

        Returns
        -------
        torch.Tensor
            Attended representation of shape ``(batch_q, seq_q, dim)``.
        """
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
    """Pre-norm transformer decoder block with optional RoPE and checkpointing.

    Parameters
    ----------
    dim : int
        Model dimensionality for all projections.
    n_heads : int
        Number of attention heads.
    ffn_hidden_dim : int or None, optional
        Hidden size of the feed-forward sub-layer. Defaults to ``4 * dim`` when ``None``.
    dropout : float, optional
        Dropout probability shared across attention and FFN layers.
    use_checkpointing : bool, optional
        Enable PyTorch gradient checkpointing to trade compute for memory.
    use_rope_self_attn : bool, optional
        Apply rotary embeddings within the self-attention component.
    use_rope_cross_attn : bool, optional
        Apply rotary embeddings within the cross-attention component.
    self_attn_norm_type : str, optional
        Normalization type for self-attention input (``"rms"``, ``"layer"`` or ``"none"``).
    cross_attn_norm_type : str, optional
        Normalization type preceding cross-attention.
    ffn_norm_type : str, optional
        Normalization type preceding the feed-forward network.
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
        rope_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Core forward pass used both directly and under checkpointing.

        Parameters
        ----------
        x : torch.Tensor
            Decoder hidden states of shape ``(batch, seq_len, dim)``.
        encoder_memory : torch.Tensor
            Encoder-derived memory to attend over, shape ``(batch, mem_len, dim)``.
        rope_emb : tuple[torch.Tensor, torch.Tensor]
            Rotary cosine and sine caches reused across attention calls.

        Returns
        -------
        torch.Tensor
            Updated decoder states with residual connections applied.
        """
        # Pre-normalize each sub-layer input before integrating via residuals.
        normed_x = self.self_attn_norm(x)
        x = x + self.self_attention(
            normed_x, normed_x,
            rope_emb=rope_emb, is_causal=True
        )
        # Cross-attend to encoder memory once the self-context has been updated.
        x = x + self.cross_attention(
            self.cross_attn_norm(x), encoder_memory, rope_emb=rope_emb
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.

        Parameters
        ----------
        x : torch.Tensor
            Decoder hidden states of shape ``(batch, seq_len, dim)``.
        encoder_memory : torch.Tensor
            Memory produced by the encoder for cross-attention.
        rope_emb : tuple[torch.Tensor, torch.Tensor]
            Rotary embeddings shared across the block.

        Returns
        -------
        torch.Tensor
            Decoder states after applying self-attention, cross-attention, and FFN.
        """
        if self.training and self.use_checkpointing:
            # Re-define ckpt_fn to be compatible with checkpoint's argument handling
            def ckpt_fn(x: torch.Tensor, encoder_memory: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
                return self._forward(x, encoder_memory, (cos, sin))
            cos, sin = rope_emb
            return checkpoint(ckpt_fn, x, encoder_memory, cos, sin, use_reentrant=False)

        return self._forward(x, encoder_memory, rope_emb)


class TransformerDecoder(nn.Module):
    """Configurable transformer decoder stack with rotary embeddings.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary for the embedding layer.
    input_dim : int or None
        Dimension of encoder memory inputs. When different from ``model_dim`` a
        projection layer is inserted.
    model_dim : int
        Hidden dimension used throughout the decoder.
    n_layers : int
        Number of decoder blocks.
    n_heads : int
        Number of attention heads per block.
    max_seq_len : int, optional
        Maximum sequence length supported by rotary embeddings, by default ``4096``.
    ffn_hidden_dim : int or None, optional
        Hidden size of the feed-forward networks inside each block.
    dropout : float, optional
        Dropout probability shared across sub-layers.
    block_self_attn_norm_type : str, optional
        Normalization type for self-attention inputs.
    block_cross_attn_norm_type : str, optional
        Normalization type for cross-attention inputs.
    block_ffn_norm_type : str, optional
        Normalization type for the feed-forward inputs.
    cross_attn_kv_norm_type : str, optional
        Normalization type applied to encoder memory prior to cross-attention.
    output_norm_type : str, optional
        Normalization type applied before the decoder outputs are returned.
    use_checkpointing : bool, optional
        Enable gradient checkpointing inside decoder blocks.
    use_rope_self_attn : bool, optional
        Apply rotary embeddings during self-attention.
    use_rope_cross_attn : bool, optional
        Apply rotary embeddings during cross-attention.
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
    ):
        super().__init__()
        head_dim = model_dim // n_heads
        self.tok_embeddings = nn.Embedding(vocab_size, model_dim)

        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len)

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

    def forward(self, tokens: torch.Tensor, encoder_memory: torch.Tensor, extra_parallel_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        """Generate decoder hidden states conditioned on encoder memory.

        Parameters
        ----------
        tokens : torch.Tensor
            Token indices with shape ``(batch, seq_len)``.
        encoder_memory : torch.Tensor
            Encoder representations to cross-attend over, shape ``(batch, mem_len, input_dim)``.
        extra_parallel_embeddings : torch.Tensor or None, optional
            Additional embeddings summed with token embeddings, e.g. modality inputs.

        Returns
        -------
        torch.Tensor
            Decoder hidden states of shape ``(batch, seq_len, model_dim)``.
        """
        seq_len = tokens.shape[1]
        h = self.tok_embeddings(tokens)

        if extra_parallel_embeddings is not None:
            # Allow callers to inject modality-specific signals alongside token embeddings.
            h = h + extra_parallel_embeddings

        rope_emb = self.rope(h, seq_len=seq_len)
        # Align encoder memory dimensionality and normalize before cross-attention.
        encoder_memory = self.cross_attn_kv_proj(encoder_memory)
        encoder_memory = self.cross_attn_kv_norm(encoder_memory)

        for layer in self.layers:
            h = layer(h, encoder_memory, rope_emb)

        h = self.output_norm(h)
        return h
