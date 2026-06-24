"""Transformer decoder stack built from reusable decoder components."""
from typing import Optional, Tuple, cast

import torch
from torch import nn

from flash_ansr.model.common import get_norm_layer
from flash_ansr.model.decoders.components import RotaryEmbedding, TransformerDecoderBlock
from flash_ansr.model.decoders.static_kv import StaticKVCache

# Type alias for the per-layer cache: ((self_attn_k, self_attn_v), (cross_attn_k, cross_attn_v))
LayerCache = Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class TransformerDecoder(nn.Module):
    """Configurable transformer decoder stack with rotary embeddings."""

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
        use_xsa_self_attn: bool = False,
        block_norm_position: str = "pre",
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
                use_xsa_self_attn=use_xsa_self_attn,
                self_attn_norm_type=block_self_attn_norm_type,
                cross_attn_norm_type=block_cross_attn_norm_type,
                ffn_norm_type=block_ffn_norm_type,
                norm_position=block_norm_position,
            )
            for _ in range(n_layers)
        ])

        self.cross_attn_kv_norm = get_norm_layer(cross_attn_kv_norm_type, model_dim)
        self.output_norm = get_norm_layer(output_norm_type, model_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_memory: torch.Tensor,
        extra_parallel_embeddings: torch.Tensor | None = None,
        past_key_values: list[LayerCache] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[LayerCache]]:
        seq_len = tokens.shape[1]
        h = self.tok_embeddings(tokens)

        if extra_parallel_embeddings is not None:
            h = h + extra_parallel_embeddings

        if past_key_values is not None:
            # Incremental decoding: tokens is only the new token(s).
            # The total sequence length so far = cached length + current tokens.
            cached_seq_len = past_key_values[0][0][0].shape[2]  # layer0 -> self_attn_cache -> K -> seq dim
            total_seq_len = cached_seq_len + seq_len
            rope_emb = self.rope(h, seq_len=total_seq_len)
            # Slice RoPE to only the new positions
            cos_full, sin_full = rope_emb
            rope_emb = (cos_full[:, :, cached_seq_len:total_seq_len, :], sin_full[:, :, cached_seq_len:total_seq_len, :])
        else:
            rope_emb = self.rope(h, seq_len=seq_len)

        # Project and normalise encoder memory (only on prefill, reuse from cache otherwise)
        if past_key_values is None:
            encoder_memory = self.cross_attn_kv_proj(encoder_memory)
            encoder_memory = self.cross_attn_kv_norm(encoder_memory)

        new_key_values: list[LayerCache] = [] if use_cache else []

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            layer_out = layer(h, encoder_memory, rope_emb, past_key_value=layer_past, use_cache=use_cache)
            if use_cache:
                h, layer_cache = layer_out
                new_key_values.append(layer_cache)
            else:
                h = layer_out

        h = self.output_norm(h)

        if use_cache:
            return h, new_key_values
        return h

    def forward_static(
        self,
        tokens: torch.Tensor,
        encoder_memory: torch.Tensor,
        extra_parallel_embeddings: torch.Tensor | None,
        static_cache: StaticKVCache,
        position: int,
    ) -> torch.Tensor:
        """Static-shape (graph-capturable) single-token decode step. `tokens` is the one new token
        (B, 1); its K/V are written into `static_cache` at absolute `position` and the full buffer is
        read under a causal mask. v23.0 path only (pre-norm, RoPE-self; XSA not yet supported here).
        Cross-attn K/V must already be seeded (from a dynamic prefill) or are computed once here."""
        h = self.tok_embeddings(tokens)
        if extra_parallel_embeddings is not None:
            h = h + extra_parallel_embeddings

        # Position-indexed RoPE for the single new token at ABSOLUTE `position` (the dynamic path infers
        # this from cache length, which is always max_len for a static buffer -> must be explicit).
        cos = self.rope.cos_cached[:, :, position:position + 1, :].to(dtype=h.dtype)
        sin = self.rope.sin_cached[:, :, position:position + 1, :].to(dtype=h.dtype)
        rope_emb = (cos, sin)

        # Project + norm encoder memory ONLY if cross-attn K/V are not yet cached (first call when not
        # seeded from a dynamic prefill). When seeded, the holders are populated -> skip (no re-project).
        if static_cache.ca[0][0] is None:
            encoder_memory = self.cross_attn_kv_proj(encoder_memory)
            encoder_memory = self.cross_attn_kv_norm(encoder_memory)

        attn_mask = static_cache.attend_mask(position)
        for i, layer in enumerate(self.layers):
            block = cast(TransformerDecoderBlock, layer)  # nn.ModuleList yields Module; narrow for forward_static
            h = block.forward_static(
                h, encoder_memory, rope_emb,
                (static_cache.sa_k[i], static_cache.sa_v[i]),
                static_cache.ca[i], position, attn_mask,
            )
        return self.output_norm(h)
