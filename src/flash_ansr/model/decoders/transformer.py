"""Transformer decoder stack built from reusable decoder components.

Two ways for the data to reach the expression tokens, chosen by ``data_mode``:

* ``cross_attention`` (v23-v25): every block cross-attends to the static encoder memory.
* ``prefix`` (v26): the encoder memory is part of the decoder's own sequence. The internal
  layout is ``tokens[0] <data> m_1 .. m_S </data> tokens[1:]`` -- the first token (``<bos>``)
  keeps its place, the memory slots follow between two learned tag vectors, then the rest of
  the token sequence. The block ``tokens[0] <data> m_1 .. m_S </data>`` attends BIDIRECTIONALLY
  within itself (the data is a set, not a sequence, and every slot may read every other); the
  expression tokens are causal over everything before them. There is no cross-attention. The
  outputs at the inserted positions are dropped, so the caller sees one hidden state per input
  token exactly as before: the state at ``</data>`` stands in for ``tokens[0]``'s and predicts
  ``tokens[1]``, the rest are the token positions themselves.
"""
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
        data_mode: str = "cross_attention",
        memory_len: int | None = None,
        data_norm_type: str = "rms",
    ):
        """``max_seq_len`` counts the caller's tokens; in ``prefix`` mode the rotary table is sized
        for the tokens plus the inserted ``memory_len + 2`` prefix positions."""
        super().__init__()
        if data_mode not in ("cross_attention", "prefix"):
            raise ValueError(f"data_mode must be 'cross_attention' or 'prefix', got {data_mode!r}")
        self.data_mode = data_mode
        head_dim = model_dim // n_heads
        self.tok_embeddings = nn.Embedding(vocab_size, model_dim)

        # The inserted prefix: <data>, the memory slots, </data>. Zero in cross-attention mode so
        # every position arithmetic below reads the same in both modes.
        if data_mode == "prefix":
            if memory_len is None or int(memory_len) < 1:
                raise ValueError("prefix data_mode needs memory_len (the number of encoder memory slots)")
            self.memory_len = int(memory_len)
            self.prefix_len = self.memory_len + 2
        else:
            self.memory_len = 0
            self.prefix_len = 0
        self.token_max_seq_len = int(max_seq_len)
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len + self.prefix_len)

        projection: nn.Module
        if input_dim is not None and input_dim != model_dim:
            projection = nn.Linear(input_dim, model_dim)
        else:
            projection = nn.Identity()
        self.cross_attn_kv_proj: nn.Module | None = None
        self.cross_attn_kv_norm: nn.Module | None = None
        self.data_proj: nn.Module | None = None
        self.data_norm: nn.Module | None = None
        if data_mode == "prefix":
            self.data_proj = projection
            self.data_norm = get_norm_layer(data_norm_type, model_dim)
            # The two tag vectors (<data>, </data>) live in the decoder, not the vocabulary: the
            # sampler can never emit them and the head carries no dead rows for them. Shaped
            # (1, 2, dim) so the optimizer roles read them as an embedding table.
            self.data_tags = nn.Parameter(torch.randn(1, 2, model_dim))
        else:
            self.cross_attn_kv_proj = projection

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
                use_cross_attention=(data_mode == "cross_attention"),
            )
            for _ in range(n_layers)
        ])

        if data_mode == "cross_attention":
            self.cross_attn_kv_norm = get_norm_layer(cross_attn_kv_norm_type, model_dim)
        self.output_norm = get_norm_layer(output_norm_type, model_dim)

    def prefix_mask(self, total_len: int, device: torch.device) -> torch.Tensor:
        """The ``(total_len, total_len)`` boolean attention mask of the prefix layout (True = attend):
        the first ``prefix_len + 1`` positions (``tokens[0]`` and the inserted block) attend to each
        other in both directions, every position attends causally to what precedes it."""
        idx = torch.arange(total_len, device=device)
        causal = idx[None, :] <= idx[:, None]
        in_block = idx < (self.prefix_len + 1)
        return causal | (in_block[:, None] & in_block[None, :])

    def build_prefix_sequence(self, h: torch.Tensor, encoder_memory: torch.Tensor) -> torch.Tensor:
        """Insert the data prefix after the first token: ``h[:, :1] <data> memory </data> h[:, 1:]``."""
        assert self.data_proj is not None and self.data_norm is not None
        mem = self.data_norm(self.data_proj(encoder_memory))
        if mem.shape[1] != self.memory_len:
            raise ValueError(f"encoder memory has {mem.shape[1]} slots, the prefix decoder was built for {self.memory_len}")
        batch = h.shape[0]
        if mem.shape[0] != batch:
            # One memory shared by every row (the sampler decodes `choices` rows from one problem).
            mem = mem.expand(batch, -1, -1)
        tags = self.data_tags.to(dtype=h.dtype).expand(batch, -1, -1)
        return torch.cat([h[:, :1], tags[:, :1], mem.to(dtype=h.dtype), tags[:, 1:], h[:, 1:]], dim=1)

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

        # Prefix mode, first pass (prefill or a full forward): splice the data into the sequence.
        # An incremental step finds the prefix in the cache already and adds nothing.
        attn_mask: torch.Tensor | None = None
        prefixed = self.data_mode == "prefix" and past_key_values is None
        if prefixed:
            if seq_len < 1:
                raise ValueError("the prefix decoder needs at least the first token to place the data after")
            h = self.build_prefix_sequence(h, encoder_memory)
            seq_len = h.shape[1]
            attn_mask = self.prefix_mask(seq_len, h.device)

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
        if past_key_values is None and self.data_mode == "cross_attention":
            assert self.cross_attn_kv_proj is not None and self.cross_attn_kv_norm is not None
            encoder_memory = self.cross_attn_kv_proj(encoder_memory)
            encoder_memory = self.cross_attn_kv_norm(encoder_memory)

        new_key_values: list[LayerCache] = [] if use_cache else []

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            layer_out = layer(h, encoder_memory, rope_emb, past_key_value=layer_past, use_cache=use_cache, attn_mask=attn_mask)
            if use_cache:
                h, layer_cache = layer_out
                new_key_values.append(layer_cache)
            else:
                h = layer_out

        h = self.output_norm(h)

        if prefixed:
            # One state per input token: </data>'s state takes tokens[0]'s place.
            h = h[:, self.prefix_len:]

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
        read under a causal mask. Pre-norm, RoPE-self only (XSA not yet supported here).
        Cross-attn K/V must already be seeded (from a dynamic prefill) or are computed once here.
        In prefix mode ``position`` is the CACHE slot (the token's index plus ``prefix_len``); the
        prefix itself entered the cache with the dynamic prefill."""
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
        if static_cache.ca[0][0] is None and self.data_mode == "cross_attention":
            assert self.cross_attn_kv_proj is not None and self.cross_attn_kv_norm is not None
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
