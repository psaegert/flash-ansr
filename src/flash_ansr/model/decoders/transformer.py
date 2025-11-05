"""Transformer decoder stack built from reusable decoder components."""
from typing import Optional

import torch
from torch import nn

from flash_ansr.model.common import get_norm_layer
from flash_ansr.model.decoders.components import RotaryEmbedding, TransformerDecoderBlock


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
    ) -> torch.Tensor:
        seq_len = tokens.shape[1]
        h = self.tok_embeddings(tokens)

        if extra_parallel_embeddings is not None:
            h = h + extra_parallel_embeddings

        rope_emb = self.rope(h, seq_len=seq_len)
        encoder_memory = self.cross_attn_kv_proj(encoder_memory)
        encoder_memory = self.cross_attn_kv_norm(encoder_memory)

        for layer in self.layers:
            h = layer(h, encoder_memory, rope_emb)

        h = self.output_norm(h)
        return h
