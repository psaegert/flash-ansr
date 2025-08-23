from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import time
from tqdm import tqdm

from flash_ansr.model.transformer import RMSNorm, SwiGLU
from flash_ansr.model.set_encoder import SetEncoder


# A good practice for enabling/disabling checkpointing globally
USE_CHECKPOINTING = True


class MultiheadAttentionBlock(nn.Module):
    """
    Multi-head attention block. Fuses QKV for self-attention, uses separate Q/KV for cross-attention.
    """
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_out: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        is_self_attention: bool = False  # Explicit flag
    ):
        super().__init__()
        if dim_out % n_heads != 0:
            raise ValueError(f"dim_out ({dim_out}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads
        self.dropout = dropout
        self.is_self_attention = is_self_attention

        if self.is_self_attention:
            if dim_q != dim_kv:
                raise ValueError("For self-attention, dim_q must be equal to dim_kv.")
            self.w_qkv = nn.Linear(dim_q, 3 * dim_out, bias=bias)
        else:
            self.w_q = nn.Linear(dim_q, dim_out, bias=bias)
            self.w_kv = nn.Linear(dim_kv, 2 * dim_out, bias=bias)

        self.w_o = nn.Linear(dim_out, dim_out, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        if self.is_self_attention:
            # The caller guarantees query is key_value
            q, k, v = self.w_qkv(query).chunk(3, dim=-1)
        else:
            q = self.w_q(query)
            k, v = self.w_kv(key_value).chunk(2, dim=-1)

        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.w_o(attn_output)


class MAB(nn.Module):
    """Multihead Attention Block with pre-normalization, FFN, and optional checkpointing."""
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim: int,
        n_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
        is_self_attention: bool = False  # Add flag here
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpointing = use_checkpointing

        self.norm_q = RMSNorm(dim_q)
        self.norm_kv = RMSNorm(dim_kv) if dim_kv != dim_q else self.norm_q

        self.attention = MultiheadAttentionBlock(
            dim_q=dim_q, dim_kv=dim_kv, dim_out=dim, n_heads=n_heads,
            dropout=dropout, is_self_attention=is_self_attention  # Pass flag down
        )
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def _forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        normed_q = self.norm_q(query)
        normed_kv = self.norm_kv(key_value)
        attn_output = self.attention(normed_q, normed_kv)

        if query.shape[-1] == self.dim:
            x = query + attn_output
        else:
            x = attn_output

        x = x + self.ffn(self.norm_ffn(x))
        return x

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpointing:
            return checkpoint(self._forward, query, key_value, use_reentrant=False)
        else:
            return self._forward(query, key_value)


class SAB(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(
            dim_q=dim, dim_kv=dim, dim=dim, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING,
            is_self_attention=True  # This is true self-attention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x)


class ISAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_heads: int, n_inducing_points: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing_points, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab_cross = MAB(
            dim_q=dim_out, dim_kv=dim_in, dim=dim_out, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING, is_self_attention=False  # This is cross-attention
        )
        self.mab_self = MAB(
            dim_q=dim_in, dim_kv=dim_out, dim=dim_out, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING, is_self_attention=False  # This is cross-attention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        h = self.mab_cross(inducing, x)
        return self.mab_self(x, h)


class PMA(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_seeds: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, n_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MAB(
            dim_q=dim, dim_kv=dim, dim=dim, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING, is_self_attention=False  # This is cross-attention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        return self.mab(seeds, x)


class SetTransformer(SetEncoder):
    def __init__(
        self, input_dim: int, output_dim: int, model_dim: int = 256, n_heads: int = 8,
        n_isab: int = 2, n_sab: int = 1, n_inducing_points: Union[int, List[int]] = 32,
        n_seeds: int = 1, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0
    ):
        super().__init__()
        if isinstance(n_inducing_points, int):
            n_inducing_points = [n_inducing_points] * n_isab
        elif len(n_inducing_points) != n_isab:
            raise ValueError(f"n_inducing_points must be an int or list of length {n_isab}")

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.encoder = nn.ModuleList([
            ISAB(model_dim, model_dim, n_heads, n_inducing_points[i], ffn_hidden_dim, dropout)
            for i in range(n_isab)
        ])
        self.pooling = PMA(model_dim, n_heads, n_seeds, ffn_hidden_dim, dropout)
        self.decoder = nn.ModuleList([
            SAB(model_dim, n_heads, ffn_hidden_dim, dropout) for _ in range(n_sab)
        ])
        self.output_norm = RMSNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for isab in self.encoder:
            x = isab(x)
        x = self.pooling(x)
        for sab in self.decoder:
            x = sab(x)
        x = self.output_norm(x)
        x = self.output_projection(x)
        return x


# --- Benchmark ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device}, AMP dtype: {amp_dtype}")

    # Model and data parameters
    batch_size, set_size, input_dim_feat = 128, 512, 32
    input_dim = input_dim_feat * 10
    output_dim = 10
    model_dim = 512
    n_heads = 8
    n_isab = 6
    n_sab = 2
    n_inducing_points = 64
    n_seeds = 64  # This determines the output set size

    model = SetTransformer(
        input_dim=input_dim, output_dim=output_dim, model_dim=model_dim, n_heads=n_heads,
        n_isab=n_isab, n_sab=n_sab,
        n_inducing_points=n_inducing_points, n_seeds=n_seeds, dropout=0.1
    ).to(device)

    x = torch.randn(batch_size, set_size, input_dim, device=device)
    y = torch.randn(batch_size, n_seeds, output_dim, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    print("\n--- Starting Benchmark ---")
    print(f"Model dim: {model_dim}, Heads: {n_heads}, Encoder Blocks: {n_isab}, Decoder Blocks: {n_sab}")
    print(f"Batch size: {batch_size}, Input set size: {set_size}, Output set size: {n_seeds}")
    print(f"Checkpointing: {'Enabled' if USE_CHECKPOINTING else 'Disabled'}")

    # Warm-up iterations
    print("Running warm-up iterations...")
    for _ in tqdm(range(5)):
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x)
            loss = F.mse_loss(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark loop
    num_iterations = 50
    model.train()
    start_time = time.time()

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x)
            loss = F.mse_loss(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iterations
    throughput = (batch_size * num_iterations) / total_time

    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_iterations} iterations: {total_time:.2f} seconds")
    print(f"Average time per iteration: {avg_time_per_iter * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
