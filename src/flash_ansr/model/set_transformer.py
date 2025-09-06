from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import time
from tqdm import tqdm

from flash_ansr.model.transformer import FeedForward
from flash_ansr.model.set_encoder import SetEncoder


# A good practice for enabling/disabling checkpointing globally
USE_CHECKPOINTING = True


class SetNorm(nn.Module):
    """
    Set Normalization layer.

    Normalizes features across the set and feature dimensions for each batch element.
    Given an input X of shape (B, M, D), it computes statistics mu and sigma
    over the M and D dimensions, resulting in statistics of shape (B, 1, 1).
    It then applies learnable affine parameters gamma and beta of shape (1, 1, D).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, M, D)
        # Calculate mean and std over the set and feature dimensions (1 and 2)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True)

        # Normalize and apply affine transformation
        x_norm = (x - mu) / (sigma + self.eps)
        return x_norm * self.gamma + self.beta


class RMSSetNorm(nn.Module):
    """
    RMS Normalization layer for sets.

    Normalizes features across the set and feature dimensions for each batch element.
    Given an input X of shape (B, M, D), it computes the RMS over the M and D dimensions,
    resulting in statistics of shape (B, 1, 1).
    It then applies a learnable affine parameter gamma of shape (1, 1, D).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, dim))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, M, D)
        rms = self._rms(x)
        x_norm = x / rms
        return x_norm * self.gamma


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
        is_self_attention: bool = False
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
            q, k, v = self.w_qkv(query).chunk(3, dim=-1)
        else:
            q = self.w_q(query)
            k, v = self.w_kv(key_value).chunk(2, dim=-1)

        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Reshape mask for PyTorch's SDPA
        # Input mask is (B, L_kv). We need it to be broadcastable to (B, H, L_q, L_kv).
        # Reshaping to (B, 1, 1, L_kv) achieves this.
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(batch_size, 1, 1, seq_len_kv)
            # If the mask is for self-attention (e.g., 4D), we can use it directly.

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        return self.w_o(attn_output)


class MAB(nn.Module):
    """Multihead Attention Block with pre-normalization (RMSSetNorm), FFN, and optional checkpointing."""
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim: int,
        n_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
        is_self_attention: bool = False
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing and USE_CHECKPOINTING

        self.norm_q = RMSSetNorm(dim_q)
        self.norm_kv = RMSSetNorm(dim_kv)

        self.attention = MultiheadAttentionBlock(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_out=dim,
            n_heads=n_heads,
            dropout=dropout,
            is_self_attention=is_self_attention
        )

        self.norm_ffn = RMSSetNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def _forward(self, query: torch.Tensor, key_value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-normalization and attention
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_output = self.attention(q_norm, kv_norm, attn_mask=attn_mask)
        query = query + attn_output

        # Pre-normalization and FFN
        q_norm = self.norm_ffn(query)
        ffn_output = self.ffn(q_norm)
        query = query + ffn_output
        return query

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            # Pass attn_mask through checkpointing
            return checkpoint(self._forward, query, key_value, attn_mask, use_reentrant=False)
        else:
            return self._forward(query, key_value, attn_mask)


class SAB(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(
            dim_q=dim, dim_kv=dim, dim=dim, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING,
            is_self_attention=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: The SAB in the decoder operates on a dense, fixed-size set from PMA.
        # It does not require a mask. If SAB were used in the encoder, it would need to accept one.
        return self.mab(x, x)


class ISAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_heads: int, n_inducing_points: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing_points, dim_out))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab_cross = MAB(
            dim_q=dim_out, dim_kv=dim_in, dim=dim_out, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING, is_self_attention=False
        )
        self.mab_self = MAB(
            dim_q=dim_in, dim_kv=dim_out, dim=dim_out, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING, is_self_attention=False
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)

        # Inducing points attend to the input set x. Mask is applied to x (key/value).
        h = self.mab_cross(inducing, x, attn_mask=attn_mask)

        # Input set x attends to the dense inducing point representation h.
        # No mask is needed for the attention calculation itself, as h (key/value) is not padded.
        out = self.mab_self(x, h)

        # Zero out the outputs corresponding to padded inputs to prevent information leakage
        # in residual connections and subsequent layers.
        if attn_mask is not None:
            out = out * attn_mask.unsqueeze(-1)

        return out


class PMA(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_seeds: int, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, n_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MAB(
            dim_q=dim, dim_kv=dim, dim=dim, n_heads=n_heads,
            ffn_hidden_dim=ffn_hidden_dim, dropout=dropout,
            use_checkpointing=USE_CHECKPOINTING, is_self_attention=False
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        # Seeds attend to the input set x. Mask is applied to x (key/value).
        return self.mab(seeds, x, attn_mask=attn_mask)


class SetTransformer(SetEncoder):
    def __init__(
        self, input_dim: int, output_dim: int | None, model_dim: int = 256, n_heads: int = 8,
        n_isab: int = 2, n_sab: int = 1, n_inducing_points: Union[int, List[int]] = 32,
        n_seeds: int = 1, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0
    ):
        super().__init__()
        if isinstance(n_inducing_points, int):
            n_inducing_points = [n_inducing_points] * n_isab

        self.embedding = nn.Linear(input_dim, model_dim)

        self.isabs = nn.ModuleList([
            ISAB(model_dim, model_dim, n_heads, n_ip, ffn_hidden_dim=ffn_hidden_dim, dropout=dropout)
            for n_ip in n_inducing_points
        ])
        self.pma = PMA(model_dim, n_heads, n_seeds, ffn_hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.sabs = nn.ModuleList([
            SAB(model_dim, n_heads, ffn_hidden_dim=ffn_hidden_dim, dropout=dropout)
            for _ in range(n_sab)
        ])
        self.output_norm = RMSSetNorm(model_dim)

        if output_dim is not None:
            self.output = nn.Linear(model_dim, output_dim)
        else:
            self.output = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)

        # Apply mask to input features before encoder to zero out padding.
        if attn_mask is not None:
            x = x * attn_mask.unsqueeze(-1)

        for isab in self.isabs:
            x = isab(x, attn_mask=attn_mask)

        x = self.pma(x, attn_mask=attn_mask)

        # The decoder operates on the dense output of PMA, so no mask is needed.
        for sab in self.sabs:
            x = sab(x)

        x = self.output_norm(x)
        return self.output(x)


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
    n_seeds = 64

    model = SetTransformer(
        input_dim=input_dim, output_dim=output_dim, model_dim=model_dim, n_heads=n_heads,
        n_isab=n_isab, n_sab=n_sab,
        n_inducing_points=n_inducing_points, n_seeds=n_seeds, dropout=0.1
    ).to(device)

    x = torch.randn(batch_size, set_size, input_dim, device=device)
    y = torch.randn(batch_size, n_seeds, output_dim, device=device)

    # Create a sample attention mask
    # Simulate batches where each set has a different size.
    # Here, we create a simple mask where ~75% of points are real and 25% are padding.
    true_set_sizes = torch.randint(int(set_size * 0.5), set_size + 1, (batch_size,), device=device)
    attn_mask = torch.arange(set_size, device=device)[None, :] < true_set_sizes[:, None]
    print(f"\nCreated a sample boolean attention mask of shape: {attn_mask.shape}")
    # Zero out the padded elements in the input tensor
    x = x * attn_mask.unsqueeze(-1)

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
            output = model(x, attn_mask=attn_mask)
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
            output = model(x, attn_mask=attn_mask)
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
