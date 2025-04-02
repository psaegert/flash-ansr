import math

import torch
from torch import nn

from flash_ansr.models.encoders.set_encoder import SetEncoder
from flash_ansr.models.generic import SwiGLU, ReLU2


class MAB_1(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            size_Q: int,
            size_KV: int,
            size: int,
            size_ff: int,
            n_heads: int,
            dropout: float = 0.1) -> None:
        super().__init__()
        if size % n_heads != 0:
            raise ValueError(f"size_V ({size}) must be divisible by n_heads ({n_heads})")

        self.size = size
        self.head_size = size // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(size_Q, size)
        self.W_k = nn.Linear(size_KV, size)
        self.W_v = nn.Linear(size_KV, size)

        self.ff_up = nn.Linear(size, size_ff)
        self.ff_down = nn.Linear(size_ff, size)

        self.ff_activation = ReLU2()

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff_up = nn.Dropout(dropout)
        self.dropout_ff_down = nn.Dropout(dropout)

        self.attention_normalization = 1 / math.sqrt(self.head_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.W_q(x)
        k: torch.Tensor = self.W_k(y)
        v: torch.Tensor = self.W_v(y)

        # Split heads and concatenate batches
        Q_ = torch.cat(q.split(self.head_size, dim=2), 0)
        K_ = torch.cat(k.split(self.head_size, dim=2), 0)
        V_ = torch.cat(v.split(self.head_size, dim=2), 0)

        # Compute attention scores and apply softmax
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) * self.attention_normalization, 2)

        # Apply attention to values
        attention_output = A.bmm(V_)

        # Reshape back: split by batch and concatenate heads
        attention_output = torch.cat(attention_output.split(x.size(0), 0), 2)

        # Apply dropout
        attention_output = self.dropout_attn(attention_output)

        # First residual connection
        h = x + attention_output

        # Feed-forward network with residual connection
        ff_output = self.ff_down(self.dropout_ff_up(self.ff_activation(self.ff_up(h))))
        output = h + self.dropout_ff_down(ff_output)

        return output


class SAB_1(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, input_size: int, output_size: int, size_ff: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mab = MAB_1(input_size, input_size, output_size, size_ff, n_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB_1(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, input_size: int, output_size: int, size_ff: int, n_heads: int, n_induce: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.inducing_points = nn.Parameter(torch.Tensor(1, n_induce, output_size))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab0 = MAB_1(output_size, input_size, output_size, size_ff, n_heads, dropout)
        self.mab1 = MAB_1(input_size, output_size, output_size, size_ff, n_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA_1(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, size: int, size_ff: int, n_heads: int, n_seeds: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, n_seeds, size))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB_1(size, size, size, size_ff, n_heads, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer_1(SetEncoder):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            input_embedding_size: int,
            input_dimension_size: int,
            output_embedding_size: int,
            size_ff: int,
            n_seeds: int,
            hidden_size: int = 512,
            n_enc_isab: int = 5,
            n_dec_sab: int = 2,
            n_induce: int | list[int] = 64,
            n_heads: int = 8,
            dropout: float = 0.1) -> None:
        super().__init__()
        if n_enc_isab < 1:
            raise ValueError(f"Number of ISABs in encoder `n_enc_isab` ({n_enc_isab}) must be greater than 0")

        if n_dec_sab < 0:
            raise ValueError(f"Number of SABs in decoder `n_dec_sab` ({n_dec_sab}) cannot be negative")

        if isinstance(n_induce, int):
            n_induce = [n_induce] * n_enc_isab
        elif len(n_induce) != n_enc_isab:
            raise ValueError(
                f"Number of inducing points `n_induce` ({n_induce}) must be an integer or a list of length {n_enc_isab}")

        self.embedding_in = SwiGLU(input_embedding_size * input_dimension_size, out_features=hidden_size)
        self.enc = nn.Sequential(*[ISAB_1(hidden_size, hidden_size, size_ff, n_heads, n_induce[i], dropout) for i in range(n_enc_isab)])
        self.pma = PMA_1(hidden_size, size_ff, n_heads, n_seeds, dropout)
        self.dec = nn.Sequential(*[SAB_1(hidden_size, hidden_size, size_ff, n_heads, dropout) for _ in range(n_dec_sab)])
        self.linear_out = nn.Linear(hidden_size, output_embedding_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, M, D, E = X.size()
        X = X.reshape(B, M, D * E)
        X = self.embedding_in(X)
        X = self.enc(X)
        X = self.pma(X)
        X = self.dec(X)
        return self.linear_out(X)
