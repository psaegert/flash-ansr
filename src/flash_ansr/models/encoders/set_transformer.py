import math

import torch
import torch.nn.functional as F
from torch import nn

from flash_ansr.models.encoders.set_encoder import SetEncoder


class MAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            size_Q: int,
            size_KV: int,
            size: int,
            n_heads: int,
            layer_norm: bool = False) -> None:
        super().__init__()
        if size % n_heads != 0:
            raise ValueError(f"size_V ({size}) must be divisible by n_heads ({n_heads})")

        self.size = size
        self.head_size = size // n_heads
        self.n_heads = n_heads
        self.layer_norm = layer_norm

        self.W_q = nn.Linear(size_Q, size)
        self.W_k = nn.Linear(size_KV, size)
        self.W_v = nn.Linear(size_KV, size)

        if layer_norm:
            self.layer_norm_0 = nn.LayerNorm(size)
            self.layer_norm_1 = nn.LayerNorm(size)

        self.fc_o = nn.Linear(size, size)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.W_q(query)
        k: torch.Tensor = self.W_k(key_value)
        v: torch.Tensor = self.W_v(key_value)

        Q_ = torch.cat(q.split(self.head_size, dim=2), 0)
        K_ = torch.cat(k.split(self.head_size, dim=2), 0)
        V_ = torch.cat(v.split(self.head_size, dim=2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.size), 2)
        output = torch.cat((Q_ + A.bmm(V_)).split(query.size(0), 0), 2)

        if self.layer_norm:
            output = self.layer_norm_0(output)

        output = output + F.relu(self.fc_o(output))

        if self.layer_norm:
            output = self.layer_norm_1(output)

        return output


class SAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, input_size: int, output_size: int, n_heads: int, layer_norm: bool = False) -> None:
        super().__init__()
        self.mab = MAB(input_size, input_size, output_size, n_heads, layer_norm)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, input_size: int, output_size: int, n_heads: int, n_induce: int, layer_norm: bool = False) -> None:
        super().__init__()

        self.inducing_points = nn.Parameter(torch.Tensor(1, n_induce, output_size))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab0 = MAB(output_size, input_size, output_size, n_heads, layer_norm)
        self.mab1 = MAB(input_size, output_size, output_size, n_heads, layer_norm)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, size: int, n_heads: int, n_seeds: int, layer_norm: bool = False) -> None:
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, n_seeds, size))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(size, size, size, n_heads, layer_norm)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(SetEncoder):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            input_embedding_size: int,
            input_dimension_size: int,
            output_embedding_size: int,
            n_seeds: int,
            hidden_size: int = 512,
            n_enc_isab: int = 2,
            n_dec_sab: int = 2,
            n_induce: int | list[int] = 64,
            n_heads: int = 4,
            layer_norm: bool = False) -> None:
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

        self.enc = nn.Sequential(
            ISAB(input_embedding_size * input_dimension_size, hidden_size, n_heads, n_induce[0], layer_norm),
            *[ISAB(hidden_size, hidden_size, n_heads, n_induce[i + 1], layer_norm) for i in range(n_enc_isab - 1)])

        self.dec = nn.Sequential(
            PMA(hidden_size, n_heads, n_seeds, layer_norm),
            *[SAB(hidden_size, hidden_size, n_heads, layer_norm) for _ in range(n_dec_sab)],
            nn.Linear(hidden_size, output_embedding_size))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, M, D, E)
        B, M, D, E = X.size()

        # X (B, M, D, E) -> (B, M, D * E)
        X = X.reshape(B, M, D * E)

        return self.dec(self.enc(X))
