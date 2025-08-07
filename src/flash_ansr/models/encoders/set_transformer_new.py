import torch
import torch.nn.functional as F
from torch import nn

from flash_ansr.models.encoders.set_encoder import SetEncoder


class NewMAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            size: int,
            n_heads: int) -> None:
        super().__init__()
        if size % n_heads != 0:
            raise ValueError(f"size_V ({size}) must be divisible by n_heads ({n_heads})")

        self.mha = nn.MultiheadAttention(embed_dim=size, num_heads=n_heads, batch_first=True,)
        self.linear_out = nn.Linear(size, size)

        self.residual_linear = nn.Linear(size, size)
        self.residual_activation = nn.ReLU()

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear_out(self.mha(query, key_value, key_value)[0])) + self.residual_linear(query)


class NewSAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, size: int, n_heads: int) -> None:
        super().__init__()
        self.mab = NewMAB(size, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class NewISAB(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, size: int, n_heads: int, n_induce: int) -> None:
        super().__init__()

        self.inducing_points = nn.Parameter(torch.Tensor(1, n_induce, size))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab0 = NewMAB(size, n_heads)
        self.mab1 = NewMAB(size, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class NewPMA(nn.Module):
    # https://github.com/juho-lee/set_transformer
    def __init__(self, size: int, n_heads: int, n_seeds: int) -> None:
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, n_seeds, size))
        nn.init.xavier_uniform_(self.S)

        self.mab = NewMAB(size, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class NewSetTransformer(SetEncoder):
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
            n_heads: int = 4) -> None:
        super().__init__()
        if n_enc_isab < 0:
            raise ValueError(f"Number of ISABs in encoder `n_enc_isab` ({n_enc_isab}) cannot be negative")

        if n_dec_sab < 0:
            raise ValueError(f"Number of SABs in decoder `n_dec_sab` ({n_dec_sab}) cannot be negative")

        if isinstance(n_induce, int):
            n_induce = [n_induce] * n_enc_isab
        elif len(n_induce) != n_enc_isab:
            raise ValueError(
                f"Number of inducing points `n_induce` ({n_induce}) must be an integer or a list of length {n_enc_isab}")

        self.linear_in = nn.Linear(input_embedding_size * input_dimension_size, hidden_size)
        self.activation_in = nn.GELU()

        self.enc = nn.Sequential(*[NewISAB(hidden_size, n_heads, n_induce[i]) for i in range(n_enc_isab)])
        self.pma = NewPMA(hidden_size, n_heads, n_seeds)
        self.dec = nn.Sequential(*[NewSAB(hidden_size, n_heads) for _ in range(n_dec_sab)])
        self.linear_out = nn.Linear(hidden_size, output_embedding_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, M, D * E)

        x = self.linear_in(X)
        x = self.activation_in(x)
        x = self.enc(x)
        x = self.pma(x)
        x = self.dec(x)
        x = self.linear_out(x)
        return x
