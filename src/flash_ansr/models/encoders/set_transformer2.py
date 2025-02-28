# https://github.com/rajesh-lab/deep_permutation_invariant

import torch
import torch.nn as nn
import math

import torch.nn.functional as F


class SetNorm(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-5):
        super().__init__()
        self.size = size

        # Make gamma and beta learnable parameters by wrapping them with nn.Parameter
        self.gamma = nn.Parameter(torch.ones(size))  # Scaling factor (γ_d)
        self.beta = nn.Parameter(torch.zeros(size))  # Shifting factor (β_d)
        self.epsilon = epsilon

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Compute mean per set (over M and D)
        mu_n = torch.mean(X, dim=(1, 2), keepdim=True)  # Shape (N, 1, 1)

        # Compute variance per set (over M and D)
        sigma_n = torch.sqrt(torch.mean((X - mu_n) ** 2, dim=(1, 2), keepdim=True) + self.epsilon)  # Shape (N, 1, 1)

        # Normalize and apply affine transformation
        a_norm = (X - mu_n) / sigma_n  # Shape (N, M, D)
        a_out = a_norm * self.gamma + self.beta  # Shape (N, M, D)

        return a_out

    def extra_repr(self) -> str:
        return f"size={self.size}, epsilon={self.epsilon}"


class MABpp(nn.Module):
    def __init__(
            self,
            size_Q: int,
            size_KV: int,
            size: int,
            n_heads: int,
            set_norm_q: bool = True,
            set_norm_ko: bool = True):
        super().__init__()

        self.size = size
        self.head_size = size // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(size_Q, size)
        self.W_k = nn.Linear(size_KV, size)
        self.W_v = nn.Linear(size_KV, size)

        self.fc_o = nn.Linear(size, size)

        self.set_norm_q = set_norm_q
        self.set_norm_ko = set_norm_ko

        if set_norm_q:
            self.norm_q = SetNorm(size_Q)

        if set_norm_ko:
            self.norm_k = SetNorm(size_KV)
            self.norm_o = SetNorm(size)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        input_query = query

        if self.set_norm_ko:
            key_value = self.norm_k(key_value)

        if self.set_norm_q:
            query = self.norm_q(query)

        q: torch.Tensor = self.W_q(query)
        k: torch.Tensor = self.W_k(key_value)
        v: torch.Tensor = self.W_v(key_value)

        Q_ = torch.cat(q.split(self.head_size, dim=2), 0)
        K_ = torch.cat(k.split(self.head_size, dim=2), 0)
        V_ = torch.cat(v.split(self.head_size, dim=2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.size), 2)

        output = torch.cat(input_query.split(self.head_size, 2), 0)
        output = torch.cat((output + A.bmm(V_)).split(query.size(0), 0), 2)

        if self.set_norm_ko:
            output = output + F.relu(self.fc_o(self.norm_o(output)))
        else:
            output = output + F.relu(self.fc_o(output))

        return output


class SABpp(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_heads: int, set_norm_q: bool = True, set_norm_ko: bool = True) -> None:
        super().__init__()
        self.mab = MABpp(input_size, input_size, output_size, n_heads, set_norm_q, set_norm_ko)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISABpp(nn.Module):
    def __init__(self, input_size: int, output_size: int, n_heads: int, n_induce: int, set_norm_ko: bool = True) -> None:
        super().__init__()

        self.inducing_points = nn.Parameter(torch.Tensor(1, n_induce, output_size))
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab0 = MABpp(output_size, input_size, output_size, n_heads, set_norm_q=False, set_norm_ko=set_norm_ko)
        self.mab1 = MABpp(input_size, output_size, output_size, n_heads, set_norm_q=True, set_norm_ko=set_norm_ko)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMApp(nn.Module):
    def __init__(self, size: int, n_heads: int, n_seeds: int, set_norm_q: bool = True, set_norm_ko: bool = False) -> None:
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, n_seeds, size))
        nn.init.xavier_uniform_(self.S)

        self.mab = MABpp(size, size, size, n_heads, set_norm_q, set_norm_ko)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer2(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            n_seeds: int,
            hidden_size: int = 512,
            n_enc_isab: int = 2,
            n_dec_sab: int = 2,
            n_induce: int | list[int] = 64,
            n_heads: int = 4,
            set_norm_ko: bool = True) -> None:
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
            nn.Linear(input_size, hidden_size),  # Need to expand size to set up clean path for ISAB
            ISABpp(hidden_size, hidden_size, n_heads, n_induce[0], set_norm_ko),
            *[ISABpp(hidden_size, hidden_size, n_heads, n_induce[i + 1], set_norm_ko) for i in range(n_enc_isab - 1)])

        # TODO: Add a set norm here?

        self.dec = nn.Sequential(
            PMApp(hidden_size, n_heads, n_seeds, set_norm_ko=False),
            *[SABpp(hidden_size, hidden_size, n_heads, set_norm_ko=False) for _ in range(n_dec_sab)],
            nn.Linear(hidden_size, output_size))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(X))
