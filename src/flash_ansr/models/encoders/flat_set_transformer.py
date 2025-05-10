from flash_ansr.models.encoders.set_encoder import SetEncoder
from flash_ansr.models.encoders.set_transformer import ISAB, PMA, SAB
from flash_ansr.models.encoders.pre_encoder import PreEncoder
import torch
from torch import nn


class FlatSetTransformer(SetEncoder):
    # https://github.com/juho-lee/set_transformer
    def __init__(
            self,
            input_embedding_size: int,
            output_embedding_size: int,
            n_seeds: int,
            input_dimension_size: int | None = None,
            hidden_size: int = 512,
            n_enc_isab: int = 2,
            n_dec_sab: int = 2,
            n_induce: int | list[int] = 64,
            n_heads: int = 4,
            layer_norm: bool = False,
            add_positional_encoding: bool = True) -> None:
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

        self.add_positional_encoding = add_positional_encoding

        if self.add_positional_encoding:
            self.dim_pre_encoder = PreEncoder(1, mode='ieee-754')
            total_input_size = input_embedding_size + self.dim_pre_encoder.output_size
        else:
            total_input_size = input_embedding_size

        self.enc = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            *[ISAB(hidden_size, n_heads, n_induce[i]) for i in range(n_enc_isab)])

        self.dec = nn.Sequential(
            PMA(hidden_size, n_heads, n_seeds),
            *[SAB(hidden_size, n_heads) for _ in range(n_dec_sab)],
            nn.Linear(hidden_size, output_embedding_size))

        self.input_embedding_size = input_embedding_size
        self.output_embedding_size = output_embedding_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, M, DE = X.shape
        D = DE // self.input_embedding_size
        E = self.input_embedding_size

        X = X.reshape(B, M, D, E)

        if self.add_positional_encoding:
            dimension_encodings = self.dim_pre_encoder(torch.arange(D, device=X.device).unsqueeze(0).expand(B, M, -1))
            X = torch.cat((X, dimension_encodings), dim=-1)

        B, M, D, E = X.shape

        out = self.dec(self.enc(X.reshape(B, M * D, E)))
        out = out.reshape(B, -1, D, self.output_embedding_size)

        return out
