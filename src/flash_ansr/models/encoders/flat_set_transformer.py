from flash_ansr.models.encoders.set_encoder import SetEncoder
from flash_ansr.models.encoders.set_transformer import ISAB, PMA, SAB
from flash_ansr.models.transformer_utils import PositionalEncoding
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

        self.enc = nn.Sequential(
            ISAB(input_embedding_size, hidden_size, n_heads, n_induce[0], layer_norm),
            *[ISAB(hidden_size, hidden_size, n_heads, n_induce[i + 1], layer_norm) for i in range(n_enc_isab - 1)])

        self.dec = nn.Sequential(
            PMA(hidden_size, n_heads, n_seeds, layer_norm),
            *[SAB(hidden_size, hidden_size, n_heads, layer_norm) for _ in range(n_dec_sab)],
            nn.Linear(hidden_size, output_embedding_size))

        self.output_embedding_size = output_embedding_size

        self.add_positional_encoding = add_positional_encoding
        self.positional_encoding_out = PositionalEncoding()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, M, D, E = X.shape

        if self.add_positional_encoding:
            X = X + self.positional_encoding_out(shape=(D, E), device=X.device)

        out = self.dec(self.enc(X.reshape(B, M * D, E)))
        out = out.reshape(B, -1, D, self.output_embedding_size)

        return out
