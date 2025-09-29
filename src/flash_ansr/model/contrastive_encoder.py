import os
import warnings
from typing import Any, Literal

import torch
from torch import nn

from flash_ansr.utils import load_config, substitute_root_path, save_config
from flash_ansr.model.pre_encoder import IEEE75432PreEncoder
from flash_ansr.model.set_transformer import SetTransformer


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        encoder_max_n_variables: int,
        encoder_dim: int = 512,
        encoder_n_heads: int = 8,
        encoder_n_isab: int = 2,
        encoder_n_sab: int = 1,
        encoder_n_inducing_points: int = 32,
        encoder_n_seeds: int = 1,
        encoder_ffn_hidden_dim: int = 2048,
        encoder_dropout: float = 0.1,
        encoder_attn_norm: str = "none",
        encoder_ffn_norm: str = "none",
        encoder_output_norm: str = "none",

        contrastive_hidden_dim: int = 2048,
        contrastive_output_dim: int = 16384,
        contrastive_dropout: float = 0.5,

        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.encoder_max_n_variables = encoder_max_n_variables

        self.pre_encoder = IEEE75432PreEncoder(input_size=encoder_max_n_variables)

        self.encoder = SetTransformer(
            input_dim=self.pre_encoder.output_size,
            output_dim=None,
            model_dim=encoder_dim,
            n_heads=encoder_n_heads,
            n_isab=encoder_n_isab,
            n_sab=encoder_n_sab,
            n_inducing_points=encoder_n_inducing_points,
            n_seeds=encoder_n_seeds,
            ffn_hidden_dim=encoder_ffn_hidden_dim,
            dropout=encoder_dropout,
            attn_norm=encoder_attn_norm,
            ffn_norm=encoder_ffn_norm,
            output_norm=encoder_output_norm,
            use_checkpointing=use_checkpointing
        )

        if self.encoder.output_dim != contrastive_hidden_dim:
            contrastive_head_input_dim = self.encoder.output_dim
        else:
            contrastive_head_input_dim = contrastive_hidden_dim

        self.contrastive_head = nn.Sequential(
            nn.Linear(contrastive_head_input_dim, contrastive_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=contrastive_dropout),
            nn.Linear(contrastive_hidden_dim, contrastive_output_dim)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "ContrastiveEncoder":
        config_ = load_config(config)

        if "contrastive_encoder" in config_.keys():
            config_ = config_["contrastive_encoder"]

        return cls(
            encoder_max_n_variables=config_["encoder_max_n_variables"],
            encoder_dim=config_["encoder_dim"],
            encoder_n_heads=config_["encoder_n_heads"],
            encoder_n_isab=config_["encoder_n_isab"],
            encoder_n_sab=config_["encoder_n_sab"],
            encoder_n_inducing_points=config_["encoder_n_inducing_points"],
            encoder_n_seeds=config_["encoder_n_seeds"],
            encoder_ffn_hidden_dim=config_["encoder_ffn_hidden_dim"],
            encoder_dropout=config_["encoder_dropout"],
            encoder_attn_norm=config_["encoder_attn_norm"],
            encoder_ffn_norm=config_["encoder_ffn_norm"],
            encoder_output_norm=config_["encoder_output_norm"],

            contrastive_hidden_dim=config_["contrastive_hidden_dim"],
            contrastive_output_dim=config_["contrastive_output_dim"],
            contrastive_dropout=config_["contrastive_dropout"],

            use_checkpointing=config_["use_checkpointing"],
        )

    def forward(self, data: torch.Tensor, data_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if data.ndim != 3:
            data = data.unsqueeze(0)

        # Pre-process input data
        data_pre_encodings: torch.Tensor = self.pre_encoder(data)
        B, M, D, E = data_pre_encodings.size()

        # If in training, add a small amount of noise to the pre-encodings for regularization
        if self.training:
            noise = torch.randn_like(data_pre_encodings) * 0.1
            data_pre_encodings = data_pre_encodings + noise

        # Encoder forward pass
        representation = self.encoder(data_pre_encodings.view(B, M, D * E), data_attn_mask)

        if representation.ndim > 3:
            representation = representation.view(B, -1, representation.size(-1))

        # DINO head forward pass
        representation = self.contrastive_head(representation)

        return representation

    @classmethod
    def load(cls, directory: str) -> tuple[dict[str, Any], "ContrastiveEncoder"]:
        directory = substitute_root_path(directory)

        config_path = os.path.join(directory, 'contrastive_encoder.yaml')

        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(os.path.join(directory, "state_dict.pt"), weights_only=True))

        return load_config(config_path), model

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True, errors: Literal['raise', 'warn', 'ignore'] = 'warn') -> None:

        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(directory, "state_dict.pt"))

        # Copy the config to the directory for best portability
        if config is None:
            if errors == 'raise':
                raise ValueError("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
            if errors == 'warn':
                warnings.warn("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
        else:
            save_config(
                load_config(config, resolve_paths=True),
                directory=directory,
                filename='contrastive_encoder.yaml',
                reference=reference,
                recursive=recursive,
                resolve_paths=True)
