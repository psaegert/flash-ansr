"""
MIT License

Copyright (c) 2019 Tony Duan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# https://github.com/tonyduan/mixture-density-network

import math
from enum import Enum, auto

import torch
import torch.nn as nn


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in: int, dim_out: int, n_components: int, hidden_dim: int, noise_type: NoiseType = NoiseType.DIAGONAL, fixed_noise_level: float | None = None) -> None:
        super().__init__()
        if noise_type is NoiseType.FIXED and fixed_noise_level is None:
            raise ValueError("fixed_noise_level must be provided when noise_type is FIXED")

        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out * n_components + num_sigma_channels)
        )

    def forward(self, x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the mixture density network.

        Returns
        -------
        log_pi: (bsz, n_components)
        mu: (bsz, n_components, dim_out)
        sigma: (bsz, n_components, dim_out)
        """
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        elif self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps)
            sigma = sigma.repeat_interleave(self.dim_out, dim=-1)
        elif self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps)
            sigma = sigma.expand(*sigma.shape[:-1], self.n_components * self.dim_out).contiguous()
        elif self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)  # type: ignore
        feature_shape = tuple(x.shape[:-1])
        if len(feature_shape) == 0:
            feature_shape = (1,)
        view_shape = (*feature_shape, self.n_components, self.dim_out)
        mu = mu.reshape(view_shape).contiguous()
        sigma = sigma.reshape(view_shape).contiguous()
        log_pi = log_pi.reshape(*feature_shape, self.n_components)
        return log_pi, mu, sigma

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_pi, mu, sigma = self.forward(x)
        return negative_log_likelihood(log_pi, mu, sigma, y).mean()

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        log_pi, mu, sigma = self.forward(x)
        return sample_from_mixture(log_pi, mu, sigma)


def negative_log_likelihood(log_pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute elementwise negative log likelihood under a Gaussian mixture."""

    conditioned = target
    if conditioned.dim() == log_pi.dim() - 1:
        conditioned = conditioned.unsqueeze(-1)
    elif conditioned.dim() != log_pi.dim():
        raise ValueError("target dimensionality must match logits or logits-1")
    sigma = torch.clamp(sigma, min=eps)
    diff = (conditioned.unsqueeze(-2) - mu) / sigma
    quad_form = -0.5 * torch.sum(diff.pow(2), dim=-1)
    log_det = -torch.sum(torch.log(sigma), dim=-1)
    dim = mu.size(-1)
    normal_loglik = quad_form + log_det - 0.5 * dim * math.log(2 * math.pi)
    log_prob = torch.logsumexp(log_pi + normal_loglik, dim=-1)
    return -log_prob


def sample_from_mixture(log_pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Sample from a Gaussian mixture defined by ``log_pi``, ``mu`` and ``sigma``."""

    pi = torch.softmax(log_pi, dim=-1)
    flat_pi = pi.reshape(-1, pi.shape[-1])
    component = torch.multinomial(flat_pi, num_samples=1)
    component = component.view(*pi.shape[:-1], 1)
    noise = torch.randn_like(mu) * sigma + mu
    gather_index = component.unsqueeze(-1).expand(*pi.shape[:-1], 1, mu.shape[-1])
    sample = torch.gather(noise, dim=-2, index=gather_index).squeeze(-2)
    if sample.shape[-1] == 1:
        sample = sample.squeeze(-1)
    return sample
