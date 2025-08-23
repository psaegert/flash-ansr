from typing import Any

import torch
from torch import nn


# https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/src/nesymres/architectures/set_encoder.py
def float2bit(f: torch.Tensor, num_e_bits: int = 5, num_m_bits: int = 10, bias: int = 127, dtype: Any = torch.float32) -> torch.Tensor:
    # Create output tensor with same shape as input plus bits dimension
    output_shape = list(f.shape) + [1 + num_e_bits + num_m_bits]
    result = torch.zeros(output_shape, device=f.device, dtype=dtype)

    # Handle special cases
    is_nan = torch.isnan(f)
    is_inf = torch.isinf(f)
    is_neg_inf = is_inf & (f < 0)
    is_pos_inf = is_inf & (f > 0)
    is_normal = ~(is_nan | is_inf)

    # For normal numbers, use existing logic
    if torch.any(is_normal):
        normal_vals = f[is_normal]

        # SIGN BIT
        s = (torch.sign(normal_vals + 0.001) * -1 + 1) * 0.5
        s = s.unsqueeze(-1)
        f1 = torch.abs(normal_vals)

        # EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = integer2bit(e_decimal, num_bits=num_e_bits)

        # MANTISSA
        f2 = f1 / 2 ** e_scientific
        m2 = remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[..., :num_m_bits]

        normal_result = torch.cat([s, e, fin_m], dim=-1).type(dtype)
        result[is_normal] = normal_result

    # Handle NaN
    if torch.any(is_nan):
        # Set all exponent bits to 1 and non-zero mantissa (conventionally first mantissa bit is 1)
        nan_pattern = torch.zeros(num_e_bits + num_m_bits + 1, dtype=dtype, device=f.device)
        # Set exponent bits (all 1s)
        nan_pattern[1:1 + num_e_bits] = 1
        # Set first mantissa bit to 1
        nan_pattern[1 + num_e_bits] = 1
        result[is_nan] = nan_pattern

    # Handle positive infinity
    if torch.any(is_pos_inf):
        inf_pattern = torch.zeros(num_e_bits + num_m_bits + 1, dtype=dtype, device=f.device)
        # Sign bit is 0
        # Set all exponent bits to 1
        inf_pattern[1:1 + num_e_bits] = 1
        # Mantissa is all zeros
        result[is_pos_inf] = inf_pattern

    # Handle negative infinity
    if torch.any(is_neg_inf):
        neg_inf_pattern = torch.zeros(num_e_bits + num_m_bits + 1, dtype=dtype, device=f.device)
        # Sign bit is 1
        neg_inf_pattern[0] = 1
        # Set all exponent bits to 1
        neg_inf_pattern[1:1 + num_e_bits] = 1
        # Mantissa is all zeros
        result[is_neg_inf] = neg_inf_pattern

    return result.type(dtype)


def remainder2bit(remainder: torch.Tensor, num_bits: int = 127) -> torch.Tensor:
    exponent_bits = torch.arange(num_bits, device=remainder.device).type(remainder.type())
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)


def integer2bit(integer: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    exponent_bits = - torch.arange(-(num_bits - 1), 1, device=integer.device).type(integer.type())
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2


def hamming(x: torch.Tensor) -> torch.Tensor:
    mask = torch.abs(x) < 1
    y = torch.zeros_like(x)
    y[mask] = torch.cos(x[mask] * torch.pi) + 1
    return y


def float32_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    # reinterpret bits as int32
    i = x.view(torch.int32)

    # build indices [31, 30, …, 0]
    bit_idx = torch.arange(31, -1, -1, device=x.device, dtype=torch.int32)

    # shift, mask, and cast to int8
    bits = ((i.unsqueeze(-1) >> bit_idx) & 1).to(torch.int8)

    return bits


class IEEE75432PreEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoding_size = 32  # Fixed for IEEE-754 32-bit representation

    @property
    def output_size(self) -> int:
        return self.encoding_size * self.input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (float32_to_ieee754_bits(x) - 0.5) * 2
