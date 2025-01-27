from typing import Any, Literal

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
        nan_pattern = torch.zeros_like(result[0])
        # Set exponent bits (all 1s)
        nan_pattern[1:1 + num_e_bits] = 1
        # Set first mantissa bit to 1
        nan_pattern[1 + num_e_bits] = 1
        result[is_nan] = nan_pattern

    # Handle positive infinity
    if torch.any(is_pos_inf):
        inf_pattern = torch.zeros_like(result[0])
        # Sign bit is 0
        # Set all exponent bits to 1
        inf_pattern[1:1 + num_e_bits] = 1
        # Mantissa is all zeros
        result[is_pos_inf] = inf_pattern

    # Handle negative infinity
    if torch.any(is_neg_inf):
        neg_inf_pattern = torch.zeros_like(result[0])
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


class PreEncoder(nn.Module):
    def __init__(self, input_size: int, mode: Literal["ieee-754", "numeric", "frexp", "sfrexp"] = "numeric", support_nan: bool = False, exponent_scale: float | None = None) -> None:
        super().__init__()

        if mode not in ["ieee-754", "numeric", "frexp", "sfrexp"]:
            raise ValueError(f"Invalid mode: {mode}, expected one of ['ieee-754', 'numeric', 'frexp', 'sfrexp']")

        if exponent_scale is not None and mode not in ["frexp", "sfrexp"]:
            raise ValueError(f"exponent_scale is only valid for modes ['frexp', 'sfrexp'], got mode: {mode}")

        if support_nan and mode == "ieee-754":
            raise ValueError("support_nan is not valid for mode 'ieee-754'")

        self.input_size = input_size
        self.mode = mode
        self.support_nan = support_nan
        self.exponent_scale = exponent_scale

    @property
    def output_size(self) -> int:
        # Increase the number of dimensions from d * (number) to d * ({sign,} mantissa, exponent, {nan_flag})
        output_size = self.input_size

        if self.mode == "frexp":
            output_size += self.input_size
        elif self.mode == "sfrexp":
            output_size += self.input_size * 2
        elif self.mode == "ieee-754":
            output_size += self.input_size * 15

        if self.mode != "ieee-754" and self.support_nan:
            output_size += self.input_size

        return output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "ieee-754":
            x_bit = float2bit(x)
            return (x_bit.view(*x_bit.shape[:-2], self.output_size) - 0.5) * 2

        x_isnan = torch.isnan(x)
        x_isinf = torch.isinf(x)

        if self.mode == "numeric":
            if self.support_nan:
                x_with_nan_mask = torch.cat((x, (x_isnan | x_isinf).float()), dim=-1)
                return x_with_nan_mask
            return x

        mantissa = torch.empty_like(x)
        exponent = torch.empty_like(x)

        x_isnan = torch.isnan(x)
        x_isinf = torch.isinf(x)

        mantissa_valid, exponent_valid = torch.frexp(x[~x_isnan & ~x_isinf])
        mantissa[~x_isnan & ~x_isinf] = mantissa_valid
        exponent[~x_isnan & ~x_isinf] = exponent_valid.float()

        mantissa[x_isnan | x_isinf] = 0
        exponent[x_isnan | x_isinf] = 0

        if self.mode == "frexp":
            x_frexp = torch.cat([mantissa, exponent / self.exponent_scale], dim=-1)
            if self.support_nan:
                x_with_nan_mask = torch.cat((x_frexp, (x_isnan | x_isinf).float()), dim=-1)
                return x_with_nan_mask
            return x_frexp

        if self.mode == "sfrexp":
            sign = torch.sign(mantissa)
            mantissa = torch.abs(mantissa)
            x_sfrexp = torch.cat([sign, mantissa, exponent / self.exponent_scale], dim=-1)

            if self.support_nan:
                x_with_nan_mask = torch.cat((x_sfrexp, (x_isnan | x_isinf).float()), dim=-1)
                return x_with_nan_mask
            return x_sfrexp

        raise ValueError(f"Invalid mode: {self.mode}, expected one of ['numeric', 'frexp', 'sfrexp']")
