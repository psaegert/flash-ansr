"""IEEE-754 bit-decomposition pre-encoders that expand float inputs into their sign/exponent/mantissa bits."""
import torch
from torch import nn


def float32_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    """Decompose IEEE-754 binary32 (single-precision) values into their 32 bits.

    Output bit order is most-significant-first: ``[sign, exp[7:0], mantissa[22:0]]``.
    """
    # reinterpret bits as int32
    i = x.view(torch.int32)

    # build indices [31, 30, …, 0]
    bit_idx = torch.arange(31, -1, -1, device=x.device, dtype=torch.int32)

    # shift, mask, and cast to int8
    bits = ((i.unsqueeze(-1) >> bit_idx) & 1).to(torch.int8)

    return bits


def float16_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    """Decompose IEEE-754 binary16 (half-precision) values into their 16 bits.

    Input tensor is cast to float16 (lossy) before bit reinterpretation.
    Output bit order is most-significant-first: ``[sign, exp[4:0], mantissa[9:0]]``.
    """
    # cast to fp16 then reinterpret bits as int16
    i = x.to(torch.float16).view(torch.int16)

    # build indices [15, 14, …, 0]
    bit_idx = torch.arange(15, -1, -1, device=x.device, dtype=torch.int16)

    # shift, mask, and cast to int8. Cast int16 -> int32 first to avoid
    # negative-shift surprises on the sign bit under arithmetic shift.
    bits = ((i.to(torch.int32).unsqueeze(-1) >> bit_idx.to(torch.int32)) & 1).to(torch.int8)

    return bits


class IEEE75432PreEncoder(nn.Module):
    """Pre-encoder that expands each of ``input_size`` features into its 32 IEEE-754 binary32 bits.

    Bits are mapped from ``{0, 1}`` to ``{-1, +1}`` before being passed on to the encoder.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoding_size = 32  # Fixed for IEEE-754 32-bit representation

    @property
    def output_size(self) -> int:
        """Flattened output dimensionality: ``encoding_size * input_size`` (32 bits per feature)."""
        return self.encoding_size * self.input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand ``x`` into its binary32 bits, mapped from ``{0, 1}`` to ``{-1, +1}``."""
        return (float32_to_ieee754_bits(x) - 0.5) * 2


class IEEE75416PreEncoder(nn.Module):
    """Half-precision (binary16) variant of :class:`IEEE75432PreEncoder`.

    Inputs are cast to float16 before bit decomposition. The cast is lossy by
    design — this encoder is used as an ablation against the 32-bit default.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoding_size = 16

    @property
    def output_size(self) -> int:
        """Flattened output dimensionality: ``encoding_size * input_size`` (16 bits per feature)."""
        return self.encoding_size * self.input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand ``x`` into its binary16 bits, mapped from ``{0, 1}`` to ``{-1, +1}``."""
        return (float16_to_ieee754_bits(x) - 0.5) * 2
