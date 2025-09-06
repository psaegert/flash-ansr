import torch
from torch import nn


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
