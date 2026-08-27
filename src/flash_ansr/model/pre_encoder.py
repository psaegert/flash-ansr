"""IEEE-754 bit-decomposition pre-encoders that expand float inputs into their sign/exponent/mantissa bits."""
import torch
from torch import nn

#: The three IEEE-754 interchange formats this module can decompose, keyed by total bit width:
#: ``(torch float dtype, same-width signed int dtype, is the input CAST or REINTERPRETED)``.
#: binary16 is the only lossy entry -- its input is cast down first, by design, because it exists
#: as an ablation against the default rather than as a faithful view of the data.
_FORMATS: dict[int, tuple[torch.dtype, torch.dtype, bool]] = {
    16: (torch.float16, torch.int16, True),
    32: (torch.float32, torch.int32, False),
    64: (torch.float64, torch.int64, False),
}


def float_to_ieee754_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Decompose IEEE-754 values into their ``bits`` bits, most-significant-first.

    ``[sign, exponent…, mantissa…]`` -- binary16 ``[s, e[4:0], m[9:0]]``, binary32
    ``[s, e[7:0], m[22:0]]``, binary64 ``[s, e[10:0], m[51:0]]``.

    The dtype check is not defensive noise. ``Tensor.view(dtype)`` does NOT validate or convert;
    it REINTERPRETS and resizes the last dimension. 18 float32 values viewed as int64 and
    expanded to 64 bits each is 576 -- exactly the width a 32-bit pre-encoder produces -- so a
    width mismatch yields a correctly-shaped tensor of scrambled bits and raises nothing
    anywhere downstream. This is the only place that can still see the contract.
    """
    if bits not in _FORMATS:
        raise ValueError(f"Unsupported IEEE-754 width {bits}; expected one of {sorted(_FORMATS)}.")
    float_dtype, int_dtype, lossy = _FORMATS[bits]

    if lossy:
        # binary16: the narrowing cast is the point of this arm, so there is nothing to assert.
        i = x.to(float_dtype).view(int_dtype)
    else:
        if x.dtype is not float_dtype:
            raise TypeError(
                f"float_to_ieee754_bits(bits={bits}) expects a {float_dtype} tensor, got {x.dtype}. "
                f"view(dtype) would silently reinterpret and resize rather than convert.")
        i = x.view(int_dtype)

    # Shift in int64 throughout: an arithmetic right-shift on a narrower signed type sign-extends,
    # and for binary64 the shift counts themselves exceed what int32 can hold.
    shifts = torch.arange(bits - 1, -1, -1, device=x.device, dtype=torch.int64)
    return ((i.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(torch.int8)


def float16_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    """Decompose to binary16 bits (lossy: the input is cast down first)."""
    return float_to_ieee754_bits(x, 16)


def float32_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    """Decompose binary32 values into their 32 bits."""
    return float_to_ieee754_bits(x, 32)


def float64_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    """Decompose binary64 values into their 64 bits."""
    return float_to_ieee754_bits(x, 64)


class IEEE754PreEncoder(nn.Module):
    """Expand each of ``input_size`` features into its ``bits`` IEEE-754 bits, mapped to ``{-1, +1}``.

    One class for all three widths (owner ruling 2026-08-27): they differed only in a bit count,
    and three near-identical copies is how a migration ends up half-applied. ``encoding_size``
    and therefore ``output_size`` follow ``bits``, so the encoder's input dimension tracks the
    numeric format automatically -- there is no literal to keep in sync.
    """

    def __init__(self, input_size: int, bits: int = 32) -> None:
        super().__init__()
        if bits not in _FORMATS:
            raise ValueError(f"Unsupported IEEE-754 width {bits}; expected one of {sorted(_FORMATS)}.")
        self.input_size = input_size
        self.bits = bits
        self.encoding_size = bits

    @property
    def output_size(self) -> int:
        """Flattened output dimensionality: ``encoding_size * input_size``."""
        return self.encoding_size * self.input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand ``x`` into its IEEE-754 bits, mapped from ``{0, 1}`` to ``{-1, +1}``."""
        return (float_to_ieee754_bits(x, self.bits) - 0.5) * 2


class IEEE75416PreEncoder(IEEE754PreEncoder):
    """binary16 pre-encoder. Lossy by design -- an ablation against the default, not a data view."""

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, bits=16)


class IEEE75432PreEncoder(IEEE754PreEncoder):
    """binary32 pre-encoder (the v24 default)."""

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, bits=32)


class IEEE75464PreEncoder(IEEE754PreEncoder):
    """binary64 pre-encoder.

    The reason the v25 line exists: binary32 cannot represent the extreme-magnitude regime the
    benchmark actually contains (measured -- 0.381% of generated support rows overflow to +-inf
    and 0.059% flush silently to 0.0), and the underflow half is finite, so no isfinite gate
    anywhere catches it.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, bits=64)
