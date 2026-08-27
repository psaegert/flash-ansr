"""Bit-level correctness tests for the IEEE-754 pre-encoders."""
import struct

import numpy as np
import pytest
import torch

from flash_ansr.model.pre_encoder import (
    IEEE75416PreEncoder,
    IEEE75432PreEncoder,
    IEEE75464PreEncoder,
    float16_to_ieee754_bits,
    float32_to_ieee754_bits,
    float64_to_ieee754_bits,
)


def _bits_from_struct_fp32(x: float) -> list[int]:
    """Reference: pack as big-endian fp32, return bits MSB-first."""
    packed = struct.pack(">f", x)
    n = int.from_bytes(packed, "big")
    return [(n >> (31 - i)) & 1 for i in range(32)]


def _bits_from_struct_fp64(x: float) -> list[int]:
    """Reference: pack as big-endian fp64, return bits MSB-first."""
    packed = struct.pack(">d", x)
    n = int.from_bytes(packed, "big")
    return [(n >> (63 - i)) & 1 for i in range(64)]


def _bits_from_struct_fp16(x: float) -> list[int]:
    """Reference: pack as big-endian fp16, return bits MSB-first."""
    packed = struct.pack(">e", x)
    n = int.from_bytes(packed, "big")
    return [(n >> (15 - i)) & 1 for i in range(16)]


# ---------- float32 ----------

class TestFloat32ToBits:
    def test_zero(self):
        bits = float32_to_ieee754_bits(torch.tensor([0.0]))
        assert bits.tolist() == [[0] * 32]

    def test_one(self):
        # 1.0 = 0 01111111 00000000000000000000000
        bits = float32_to_ieee754_bits(torch.tensor([1.0]))
        expected = [0] + [0, 1, 1, 1, 1, 1, 1, 1] + [0] * 23
        assert bits.tolist() == [expected]

    def test_neg_one(self):
        # -1.0 = 1 01111111 00000000000000000000000
        bits = float32_to_ieee754_bits(torch.tensor([-1.0]))
        expected = [1] + [0, 1, 1, 1, 1, 1, 1, 1] + [0] * 23
        assert bits.tolist() == [expected]

    def test_two(self):
        # 2.0 = 0 10000000 00000000000000000000000
        bits = float32_to_ieee754_bits(torch.tensor([2.0]))
        expected = [0] + [1, 0, 0, 0, 0, 0, 0, 0] + [0] * 23
        assert bits.tolist() == [expected]

    def test_neg_zero_sign_bit(self):
        bits = float32_to_ieee754_bits(torch.tensor([-0.0]))[0].tolist()
        assert bits[0] == 1
        assert bits[1:] == [0] * 31

    def test_inf(self):
        bits = float32_to_ieee754_bits(torch.tensor([float("inf")]))[0].tolist()
        # 0 11111111 0...0
        assert bits == [0] + [1] * 8 + [0] * 23

    def test_neg_inf(self):
        bits = float32_to_ieee754_bits(torch.tensor([float("-inf")]))[0].tolist()
        assert bits == [1] + [1] * 8 + [0] * 23

    def test_nan_exponent_all_ones(self):
        bits = float32_to_ieee754_bits(torch.tensor([float("nan")]))[0].tolist()
        # exponent must be all-ones; mantissa must be non-zero
        assert bits[1:9] == [1] * 8
        assert any(b == 1 for b in bits[9:])

    def test_property_roundtrip_random(self):
        torch.manual_seed(0)
        x = torch.randn(1024, dtype=torch.float32) * 1e3
        # Reconstruct float from emitted bits and compare to original
        bits = float32_to_ieee754_bits(x).numpy().astype(np.uint32)
        # bits[i, k] is bit (31 - k); recompose into uint32
        weights = (1 << np.arange(31, -1, -1)).astype(np.uint32)
        ints = (bits * weights).sum(axis=-1).astype(np.uint32)
        recovered = ints.view(np.float32)
        np.testing.assert_array_equal(recovered, x.numpy())

    def test_matches_struct_pack(self):
        for v in [0.5, -0.5, 3.14, -3.14, 1e-30, 1e30, 1.17549435e-38]:
            bits = float32_to_ieee754_bits(torch.tensor([v]))[0].tolist()
            assert bits == _bits_from_struct_fp32(v), f"mismatch at {v}"


# ---------- float16 ----------

class TestFloat16ToBits:
    def test_zero(self):
        bits = float16_to_ieee754_bits(torch.tensor([0.0]))
        assert bits.tolist() == [[0] * 16]

    def test_one(self):
        # 1.0 fp16 = 0 01111 0000000000  (exp bias = 15)
        bits = float16_to_ieee754_bits(torch.tensor([1.0]))
        expected = [0] + [0, 1, 1, 1, 1] + [0] * 10
        assert bits.tolist() == [expected]

    def test_neg_one(self):
        bits = float16_to_ieee754_bits(torch.tensor([-1.0]))
        expected = [1] + [0, 1, 1, 1, 1] + [0] * 10
        assert bits.tolist() == [expected]

    def test_two(self):
        # 2.0 = 0 10000 0000000000
        bits = float16_to_ieee754_bits(torch.tensor([2.0]))
        expected = [0] + [1, 0, 0, 0, 0] + [0] * 10
        assert bits.tolist() == [expected]

    def test_half(self):
        # 0.5 = 0 01110 0000000000
        bits = float16_to_ieee754_bits(torch.tensor([0.5]))
        expected = [0] + [0, 1, 1, 1, 0] + [0] * 10
        assert bits.tolist() == [expected]

    def test_inf(self):
        bits = float16_to_ieee754_bits(torch.tensor([float("inf")]))[0].tolist()
        assert bits == [0] + [1] * 5 + [0] * 10

    def test_neg_zero_sign_bit(self):
        bits = float16_to_ieee754_bits(torch.tensor([-0.0]))[0].tolist()
        assert bits[0] == 1
        assert bits[1:] == [0] * 15

    def test_lossy_cast_from_fp32(self):
        """Values outside fp16 range or precision are mapped through the cast."""
        # fp16 max ~= 65504, so 1e6 saturates to inf
        bits = float16_to_ieee754_bits(torch.tensor([1e6]))[0].tolist()
        assert bits[1:6] == [1] * 5  # inf exponent

    def test_matches_struct_pack(self):
        for v in [0.5, -0.5, 3.14, -3.14, 1.0, 2.0, 100.0]:
            bits = float16_to_ieee754_bits(torch.tensor([v]))[0].tolist()
            assert bits == _bits_from_struct_fp16(v), f"mismatch at {v}"

    def test_property_roundtrip_random(self):
        torch.manual_seed(0)
        # Stay in fp16 representable range (max ~65504)
        x = (torch.randn(1024, dtype=torch.float32) * 100).to(torch.float16)
        bits = float16_to_ieee754_bits(x.to(torch.float32)).numpy().astype(np.uint32)
        weights = (1 << np.arange(15, -1, -1)).astype(np.uint32)
        ints = (bits * weights).sum(axis=-1).astype(np.uint16)
        recovered = ints.view(np.float16)
        # Skip NaN slots
        mask = ~np.isnan(x.numpy())
        np.testing.assert_array_equal(recovered[mask], x.numpy()[mask])


# ---------- float64 ----------

class TestFloat64ToBits:
    """The v25 width. Every case here has a binary32 twin above; what is new is the range --
    1e300 and 1e-300 have no float32 spelling at all, which is the reason for the migration."""

    def test_zero(self):
        bits = float64_to_ieee754_bits(torch.tensor([0.0], dtype=torch.float64))
        assert bits.tolist() == [[0] * 64]

    def test_one(self):
        bits = float64_to_ieee754_bits(torch.tensor([1.0], dtype=torch.float64))
        # 1.0 is 0x3ff0000000000000
        assert bits.tolist() == [_bits_from_struct_fp64(1.0)]

    def test_neg_two(self):
        bits = float64_to_ieee754_bits(torch.tensor([-2.0], dtype=torch.float64))
        assert bits[0, 0].item() == 1                      # sign
        assert bits.tolist() == [_bits_from_struct_fp64(-2.0)]

    def test_neg_zero_sign_bit(self):
        bits = float64_to_ieee754_bits(torch.tensor([-0.0], dtype=torch.float64))
        assert bits[0, 0].item() == 1
        assert bits[0, 1:].sum().item() == 0

    def test_inf_and_nan(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            bits = float64_to_ieee754_bits(torch.tensor([value], dtype=torch.float64))
            assert bits[0, 1:12].tolist() == [1] * 11      # exponent all ones

    def test_the_magnitudes_float32_cannot_hold(self):
        """0.381% of generated support rows overflow binary32 and 0.059% flush to zero
        (measured). Both halves survive here; each is a DISTINCT bit pattern from the
        inf-or-zero that a binary32 stack collapses it onto."""
        for value in (1e300, -1e300, 1e-300, 5e-324):
            with np.errstate(over="ignore"):
                narrowed = np.float32(value)   # the overflow IS the measurement
            assert np.isinf(narrowed) or narrowed == 0.0, f"{value} is representable in f32"
            bits = float64_to_ieee754_bits(torch.tensor([value], dtype=torch.float64))
            assert bits.tolist() == [_bits_from_struct_fp64(value)]
            # ... and it is NOT the pattern of the value float32 collapsed it onto.
            collapsed = float(narrowed)
            assert bits.tolist() != [_bits_from_struct_fp64(collapsed)]

    def test_matches_struct_pack(self):
        values = [0.0, -0.0, 1.0, -1.0, 2.5, 1e300, 1e-300, 3.141592653589793]
        bits = float64_to_ieee754_bits(torch.tensor(values, dtype=torch.float64))
        for i, value in enumerate(values):
            assert bits[i].tolist() == _bits_from_struct_fp64(value), value

    def test_property_roundtrip_random(self):
        rng = np.random.default_rng(0)
        x = torch.tensor(rng.standard_normal(256) * 1e100, dtype=torch.float64)
        bits = float64_to_ieee754_bits(x).to(torch.int64)
        weights = 2 ** torch.arange(63, -1, -1, dtype=torch.int64)
        recovered = (bits * weights).sum(dim=-1).numpy().astype(np.int64).view(np.float64)
        np.testing.assert_array_equal(recovered, x.numpy())


# ---------- width contract ----------

class TestWidthIsNotNegotiable:
    """`view(dtype)` reinterprets and RESIZES; it never converts. These are the two
    directions a half-applied migration takes, and only one of them is self-announcing."""

    def test_64bit_refuses_float32(self):
        """The dangerous direction: 18 f32 values viewed as int64 would be 9 int64s ->
        576 bits, exactly what a 32-bit pre-encoder emits. Nothing downstream could tell."""
        with pytest.raises(TypeError, match="float64"):
            float64_to_ieee754_bits(torch.zeros(2, 18, dtype=torch.float32))

    def test_32bit_refuses_float64(self):
        with pytest.raises(TypeError, match="float32"):
            float32_to_ieee754_bits(torch.zeros(2, 18, dtype=torch.float64))

    def test_the_pre_encoder_module_refuses_it_too(self):
        """Not just the free function -- the nn.Module path is what the model calls."""
        with pytest.raises(TypeError, match="float64"):
            IEEE75464PreEncoder(input_size=4)(torch.randn(2, 5, 4))

    def test_binary16_still_casts_because_that_is_its_point(self):
        """The lossy arm is an ablation against the default, not a view of the data."""
        out = IEEE75416PreEncoder(input_size=4)(torch.randn(2, 5, 4, dtype=torch.float64))
        assert out.shape == (2, 5, 4, 16)


# ---------- pre-encoder modules ----------

class TestPreEncoderModules:
    def test_32bit_output_shape(self):
        enc = IEEE75432PreEncoder(input_size=4)
        x = torch.randn(2, 5, 4)
        out = enc(x)
        assert out.shape == (2, 5, 4, 32)
        assert enc.output_size == 4 * 32

    def test_64bit_output_shape(self):
        enc = IEEE75464PreEncoder(input_size=4)
        x = torch.randn(2, 5, 4, dtype=torch.float64)
        out = enc(x)
        assert out.shape == (2, 5, 4, 64)
        assert enc.output_size == 4 * 64

    def test_16bit_output_shape(self):
        enc = IEEE75416PreEncoder(input_size=4)
        x = torch.randn(2, 5, 4)
        out = enc(x)
        assert out.shape == (2, 5, 4, 16)
        assert enc.output_size == 4 * 16

    def test_output_values_are_pm_one(self):
        for cls, dtype in ((IEEE75432PreEncoder, torch.float32),
                           (IEEE75416PreEncoder, torch.float32),
                           (IEEE75464PreEncoder, torch.float64)):
            enc = cls(input_size=3)
            x = torch.randn(4, 7, 3, dtype=dtype)
            out = enc(x).flatten().tolist()
            assert set(out).issubset({-1, 1})

    def test_the_bits_stop_at_the_pre_encoder(self):
        """f64 in, default-dtype out: {0,1} -> {+-1} promotes, so binary64 never reaches an
        nn.Linear or an autocast-listed op. This is why widening the encoder input does not
        touch amp_dtype, TF32 or the GradScaler."""
        out = IEEE75464PreEncoder(input_size=3)(torch.randn(4, 7, 3, dtype=torch.float64))
        assert out.dtype is torch.get_default_dtype()

    @pytest.mark.parametrize("cls,bits", [(IEEE75432PreEncoder, 32), (IEEE75416PreEncoder, 16),
                                          (IEEE75464PreEncoder, 64)])
    def test_no_trainable_params(self, cls, bits):
        enc = cls(input_size=2)
        assert sum(p.numel() for p in enc.parameters()) == 0
        assert enc.encoding_size == bits
