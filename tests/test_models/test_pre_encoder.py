"""Bit-level correctness tests for the IEEE-754 pre-encoders."""
import struct

import numpy as np
import pytest
import torch

from flash_ansr.model.pre_encoder import (
    IEEE75416PreEncoder,
    IEEE75432PreEncoder,
    float16_to_ieee754_bits,
    float32_to_ieee754_bits,
)


def _bits_from_struct_fp32(x: float) -> list[int]:
    """Reference: pack as big-endian fp32, return bits MSB-first."""
    packed = struct.pack(">f", x)
    n = int.from_bytes(packed, "big")
    return [(n >> (31 - i)) & 1 for i in range(32)]


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


# ---------- pre-encoder modules ----------

class TestPreEncoderModules:
    def test_32bit_output_shape(self):
        enc = IEEE75432PreEncoder(input_size=4)
        x = torch.randn(2, 5, 4)
        out = enc(x)
        assert out.shape == (2, 5, 4, 32)
        assert enc.output_size == 4 * 32

    def test_16bit_output_shape(self):
        enc = IEEE75416PreEncoder(input_size=4)
        x = torch.randn(2, 5, 4)
        out = enc(x)
        assert out.shape == (2, 5, 4, 16)
        assert enc.output_size == 4 * 16

    def test_output_values_are_pm_one(self):
        for cls in (IEEE75432PreEncoder, IEEE75416PreEncoder):
            enc = cls(input_size=3)
            x = torch.randn(4, 7, 3)
            out = enc(x).flatten().tolist()
            assert set(out).issubset({-1, 1})

    @pytest.mark.parametrize("cls,bits", [(IEEE75432PreEncoder, 32), (IEEE75416PreEncoder, 16)])
    def test_no_trainable_params(self, cls, bits):
        enc = cls(input_size=2)
        assert sum(p.numel() for p in enc.parameters()) == 0
        assert enc.encoding_size == bits
