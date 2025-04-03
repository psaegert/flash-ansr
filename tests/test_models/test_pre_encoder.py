import unittest
import torch

from flash_ansr import PreEncoder


class TestPreEncoder(unittest.TestCase):
    def test_numeric(self):
        pre_encoder = PreEncoder(10, mode="numeric", support_nan=False)

        output_size = pre_encoder.output_size

        assert output_size == 10

        x = torch.randn(67, 10)
        y = pre_encoder.forward(x)

        assert y.shape == (67, 10, 1)

    def test_frexp(self):
        pre_encoder = PreEncoder(10, mode="frexp", support_nan=False, exponent_scale=1)

        output_size = pre_encoder.output_size

        assert output_size == 20

        x = torch.randn(67, 10)
        y = pre_encoder.forward(x)

        assert y.shape == (67, 10, 2)

    def test_sfrexp(self):
        pre_encoder = PreEncoder(10, mode="sfrexp", support_nan=False, exponent_scale=1)

        output_size = pre_encoder.output_size

        assert output_size == 30

        x = torch.randn(67, 10)
        y = pre_encoder.forward(x)

        assert y.shape == (67, 10, 3)

    def test_numeric_nan(self):
        pre_encoder = PreEncoder(10, mode="numeric", support_nan=True)

        output_size = pre_encoder.output_size

        assert output_size == 20

        x = torch.randn(67, 10)
        x[0, 0] = float("nan")
        y = pre_encoder.forward(x)

        assert y.shape == (67, 10, 2)

    def test_frexp_nan(self):
        pre_encoder = PreEncoder(10, mode="frexp", support_nan=True, exponent_scale=1)

        output_size = pre_encoder.output_size

        assert output_size == 30  # 10 mantissa, 10 exponent, 10 nan flag

        x = torch.randn(67, 10)
        x[0, 0] = float("nan")
        y = pre_encoder.forward(x)

        assert y.shape == (67, 10, 3)
        assert (~torch.isnan(y)).all()
        print(y)
        assert y[0, 0, -1] == 1  # 10 + 10 + 0 should be flagged as nan
        assert y[0, 1, -1] == 0

    def test_sfrexp_nan(self):
        pre_encoder = PreEncoder(10, mode="sfrexp", support_nan=True, exponent_scale=1)

        output_size = pre_encoder.output_size

        assert output_size == 40  # 10 sign, 10 mantissa, 10 exponent, 10 nan flag

        x = torch.randn(67, 10)
        x[0, 0] = float("nan")
        y = pre_encoder.forward(x)

        assert y.shape == (67, 10, 4)
        assert (~torch.isnan(y)).all()
        assert y[0, 0, -1] == 1  # 10 + 10 10 + 0 should be flagged as nan
        assert y[0, 1, -1] == 0
