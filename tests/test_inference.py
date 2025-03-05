import torch
import numpy as np
import os

from flash_ansr import FlashANSR, get_path, install_model
from flash_ansr.expressions.utils import codify, num_to_constants

import unittest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "psaegert/flash-ansr-v7.0"


class TestInference(unittest.TestCase):
    def test_out_of_the_box_inference(self):
        # TODO: Install new if already exists
        install_model(MODEL)

        assert os.path.exists(get_path('models', MODEL))

        nsr = FlashANSR.load(
            directory=get_path('models', MODEL),
            n_restarts=32,
        ).to(device)

        assert isinstance(nsr, FlashANSR)

        expression = 'exp(- (x - 3.4)**2)'
        constants = (3.4,)
        xlim = (-5, 5)

        prefix_expression = nsr.expression_space.parse_expression(expression, mask_numbers=True)
        prefix_expression_w_num = nsr.expression_space.operators_to_realizations(prefix_expression)
        prefix_expression_w_constants, constants_names = num_to_constants(prefix_expression_w_num)
        code_string = nsr.expression_space.prefix_to_infix(prefix_expression_w_constants, realization=True)
        code = codify(code_string, nsr.expression_space.variables + constants_names)

        def demo_function(x):
            return nsr.expression_space.code_to_lambda(code)(x, 0, 0, *constants)

        x = np.random.uniform(*xlim, 100)

        y = demo_function(x)

        if isinstance(y, float):
            y = np.full_like(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

        nsr.fit(x_tensor, y_tensor)
