import os
import unittest
import warnings

import numpy as np
import torch

from flash_ansr import (
    FlashANSR,
    GenerationConfig,
    BeamSearchConfig,
    SoftmaxSamplingConfig,
    MCTSGenerationConfig,
    get_path,
    install_model,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "psaegert/flash-ansr-v23.0-3M"


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_model(MODEL)
        cls.model_dir = get_path('models', MODEL)
        assert os.path.exists(cls.model_dir), "Pretrained model should be available after installation"

        cls.device = device
        cls.constants = (3.4,)
        cls.xlim = (-5, 5)

        rng = np.random.default_rng(0)
        x = rng.uniform(*cls.xlim, 96)
        y = cls._target_function(x)

        cls.x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(cls.device)
        cls.y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(cls.device)

    @classmethod
    def _target_function(cls, x: np.ndarray) -> np.ndarray:
        return np.exp(-((x - cls.constants[0]) ** 2))

    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = self.__class__.device
        self.model_dir = self.__class__.model_dir
        self.x_tensor = self.__class__.x_tensor
        self.y_tensor = self.__class__.y_tensor

    def _fit_with_generation_config(
            self,
            generation_config: GenerationConfig,
            n_restarts: int = 4) -> FlashANSR:
        nsr = FlashANSR.load(
            directory=self.model_dir,
            generation_config=generation_config,
            n_restarts=n_restarts,
        ).to(self.device)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in power",
                category=RuntimeWarning,
            )
            nsr.fit(self.x_tensor, self.y_tensor)
        return nsr

    def _assert_valid_results(self, nsr: FlashANSR) -> None:
        self.assertFalse(nsr.results.empty, "Expected at least one candidate result")
        scores = nsr.results['score'].to_numpy(dtype=float, copy=True)
        self.assertTrue(np.isfinite(scores).any(), "Expected at least one finite score")
        expressions = nsr.results['expression']
        self.assertTrue(any(len(expr) > 0 for expr in expressions), "Expressions should not be empty")

        expression_infix = nsr.get_expression()
        self.assertIsInstance(expression_infix, str)
        self.assertGreater(len(expression_infix), 0, "Infix expression should not be empty")
        self.assertNotIn('nan', expression_infix.lower(), "Expression rendering should not emit NaN constants")

        expression_prefix = nsr.get_expression(return_prefix=True)
        self.assertIsInstance(expression_prefix, list)
        self.assertTrue(expression_prefix, "Prefix expression should not be empty")

        predictions = nsr.predict(self.x_tensor)
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().numpy()
        else:
            predictions_np = np.asarray(predictions)

        self.assertEqual(predictions_np.shape, (self.x_tensor.shape[0], 1))
        self.assertTrue(np.isfinite(predictions_np).any(), "Predictions should include at least one finite value")

        def placeholder_count(expr_tokens: list[str]) -> int:
            return sum(token == '<constant>' for token in expr_tokens)

        # Validate DataFrame expressions against fitted constants
        for expr_tokens, fit_constants in zip(nsr.results['expression'], nsr.results['fit_constants']):
            if isinstance(fit_constants, np.ndarray):
                self.assertEqual(placeholder_count(expr_tokens), len(fit_constants))

        # Validate cached refiner state and rendered expressions for every beam/fit pair
        for beam_idx, result in enumerate(nsr._results):
            fits = result.get('fits', [])
            if not fits:
                continue

            placeholders = placeholder_count(result['expression'])
            refiner_constants = [constants for constants, *_ in result['refiner']._all_constants_values]

            for fit_idx, (constants, _cov, _loss) in enumerate(fits):
                self.assertEqual(len(constants), placeholders)

                for ref_constants in refiner_constants:
                    self.assertEqual(len(ref_constants), placeholders)

                expr_str = nsr.get_expression(nth_best_beam=beam_idx, nth_best_constants=fit_idx)
                self.assertNotIn('nan', expr_str.lower(), f"Beam {beam_idx} fit {fit_idx} expression contains NaN")

                expr_prefix = nsr.get_expression(nth_best_beam=beam_idx, nth_best_constants=fit_idx, return_prefix=True)
                self.assertFalse(any(token == '<constant>' for token in expr_prefix),
                                 f"Beam {beam_idx} fit {fit_idx} prefix retains <constant> placeholders")

    def test_beam_search_inference(self) -> None:
        generation_config = BeamSearchConfig(
            beam_width=8,
            max_len=24,
            batch_size=32,
            unique=True,
        )

        nsr = self._fit_with_generation_config(generation_config, n_restarts=6)

        self.assertEqual(nsr.generation_config.method, 'beam_search')
        self._assert_valid_results(nsr)

    def test_softmax_sampling_inference(self) -> None:
        generation_config = SoftmaxSamplingConfig(
            choices=16,
            top_k=8,
            top_p=0.95,
            max_len=24,
            batch_size=32,
            temperature=0.8,
            simplify=True,
            unique=True,
        )

        nsr = self._fit_with_generation_config(generation_config, n_restarts=6)

        self.assertEqual(nsr.generation_config.method, 'softmax_sampling')
        self._assert_valid_results(nsr)

    def test_mcts_inference(self) -> None:
        generation_config = MCTSGenerationConfig(
            beam_width=6,
            simulations=24,
            expansion_top_k=12,
            max_depth=24,
            temperature=1.0,
            completion_sort='reward',
            min_visits_before_expansion=1,
            invalid_penalty=1e5,
        )

        nsr = self._fit_with_generation_config(generation_config, n_restarts=6)

        self.assertEqual(nsr.generation_config.method, 'mcts')
        self._assert_valid_results(nsr)
