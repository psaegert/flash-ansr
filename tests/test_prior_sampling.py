import math
import unittest
from unittest.mock import patch

import torch

from flash_ansr import FlashANSR
from flash_ansr.expressions import NoValidSampleFoundError
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.generation import PriorSamplingConfig


class _DummyTransformer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device('cpu')
        self.encoder_max_n_variables = 3
        self.preprocessor = None

    def eval(self) -> "_DummyTransformer":
        return self

    def to(self, device: torch.device | str) -> "_DummyTransformer":
        self.device = torch.device(device)
        return self


class _DummyEngine:
    def is_valid(self, _expression: list[str]) -> bool:
        return True


class _StubPool:
    def __init__(self, skeletons: set[tuple[str, ...]], generated: list[tuple[tuple[str, ...], None, list[str]]] | None = None) -> None:
        self.skeletons = skeletons
        self._generated = generated or []
        self._index = 0

    def sample_skeleton(self, new: bool = False, decontaminate: bool = True) -> tuple[tuple[str, ...], None, list[str]]:
        if self._index >= len(self._generated):
            raise NoValidSampleFoundError('exhausted')
        value = self._generated[self._index]
        self._index += 1
        return value

    def clear_holdouts(self) -> None:
        return None


class TestPriorSampling(unittest.TestCase):
    def setUp(self) -> None:
        vocab = ['x1', 'sin', 'add']
        self.tokenizer = Tokenizer(vocab=vocab)
        self.transformer = _DummyTransformer(self.tokenizer)
        self.engine = _DummyEngine()

    def test_prior_sampling_generates_expected_beams(self) -> None:
        config = PriorSamplingConfig(samples=2, unique=True, seed=0, skeleton_pool='dummy-path')
        nsr = FlashANSR(
            simplipy_engine=self.engine,
            flash_ansr_model=self.transformer,
            tokenizer=self.tokenizer,
            generation_config=config,
        )

        skeletons = {('x1',), ('sin', 'x1')}
        stub_pool = _StubPool(skeletons)

        with patch.object(FlashANSR, '_ensure_prior_pool', return_value=stub_pool):
            beams, log_probs, completed, rewards = nsr.generate(data=torch.zeros((1, 1, nsr.n_variables + 1)))

        self.assertEqual(len(beams), 2)
        self.assertTrue(all(completed))
        self.assertTrue(all(math.isnan(value) for value in rewards))

        expected_log_prob = -math.log(len(skeletons))
        self.assertTrue(all(abs(value - expected_log_prob) < 1e-9 for value in log_probs))

        decoded = [self.tokenizer.decode(self.tokenizer.extract_expression_from_beam(beam)[0]) for beam in beams]
        flattened = [tuple(tokens) for tokens in decoded]
        self.assertEqual(sorted(flattened), sorted(tuple(expr) for expr in skeletons))

    def test_prior_sampling_uses_dynamic_generation_when_no_skeletons(self) -> None:
        config = PriorSamplingConfig(samples=2, unique=True, seed=None, skeleton_pool='dummy-path')
        nsr = FlashANSR(
            simplipy_engine=self.engine,
            flash_ansr_model=self.transformer,
            tokenizer=self.tokenizer,
            generation_config=config,
        )

        generated = [
            (('x1',), None, []),
            (('sin', 'x1'), None, []),
        ]
        stub_pool = _StubPool(set(), generated)

        with patch.object(FlashANSR, '_ensure_prior_pool', return_value=stub_pool):
            beams, _, _, _ = nsr.generate(data=torch.zeros((1, 1, nsr.n_variables + 1)))

        decoded = [self.tokenizer.decode(self.tokenizer.extract_expression_from_beam(beam)[0]) for beam in beams]
        self.assertEqual(sorted(tuple(tokens) for tokens in decoded), sorted(tuple(expr) for expr, *_ in generated))


if __name__ == '__main__':
    unittest.main()
