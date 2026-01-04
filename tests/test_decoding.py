import math
import unittest

import torch

from flash_ansr.decoding.mcts import MCTSConfig, MonteCarloTreeSearch, PolicyStep


class TestMonteCarloTreeSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.bos = 0
        self.token_a = 1
        self.eos = 2

    def _policy_step(self, probs: list[float]) -> PolicyStep:
        log_probs = torch.log_softmax(torch.tensor(probs, dtype=torch.float32), dim=-1)
        return PolicyStep(log_probs=log_probs)

    def test_collects_completions(self) -> None:
        def policy_fn(tokens: tuple[int, ...], _) -> PolicyStep:
            if tokens[-1] == self.token_a:
                return self._policy_step([float('-inf'), float('-inf'), 0.0])  # force EOS
            return self._policy_step([float('-inf'), 0.0, float('-inf')])

        def value_fn(tokens: tuple[int, ...]) -> float:
            return 1.0 if tokens[-1] == self.eos else -1.0

        def terminal_fn(tokens: tuple[int, ...]) -> bool:
            return tokens[-1] == self.eos

        config = MCTSConfig(simulations=8, expansion_top_k=2, max_depth=4, invalid_penalty=10.0)
        mcts = MonteCarloTreeSearch(
            policy_fn=policy_fn,
            value_fn=value_fn,
            terminal_fn=terminal_fn,
            config=config,
            eos_token_id=self.eos,
        )

        mcts.run((self.bos,))
        completions = mcts.get_top_completions()

        self.assertTrue(completions, "MCTS did not record any completions")
        best_tokens, reward, log_prob = completions[0]
        self.assertEqual(best_tokens, (self.bos, self.token_a, self.eos))
        self.assertEqual(reward, 1.0)
        self.assertTrue(math.isfinite(log_prob))

    def test_invalid_sequence_filter(self) -> None:
        forbidden = 3

        def policy_fn(tokens: tuple[int, ...], _) -> PolicyStep:
            if tokens[-1] == self.token_a:
                return self._policy_step([float('-inf'), float('-inf'), 0.0, float('-inf')])
            return self._policy_step([float('-inf'), math.log(0.5), float('-inf'), math.log(0.5)])

        def value_fn(tokens: tuple[int, ...]) -> float:
            return 1.0 if tokens[-1] == self.eos else -1.0

        def terminal_fn(tokens: tuple[int, ...]) -> bool:
            return tokens[-1] == self.eos

        def invalid_sequence_fn(tokens: tuple[int, ...]) -> bool:
            return forbidden in tokens

        config = MCTSConfig(simulations=10, expansion_top_k=3, max_depth=4, invalid_penalty=5.0)
        mcts = MonteCarloTreeSearch(
            policy_fn=policy_fn,
            value_fn=value_fn,
            terminal_fn=terminal_fn,
            config=config,
            eos_token_id=self.eos,
            pad_token_id=None,
            invalid_sequence_fn=invalid_sequence_fn,
        )

        mcts.run((self.bos,))
        completions = mcts.get_top_completions()

        for tokens, *_ in completions:
            self.assertNotIn(forbidden, tokens)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            MCTSConfig(simulations=0)
        with self.assertRaises(ValueError):
            MCTSConfig(expansion_top_k=0)
        with self.assertRaises(ValueError):
            MCTSConfig(max_depth=0)
        with self.assertRaises(ValueError):
            MCTSConfig(rollout_policy='invalid')
        with self.assertRaises(ValueError):
            MCTSConfig(temperature=0)


if __name__ == "__main__":
    unittest.main()
