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


class TestMCTSCorrectness(unittest.TestCase):
    """Regression tests for the reconciled fix spec (2026-07 decoder rewrite)."""

    def _dummy_mcts(self, config: MCTSConfig | None = None, canonicalize_fn=None) -> MonteCarloTreeSearch:
        return MonteCarloTreeSearch(
            policy_fn=lambda t, s: PolicyStep(log_probs=torch.zeros(4)),
            value_fn=lambda t: 0.0,
            terminal_fn=lambda t: bool(t) and t[-1] == 2,
            config=config or MCTSConfig(simulations=4),
            eos_token_id=2,
            canonicalize_fn=canonicalize_fn,
        )

    def test_q_value_anchors_and_bounds_fvu(self) -> None:
        """B1/B3, value_objective='fvu': bounded q in [0, 1] anchored on log10(fvu)."""
        mcts = self._dummy_mcts(MCTSConfig(simulations=4, value_objective="fvu"))
        self.assertAlmostEqual(mcts._q_value(0.0, {"log_fvu": 0.0}), 0.0)     # fvu=1 -> q=0
        self.assertAlmostEqual(mcts._q_value(0.0, {"log_fvu": -8.0}), 1.0)    # fvu=1e-8 -> q=1
        self.assertAlmostEqual(mcts._q_value(0.0, {"log_fvu": -4.0}), 0.5)    # midpoint
        self.assertEqual(mcts._q_value(0.0, {"log_fvu": -20.0}), 1.0)         # clipped, not >1
        self.assertEqual(mcts._q_value(0.0, {"log_fvu": +5.0}), 0.0)          # clipped, not <0
        self.assertEqual(mcts._q_value(-1e6, {"log_fvu": float("nan")}), 0.0)  # invalid floor, NOT -1e6
        for lf in (-16.0, -8.0, -2.0, 0.0, 3.0, float("nan")):
            self.assertTrue(0.0 <= mcts._q_value(123.0, {"log_fvu": lf}) <= 1.0)
        # generic (no log_fvu) fallback stays bounded and never raises, even for extreme/NaN raw reward
        self.assertTrue(0.0 <= mcts._q_value(-1e6, {}) <= 1.0)   # would OverflowError with a naive sigmoid
        self.assertTrue(0.0 <= mcts._q_value(1e6, {}) <= 1.0)
        self.assertEqual(mcts._q_value(float("nan"), {}), 0.0)   # non-finite -> floor, not NaN

    def test_q_value_score_mode(self) -> None:
        """Default value_objective='score': q anchors on the FULL penalized reward (score = -raw_reward)."""
        mcts = self._dummy_mcts()   # default -> 'score'
        self.assertEqual(mcts.config.value_objective, "score")
        # q = (hi + raw_reward)/(hi - lo) with hi=0, lo=-8 -> raw_reward/8, clipped to [0,1].
        self.assertAlmostEqual(mcts._q_value(0.0, {"log_fvu": -2.0}), 0.0)    # score=0 -> q=0
        self.assertAlmostEqual(mcts._q_value(4.0, {"log_fvu": -2.0}), 0.5)    # reward=4 -> q=0.5
        self.assertAlmostEqual(mcts._q_value(8.0, {"log_fvu": -2.0}), 1.0)    # reward=8 -> q=1
        self.assertEqual(mcts._q_value(20.0, {"log_fvu": -16.0}), 1.0)        # clipped, not >1
        self.assertEqual(mcts._q_value(-3.0, {"log_fvu": -2.0}), 0.0)         # clipped, not <0
        self.assertEqual(mcts._q_value(-1.0, {"log_fvu": float("nan")}), 0.0)  # invalid floor
        # a parsimony-penalized (longer) candidate at the SAME fvu gets a LOWER q than a shorter one
        q_short = mcts._q_value(6.0, {"log_fvu": -6.0})   # short: smaller penalty -> higher reward
        q_long = mcts._q_value(5.0, {"log_fvu": -6.0})    # long: bigger penalty -> lower reward
        self.assertGreater(q_short, q_long)

    def test_valid_dominates_invalid_same_canonical_key(self) -> None:
        """Regression (verify workflow): validity is the primary dedup key; raw reward only breaks ties
        WITHIN a validity class. The invalid floor (-1.0) sits inside the valid raw-reward range, so a plain
        reward comparison would let an invalid entry evict/suppress a valid one of the same canonical key."""
        def canon(toks):
            return tuple(toks[:2])

        # (a) invalid registered first, then a valid POOR fit whose raw reward is BELOW the invalid floor
        mcts = self._dummy_mcts(canonicalize_fn=canon)
        mcts._register_completion((0, 1, 9), -1.0, -1.0, {"log_fvu": float("nan")})   # invalid
        mcts._register_completion((0, 1, 2), -1.5, -1.0, {"log_fvu": 0.5})            # valid poor fit
        top = mcts.get_top_completions(limit=10)
        self.assertEqual([t[0] for t in top], [(0, 1, 2)])   # valid poor-fit harvested, not suppressed
        self.assertEqual(mcts._n_distinct_valid, 1)

        # (b) valid registered first, then a HIGHER-raw-reward invalid: the valid rep must survive
        mcts2 = self._dummy_mcts(canonicalize_fn=canon)
        mcts2._register_completion((0, 1, 2), -1.5, -1.0, {"log_fvu": 0.5})           # valid
        mcts2._register_completion((0, 1, 9), -1.0, -1.0, {"log_fvu": float("nan")})  # invalid, higher reward
        top2 = mcts2.get_top_completions(limit=10)
        self.assertEqual([t[0] for t in top2], [(0, 1, 2)])   # valid representative retained
        self.assertEqual(mcts2._n_distinct_valid, 1)

    def test_nan_logits_do_not_crash_expansion(self) -> None:
        """Regression: a degenerate forward emitting NaN logits must not abort run() via NaN priors."""
        def policy_fn(tokens, _):
            return PolicyStep(log_probs=torch.tensor([0.0, float("nan"), 0.0, float("nan")]))

        config = MCTSConfig(simulations=6, max_depth=4)
        mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=lambda t: (1.0, {"log_fvu": -3.0}),
                                    terminal_fn=lambda t: bool(t) and t[-1] == 2,
                                    config=config, eos_token_id=2)
        mcts.run((0,))   # must not raise RuntimeError

    def test_generation_config_validates_enums(self) -> None:
        from flash_ansr.utils.generation import MCTSGenerationConfig
        with self.assertRaises(ValueError):
            MCTSGenerationConfig(completion_sort="best")
        with self.assertRaises(ValueError):
            MCTSGenerationConfig(backup="invalid")
        with self.assertRaises(ValueError):
            MCTSGenerationConfig(rollout_policy="nope")

    def test_invalid_does_not_evict_subtree(self) -> None:
        """B2: under max backup an invalid (q=0) rollout is inert -- it never drags a good subtree down."""
        mcts = self._dummy_mcts(MCTSConfig(simulations=4, backup="max"))
        from flash_ansr.decoding.mcts import MCTSNode
        node = MCTSNode(tokens=(0,), prior=1.0)
        mcts._backpropagate([node], 0.9)   # a great leaf
        mcts._backpropagate([node], 0.0)   # an invalid / hopeless leaf (floor)
        self.assertEqual(node.best_value, 0.9)          # max backup: subtree keeps its gem
        self.assertAlmostEqual(node.exploitation("max"), 0.9)
        self.assertAlmostEqual(node.exploitation("mean"), 0.45)

    def test_dedup_by_canonical_key_before_cut(self) -> None:
        """B4: completions are deduplicated by canonical key, keeping the best-reward representative."""
        def canon(toks):   # collapse many raw seqs onto a 2-token canonical key
            return tuple(toks[:2])
        mcts = self._dummy_mcts(canonicalize_fn=canon)
        mcts._register_completion((0, 1, 2), 0.5, -1.0, {"log_fvu": -2.0})
        mcts._register_completion((0, 1, 3), 0.9, -1.0, {"log_fvu": -4.0})  # same canonical key (0, 1)
        mcts._register_completion((0, 5, 2), 0.7, -1.0, {"log_fvu": -3.0})  # distinct key (0, 5)
        top = mcts.get_top_completions(limit=10)
        self.assertEqual(len(top), 2)                     # 2 distinct canonical candidates, not 3
        self.assertEqual(mcts._n_distinct_valid, 2)
        self.assertEqual(sorted(r for _, r, _ in top), [0.7, 0.9])   # kept best rep for key (0, 1)

    def test_invalid_completions_excluded_from_harvest(self) -> None:
        mcts = self._dummy_mcts()
        mcts._register_completion((0, 1, 2), 0.8, -1.0, {"log_fvu": -3.0})     # valid
        mcts._register_completion((0, 9, 2), -1.0, -1.0, {"log_fvu": float("nan")})  # invalid
        top = mcts.get_top_completions(limit=10)
        self.assertEqual(len(top), 1)
        self.assertEqual(mcts._n_distinct_valid, 1)

    def test_refine_budget_stops_early(self) -> None:
        """B5: search stops at refine_budget distinct valid completions, well before max_rollouts."""
        torch.manual_seed(0)

        # from bos(0): uniform over tokens {3,4,5,6}; each forces eos(2) -> 4 distinct completions
        def policy_fn(tokens, _):
            if tokens[-1] in (3, 4, 5, 6):
                return PolicyStep(log_probs=torch.tensor([-1e9, -1e9, 0.0, -1e9, -1e9, -1e9, -1e9]))
            return PolicyStep(log_probs=torch.tensor([-1e9, -1e9, -1e9, 0.0, 0.0, 0.0, 0.0]))

        def value_fn(tokens):
            return (0.5, {"log_fvu": -2.0})   # all valid

        config = MCTSConfig(simulations=999, max_rollouts=500, refine_budget=3, expansion_top_k=4)
        mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=value_fn,
                                    terminal_fn=lambda t: bool(t) and t[-1] == 2,
                                    config=config, eos_token_id=2, pad_token_id=None)
        mcts.run((0,))
        self.assertEqual(mcts._n_distinct_valid, 3)   # stopped exactly at the budget (4 were reachable)

    def test_rollout_guard_greedy_terminates(self) -> None:
        """M2: greedy rollout masks a pad/invalid argmax and advances instead of spinning forever."""
        pad = 3

        # argmax is always the pad token until masked; then argmax is token_a(1); token_a forces eos(2)
        def policy_fn(tokens, _):
            if tokens[-1] == 1:
                return PolicyStep(log_probs=torch.tensor([-1e9, -1e9, 0.0, -1e9]))
            return PolicyStep(log_probs=torch.tensor([-1e9, -1.0, -1e9, 0.0]))  # pad(3) highest, then a(1)

        config = MCTSConfig(simulations=6, max_depth=6, rollout_policy="greedy")
        mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=lambda t: (1.0, {"log_fvu": -3.0}),
                                    terminal_fn=lambda t: bool(t) and t[-1] == 2,
                                    config=config, eos_token_id=2, pad_token_id=pad)
        mcts.run((0,))   # must return (no hang)
        top = mcts.get_top_completions()
        self.assertTrue(top)
        for tokens, *_ in top:
            self.assertNotIn(pad, tokens)

    def test_truncated_non_eos_not_scored(self) -> None:
        """A max_depth-truncated sequence without EOS is non-terminal -> floored, never harvested."""
        # policy always emits token_a(1), never eos(2)
        def policy_fn(tokens, _):
            return PolicyStep(log_probs=torch.tensor([-1e9, 0.0, -1e9]))

        config = MCTSConfig(simulations=8, max_depth=4)
        mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=lambda t: (1.0, {"log_fvu": -3.0}),
                                    terminal_fn=lambda t: bool(t) and t[-1] == 2,
                                    config=config, eos_token_id=2)
        mcts.run((0,))   # must terminate
        self.assertEqual(mcts.get_top_completions(), [])
        self.assertEqual(mcts._n_distinct_valid, 0)

    def test_tree_stats_healthy_and_degenerate(self) -> None:
        """tree_stats reports balance; a degenerate (single-child) tree is distinguishable from a branching one."""
        from flash_ansr.decoding.mcts import MCTSNode, tree_stats
        # Hand-built tree: root with 3 children, visits track best_value (healthy).
        root = MCTSNode(tokens=(0,), prior=1.0, depth=0, visits=30)
        for i, (vis, val, pri) in enumerate([(20, 0.9, 0.6), (7, 0.5, 0.3), (3, 0.2, 0.1)]):
            c = MCTSNode(tokens=(0, i + 1), prior=pri, depth=1, visits=vis, best_value=val)
            root.children[i + 1] = c
        st = tree_stats(root, uct_c=1.4)
        self.assertEqual(st["root_n_children"], 3)
        self.assertEqual(st["root_n_visited"], 3)          # explored all three
        self.assertGreater(st["visit_value_corr"], 0.5)    # visits track value
        self.assertLess(st["root_top1_visit_frac"], 1.0)   # not fully greedy
        self.assertGreater(st["root_explore_mean"], 0.0)   # exploration term is live
        # Degenerate: only one child ever visited.
        root2 = MCTSNode(tokens=(0,), prior=1.0, depth=0, visits=10)
        root2.children[1] = MCTSNode(tokens=(0, 1), prior=0.9, depth=1, visits=10, best_value=0.5)
        root2.children[2] = MCTSNode(tokens=(0, 2), prior=0.1, depth=1, visits=0, best_value=0.0)
        st2 = tree_stats(root2, uct_c=1.4)
        self.assertEqual(st2["root_n_visited"], 1)         # greedy collapse signature

    def test_batched_loop_runs_and_leaves_no_vloss(self) -> None:
        """Leaf-parallel batched run: completes, respects the budget, and leaves NO virtual-loss residue
        (the #1 leaf-parallel bug: a leaked reservation biases all future selection)."""
        # vocab: 0=bos, 1=A, 2=eos, 3=B, 4=C -- branching tokens {1,3,4}, low eos

        def batched_policy_fn(token_lists):
            probs = torch.tensor([-1e9, 0.0, -5.0, 0.0, 0.0])  # favor 1,3,4; discourage eos
            return [torch.log_softmax(probs, dim=-1) for _ in token_lists]

        def batched_rollout_fn(token_lists):
            # each rollout completes its leaf with eos -> a distinct terminal completion per distinct leaf
            return [(tuple(t) + (2,), True, 0.0) for t in token_lists]

        def value_fn(tokens):
            return (0.5, {"log_fvu": -2.0})

        config = MCTSConfig(simulations=999, max_rollouts=400, refine_budget=8,
                            batch_width=4, expansion_top_k=3, max_depth=6, backup="max")
        mcts = MonteCarloTreeSearch(
            policy_fn=lambda t, s: PolicyStep(log_probs=torch.zeros(5)),
            value_fn=value_fn, terminal_fn=lambda t: bool(t) and t[-1] == 2,
            config=config, eos_token_id=2, pad_token_id=None,
            batched_policy_fn=batched_policy_fn, batched_rollout_fn=batched_rollout_fn)
        root = mcts.run((0,))

        # (a) budget reached
        self.assertGreaterEqual(mcts._n_distinct_valid, 8)
        top = mcts.get_top_completions()
        self.assertTrue(top)
        # registered log_prob must include the PREFIX (root->sim), not just the rollout suffix -- deeper
        # completions therefore have a strictly negative cumulative log_prob (suffix-only would be 0.0 here).
        self.assertTrue(any(entry[2] < 0 for entry in top), "batched harvest dropped the prefix log_prob")
        # (b) NO virtual-loss residue anywhere in the tree (sum-zero invariant)
        stack, leaked = [root], 0
        while stack:
            n = stack.pop()
            leaked += abs(n.vloss)
            stack.extend(n.children.values())
        self.assertEqual(leaked, 0, "virtual-loss reservation leaked -> biased selection")

    def test_batch_width_k1_dispatches_serial(self) -> None:
        """batch_width=1 uses the serial loop even when batched fns are provided (byte-identical anchor)."""
        called = {"batched": 0}

        def batched_policy_fn(token_lists):
            called["batched"] += 1
            return [torch.zeros(3) for _ in token_lists]

        def policy_fn(tokens, _):
            if tokens[-1] == 1:
                return PolicyStep(log_probs=torch.tensor([-1e9, -1e9, 0.0]))
            return PolicyStep(log_probs=torch.tensor([-1e9, 0.0, -1e9]))

        config = MCTSConfig(simulations=6, batch_width=1)
        mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=lambda t: (1.0, {"log_fvu": -3.0}),
                                    terminal_fn=lambda t: bool(t) and t[-1] == 2, config=config, eos_token_id=2,
                                    batched_policy_fn=batched_policy_fn, batched_rollout_fn=lambda x: [])
        mcts.run((0,))
        self.assertEqual(called["batched"], 0)   # serial path never touched the batched primitive

    def test_config_validation_new_fields(self) -> None:
        with self.assertRaises(ValueError):
            MCTSConfig(backup="invalid")
        with self.assertRaises(ValueError):
            MCTSConfig(value_objective="nonsense")
        with self.assertRaises(ValueError):
            MCTSConfig(batch_width=0)
        with self.assertRaises(ValueError):
            MCTSConfig(batch_width=4, rollout_policy="greedy")   # batched divergence needs stochastic rollouts
        with self.assertRaises(ValueError):
            MCTSConfig(reward_log_fvu_hi=-8.0, reward_log_fvu_lo=0.0)   # hi must exceed lo
        with self.assertRaises(ValueError):
            MCTSConfig(refine_budget=0)
        with self.assertRaises(ValueError):
            MCTSConfig(max_rollouts=0)
        with self.assertRaises(ValueError):
            MCTSConfig(fpu_reduction=-0.1)


if __name__ == "__main__":
    unittest.main()
