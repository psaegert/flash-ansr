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
    ConvergenceError,
    create_generation_config,
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

    def test_infer_returns_ledger_on_all_fail_but_fit_raises(self):
        """ISSUE-009: on 0 converged beams, infer() returns the full ledger (empty candidates);
        fit()'s read-back still raises. Exercised at the shared empty-results core."""
        nsr = FlashANSR.load(
            directory=self.model_dir,
            generation_config=create_generation_config(
                method="beam_search", beam_width=4, max_len=32, batch_size=4,
                unique=True, limit_expansions=True, use_cache=True),
            n_restarts=2,
        ).to(self.device)
        # fit's path (allow_empty=False) raises on empty; infer's path (allow_empty=True) returns empty.
        with self.assertRaises(ConvergenceError):
            nsr._compile_results_pure([], 0.0, 0.0, 0.0)
        results, results_df = nsr._compile_results_pure([], 0.0, 0.0, 0.0, allow_empty=True)
        self.assertEqual(results, [])
        self.assertEqual(len(results_df), 0)

    def test_infer_matches_fit_faithfully(self):
        # infer() reuses _fit_generate/_fit_refine (minus _apply_fit_result) + adds the candidate
        # ledger. Pin ONE generation (a captured GenState reused by both paths) so fit() and infer()
        # refine identically, then assert: identical candidate set (expression, order, fvu), best
        # y_pred == predict(nth_best_beam=0), the ledger covers the full generation pool, and infer()
        # writes NOTHING to instance state (self._results unchanged).
        from flash_ansr.inference import FIT_OK

        # Run on CPU: the constant-pruning rescore (model.forward) is not bit-reproducible on CUDA,
        # which would reorder the tail candidates and make a strict fit-vs-infer comparison flaky for
        # reasons unrelated to infer(). CPU forward is deterministic, so fit() and infer() refining the
        # SAME pinned GenState with the SAME refine_seed are bit-identical.
        cpu = torch.device("cpu")
        x = self.x_tensor.to(cpu)
        y = self.y_tensor.to(cpu)
        nsr = FlashANSR.load(
            directory=self.model_dir,
            generation_config=SoftmaxSamplingConfig(),
            n_restarts=4,
        ).to(cpu)

        # Pin BOTH phases so fit() and infer() share one IDENTICAL FitResult. _fit_refine itself is
        # nondeterministic run-to-run (it dedups to distinct expressions, and which raw_beam represents
        # a duplicate -- and its seed -> fvu -- depends on parallel-completion order). Re-running it
        # would test that nondeterminism, not infer()'s faithfulness. Pinning isolates the thing under
        # test: that infer() builds the SAME candidate VIEW fit() commits, from the same FitResult.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gen_state = nsr._fit_generate(x, y)
            fit_result = nsr._fit_refine(gen_state, refine_seed=0)   # the ONE refinement both paths share
            nsr._fit_generate = lambda *args, **kwargs: gen_state
            nsr._fit_refine = lambda *args, **kwargs: fit_result

            nsr.fit(x, y, refine_seed=0)                             # commits fit_result.results to _results
            fit_pred = nsr.predict(x) if nsr._results else None
            result = nsr.infer(x, y, refine_seed=0)                  # builds candidates from the SAME fit_result.results

        ref = fit_result.results
        # infer()'s candidates are exactly fit's results (same objects, same order): expression / fvu /
        # score / constants must match element-for-element.
        self.assertEqual([list(c.expression) for c in result.candidates], [list(r['expression']) for r in ref])
        np.testing.assert_allclose([c.fvu for c in result.candidates], [float(r['fvu']) for r in ref], equal_nan=True, rtol=1e-9)
        np.testing.assert_allclose([c.score for c in result.candidates], [float(r['score']) for r in ref], equal_nan=True, rtol=1e-9)

        self.assertGreaterEqual(len(result.ledger), len(gen_state.raw_beams))     # ledger covers the gen pool
        self.assertEqual(len(result.ledger.token_lists), len(result.ledger.fit_status))
        self.assertEqual(any(s == FIT_OK for s in result.ledger.fit_status), bool(result.candidates))

        if result.best is not None and fit_pred is not None:
            self.assertEqual(result.best.expression, list(ref[0]['expression']))
            fp = fit_pred.detach().cpu().numpy() if isinstance(fit_pred, torch.Tensor) else np.asarray(fit_pred)
            np.testing.assert_allclose(np.asarray(result.best.y_pred).ravel(), fp.ravel(), rtol=1e-5, atol=1e-6)
            # expression_infix is the variable-mapped infix string -- exactly the model's native
            # get_expression(map_variables=True); this closes srbf's last model-derived record field.
            self.assertEqual(result.best.expression_infix,
                             nsr.get_expression(nth_best_beam=0, nth_best_constants=0, map_variables=True))
            self.assertIsInstance(result.best.expression_infix, str)

        # infer() did NOT call _apply_fit_result -> instance state is still exactly what fit() set
        self.assertEqual([list(r['expression']) for r in nsr._results], [list(r['expression']) for r in ref])

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

    def test_softmax_sampling_batched_path(self) -> None:
        # batch_size < choices routes to the batched (chunked, KV-cache-freeing) path,
        # which is what the deployed c=1024 / batch_size=128 config uses. Asserts the
        # path produces valid results and never returns more than `choices` unique beams.
        generation_config = SoftmaxSamplingConfig(
            choices=16,
            top_k=8,
            top_p=0.95,
            max_len=24,
            batch_size=4,  # < choices -> batched path (4 chunks, cross-chunk dedup)
            temperature=0.8,
            simplify=True,
            unique=True,
        )

        nsr = self._fit_with_generation_config(generation_config, n_restarts=6)

        self.assertEqual(nsr.generation_config.method, 'softmax_sampling')
        self._assert_valid_results(nsr)

    def test_softmax_sampling_auto_batch_single_shot(self) -> None:
        # Regression: batch_size='auto' (the library default) at a small `choices` resolves to a
        # batch >= choices, taking the single-shot path. Previously that path forwarded the
        # unresolved 'auto' string to sample_top_kp -> `range(0, n, 'auto')` -> TypeError. The
        # resolved int must reach sample_top_kp regardless of which path is taken.
        generation_config = SoftmaxSamplingConfig(
            choices=16,
            top_k=8,
            top_p=0.95,
            max_len=24,
            batch_size='auto',  # resolves to >= choices -> single-shot path
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
