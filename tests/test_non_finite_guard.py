"""Non-finite skeletons must never enter the candidate stream.

THE DEFECT. SimpliPy folds a degenerate sub-expression to a reserved non-finite spelling
instead of failing::

    engine.simplify(['/', 'x1', '0']) -> ['*', 'float("inf")', 'x1']
    engine.simplify(['/', '0',  '0']) -> ['float("nan")']
    engine.simplify(['log', '0'])     -> ['float("-inf")']

``float("inf")`` / ``float("-inf")`` / ``float("nan")`` are real vocabulary tokens (ids
25/26/27), so ``tokenizer.encode`` accepts them and the skeleton re-enters the candidate
stream marked ``is_valid=True``. symbolic-data's generator already rejects exactly this set
before yielding a sample; flash-ansr had no equivalent on its own simplification path.

The reachable input needs NO numeric literal -- ``x2 - x2`` folds to ``0`` first -- so the
incidental mitigation (``decode(special_tokens='<constant>')`` strips bare literal tokens)
does not cover it: ``['/', 'x1', '-', 'x2', 'x2']`` survives decoding intact and comes back
out of the model as ``['*', 'float("inf")', 'x1']``.

THE GUARD sits at the simplification seam (``flash_ansr.utils.skeleton``), the single point
every candidate producer shares -- softmax post-processing,
the constant-pruning lane, the parallel simplify pool and dataset conversion all call it. It
RAISES there (a function that returns a skeleton cannot return "no skeleton" without spreading
an optional-return contract nobody can be forced to check), and every CANDIDATE producer
catches that and DROPS the candidate while counting the drop -- a sampled expression that folds
to nan is a normal rejectable sample, the same kind of event as the ``is_valid`` rejection and
the re-encode ``KeyError`` those sites already skip on. The INGEST direction does not drop: a
non-finite arriving from a data producer is a broken upstream contract, and swallowing it would
silently reshape the training distribution.

These tests observe from outside: the public ``sample_top_kp`` / ``parse_data`` APIs, the model's
post-processing entry point, and the published drop counter.
"""
import unittest

import pandas as pd

from simplipy import SimpliPyEngine

from flash_ansr import FlashANSRModel, LampleChartonCatalog, get_path
from flash_ansr.convert_data import SOOSEParser
from flash_ansr.utils.skeleton import (
    NON_FINITE_TOKENS,
    NonFiniteExpressionError,
    find_non_finite,
    mask_literals_positional,
    non_finite_drops,
    reset_non_finite_drops,
    simplify_and_mask,
)

# A candidate with NO numeric literal token whose simplification is non-finite: `x2 - x2`
# folds to 0 first. This is the shape that defeats the decode-strips-literals mitigation.
LITERAL_FREE_INF = ['/', 'x1', '-', 'x2', 'x2']       # -> ['*', 'float("inf")', 'x1']
LITERAL_FREE_NEG_INF = ['log', '-', 'x1', 'x1']       # -> ['float("-inf")']
LITERAL_FREE_NAN = ['/', '-', 'x1', 'x1', '-', 'x2', 'x2']   # -> ['float("nan")']


def _engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-4-3', install=True)


def _model() -> FlashANSRModel:
    return FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))


class TestPremise(unittest.TestCase):
    """The upstream behaviour the guard exists for. If SimpliPy ever stops emitting these,
    the guard goes inert rather than wrong -- but we want to be told."""

    def test_simplipy_still_folds_degenerate_expressions_to_non_finite(self) -> None:
        engine = _engine()
        for expression, expected in [
            (['/', 'x1', '0'], 'float("inf")'),
            (['/', '0', '0'], 'float("nan")'),
            (['log', '0'], 'float("-inf")'),
            (LITERAL_FREE_INF, 'float("inf")'),
        ]:
            simplified = engine.simplify(list(expression))
            self.assertIn(expected, simplified, f"{expression} -> {simplified}")
            # ... and the engine calls the result a valid expression, which is why nothing
            # downstream rejected it.
            self.assertTrue(engine.is_valid(simplified))

    def test_non_finite_tokens_are_encodable_vocabulary(self) -> None:
        # The reason these skeletons travel: the tokenizer has ids for them, so the
        # re-encode step in post-processing does not drop them.
        tokenizer = _model().tokenizer
        for token in ('float("inf")', 'float("-inf")', 'float("nan")'):
            self.assertIsNotNone(tokenizer.token2idx.get(token))


class TestTokenSet(unittest.TestCase):
    def test_matches_symbolic_datas_forbidden_set(self) -> None:
        # symbolic_data.generative.LampleChartonCatalog._sympy_simplify_skeleton:
        #   ['float("inf")', 'float("-inf")', 'float("nan")', 'zoo', 'nan', 'oo']
        # (its SimpliPy branch checks the first three -- the only spellings SimpliPy emits.)
        self.assertEqual(
            set(NON_FINITE_TOKENS),
            {'float("inf")', 'float("-inf")', 'float("nan")', 'zoo', 'nan', 'oo'})

    def test_find_non_finite_is_exact_token_matching(self) -> None:
        self.assertEqual(find_non_finite(['*', 'float("inf")', 'x1']), ['float("inf")'])
        self.assertEqual(find_non_finite(['zoo']), ['zoo'])
        self.assertEqual(find_non_finite(['+', 'x1', 'nan']), ['nan'])
        # Not substring matching: a variable or literal that merely contains the letters is fine.
        self.assertEqual(find_non_finite(['+', 'x1', '<constant>', 'np.e', '0', '1']), [])


class TestSimplifySeamRaises(unittest.TestCase):
    """`simplify_and_mask` is the one seam every candidate producer shares."""

    def test_raises_on_folded_non_finite(self) -> None:
        engine = _engine()
        for expression in (['/', 'x1', '0'], ['/', '0', '0'], ['log', '0'],
                           LITERAL_FREE_INF, LITERAL_FREE_NEG_INF, LITERAL_FREE_NAN):
            with self.assertRaises(NonFiniteExpressionError):
                simplify_and_mask(engine, list(expression))

    def test_error_names_the_token_and_the_expression(self) -> None:
        engine = _engine()
        with self.assertRaises(NonFiniteExpressionError) as ctx:
            simplify_and_mask(engine, list(LITERAL_FREE_INF))
        self.assertEqual(ctx.exception.tokens, ['float("inf")'])
        self.assertIn('float("inf")', str(ctx.exception))

    def test_finite_expressions_are_untouched(self) -> None:
        # '+ x1 x1' -> '* 2 x1' -> literals masked -> '* <constant> x1'
        self.assertEqual(simplify_and_mask(_engine(), ['+', 'x1', 'x1']), ['*', '<constant>', 'x1'])


class TestPostprocessDropsAndCounts(unittest.TestCase):
    """The softmax-sampling lane: `_postprocess_sampled` is where a sampled sequence is
    simplified and re-encoded."""

    def setUp(self) -> None:
        self.model = _model()
        reset_non_finite_drops()

    def _sequence(self, expression: list[str]) -> list[int]:
        return self.model.tokenizer.encode(
            ['<bos>', '<expression>', *expression, '</expression>', '<eos>'])

    def test_non_finite_candidate_is_dropped(self) -> None:
        for expression in (LITERAL_FREE_INF, LITERAL_FREE_NEG_INF, LITERAL_FREE_NAN):
            sequences, scores, is_valid = self.model._postprocess_sampled(
                [self._sequence(expression)], [0.0], simplify=True)
            self.assertEqual(sequences, [], f"{expression} entered the candidate stream")
            self.assertEqual(scores, [])
            self.assertEqual(is_valid, [])

    def test_non_finite_candidate_is_dropped_with_valid_only_false(self) -> None:
        # valid_only=False keeps grammar-invalid candidates for the ledger; it must not
        # smuggle the non-finite one back in.
        sequences, _, _ = self.model._postprocess_sampled(
            [self._sequence(LITERAL_FREE_INF)], [0.0], simplify=True, valid_only=False)
        for sequence in sequences:
            self.assertEqual(find_non_finite(self.model.tokenizer.decode(sequence)), [])

    def test_the_drop_is_counted(self) -> None:
        before = non_finite_drops()
        self.model._postprocess_sampled([self._sequence(LITERAL_FREE_INF)], [0.0], simplify=True)
        self.assertEqual(non_finite_drops(), before + 1)

    def test_finite_candidates_still_survive(self) -> None:
        sequences, _, is_valid = self.model._postprocess_sampled(
            [self._sequence(['+', 'x1', 'x1'])], [0.0], simplify=True)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(is_valid, [True])
        self.assertEqual(non_finite_drops(), 0)

    def test_simplify_false_is_unaffected(self) -> None:
        # Without simplification nothing folds, so nothing is dropped: the guard costs the
        # simplify=False path nothing.
        sequences, _, _ = self.model._postprocess_sampled(
            [self._sequence(LITERAL_FREE_INF)], [0.0], simplify=False)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(non_finite_drops(), 0)


class TestParallelSimplifyPoolAgreesWithSerial(unittest.TestCase):
    """The chunked generate() path simplifies in forked workers and feeds the results back as a
    lookup map. A worker cannot raise across the fork without killing the run, and it cannot
    count into the parent's counter either -- so it reports the drop by OMITTING the entry, and
    the parent falls back to the serial call that raises, drops and counts."""

    def test_worker_reports_non_finite_by_omission(self) -> None:
        from flash_ansr.flash_ansr import _simplify_pool_init, _simplify_pool_worker

        _simplify_pool_init(_engine())
        self.assertIsNone(_simplify_pool_worker(tuple(LITERAL_FREE_INF)))
        self.assertEqual(_simplify_pool_worker(('+', 'x1', 'x1')),
                         tuple(simplify_and_mask(_engine(), ['+', 'x1', 'x1'])))

    def test_map_miss_falls_back_to_the_guarded_serial_path(self) -> None:
        model = _model()
        reset_non_finite_drops()
        sequence = model.tokenizer.encode(
            ['<bos>', '<expression>', *LITERAL_FREE_INF, '</expression>', '<eos>'])
        # A map that omits the offending key (what the worker produces) must not re-open the hole.
        sequences, _, _ = model._postprocess_sampled(
            [sequence], [0.0], simplify=True, simplify_map={})
        self.assertEqual(sequences, [])
        self.assertEqual(non_finite_drops(), 1)


class TestIngestRaisesInsteadOfDropping(unittest.TestCase):
    """The training-data ingest seam. flash-ansr is the CONSUMER here: symbolic-data rejects
    these before yielding a sample, so one arriving means the producer's contract broke.
    Dropping the sample would hide that and silently reshape the training distribution."""

    def test_mask_literals_positional_raises(self) -> None:
        with self.assertRaises(NonFiniteExpressionError):
            mask_literals_positional(_engine(), ['*', 'float("inf")', 'x1'])

    def test_finite_training_expressions_are_unaffected(self) -> None:
        skeleton, values = mask_literals_positional(_engine(), ['+', 'x1', '2.5'])
        self.assertEqual(skeleton, ['+', 'x1', '<constant>'])
        self.assertEqual(values, [2.5])


class TestDatasetConversionCountsItAsInvalid(unittest.TestCase):
    """`convert_data` imports external benchmark files. A row that folds to non-finite is the
    designed, reported attrition of importing an external set (the invalid tally), not a crash."""

    def test_non_finite_row_is_skipped_and_counted(self) -> None:
        reset_non_finite_drops()
        catalog = SOOSEParser().parse_data(
            test_set_df=pd.DataFrame({'eq': ['x_1/(x_2 - x_2)', 'x_1 + x_2']}),
            simplipy_engine=_engine(),
            base_catalog=LampleChartonCatalog.from_config(get_path('configs', 'test', 'catalog_test.yaml')))
        self.assertEqual(len(catalog.skeletons), 1)
        for skeleton in catalog.skeletons:
            self.assertEqual(find_non_finite(list(skeleton)), [])
        self.assertEqual(non_finite_drops(), 1)


if __name__ == '__main__':
    unittest.main()
