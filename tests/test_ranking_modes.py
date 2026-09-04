"""The three ranking modes: primitives, resolution, and the sort they share.

RANKING_SPEC.md §5 item 4: the non-dominated front against a brute-force oracle, an all-unpriceable
pool raising, and a tie-break outside the declared metric set giving a total order.
"""
from __future__ import annotations

import numpy as np
import pytest

from flash_ansr.scoring import (
    ND_MAX_CANDIDATES,
    RANKING_METRICS,
    RankingError,
    non_dominated_ranks,
    objective_vector,
)


def _brute_force_ranks(V: np.ndarray) -> np.ndarray:
    """Front index per row by the definition: peel the rows nobody alive dominates."""
    n = V.shape[0]
    ranks = np.full(n, -1, dtype=int)
    alive = list(range(n))
    front = 0
    while alive:
        current = []
        for i in alive:
            dominated = False
            for j in alive:
                if j == i:
                    continue
                if np.all(V[j] <= V[i]) and np.any(V[j] < V[i]):
                    dominated = True
                    break
            if not dominated:
                current.append(i)
        for i in current:
            ranks[i] = front
        alive = [i for i in alive if i not in current]
        front += 1
    return ranks


class TestNonDominatedRanks:
    @pytest.mark.parametrize("seed", range(40))
    def test_matches_brute_force_on_tie_dense_pools(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n, m = int(rng.integers(1, 60)), int(rng.integers(1, 4))
        # Small integer grids make ties and exact dominance common, which is where a vectorised
        # peel goes wrong (strict vs non-strict comparisons).
        V = rng.integers(0, 4, size=(n, m)).astype(float)
        np.testing.assert_array_equal(non_dominated_ranks(V), _brute_force_ranks(V))

    def test_unrankable_rows_sink_to_the_last_front(self) -> None:
        V = np.array([[1.0, 1.0], [np.inf, 1.0], [np.inf, np.inf]])
        ranks = non_dominated_ranks(V)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert ranks[2] == 2

    def test_all_inf_pool_is_one_front_not_zero_fronts(self) -> None:
        V = np.full((5, 2), np.inf)
        ranks = non_dominated_ranks(V)
        assert set(ranks.tolist()) == {0}

    def test_empty_pool(self) -> None:
        assert non_dominated_ranks(np.zeros((0, 2))).shape == (0,)

    def test_quadratic_guard(self) -> None:
        with pytest.raises(RankingError):
            non_dominated_ranks(np.zeros((ND_MAX_CANDIDATES + 1, 1)))


class TestObjectiveVector:
    def test_missing_and_non_finite_become_plus_inf(self) -> None:
        rows = [
            {'fvu': 0.1, 'expression': ['+', 'x1', '<constant>'], 'mdl': 1000.0},
            {'fvu': float('nan'), 'expression': ['x1'], 'mdl': None},
            {'fvu': 0.2, 'expression': ['x1'], 'mdl': float('inf')},
        ]
        V = objective_vector(rows, ('fvu', 'mdl', 'n_nodes'))
        assert V[0].tolist() == [0.1, 1.0, 3.0]
        assert V[1, 0] == np.inf and V[1, 1] == np.inf and V[1, 2] == 1.0
        assert V[2, 1] == np.inf

    def test_every_registry_metric_reads_a_plain_row(self) -> None:
        row = {'fvu': 0.5, 'log_prob': -2.0, 'mdl': 12000.0,
               'expression': ['*', '<constant>', 'pow', 'x1', '2']}
        values = {name: RANKING_METRICS[name](row) for name in RANKING_METRICS}
        assert values['fvu'] == 0.5
        assert values['neg_log_prob'] == 2.0
        assert values['mdl'] == 12.0          # bits, not milli-bits
        assert values['n_nodes'] == 5
        assert values['n_constant_placeholders'] == 1
        assert values['n_typed_literals'] == 1
        assert values['n_constants'] == 2


# --- resolution: each knob belongs to exactly one mode ------------------------------------------
from flash_ansr.scoring import (  # noqa: E402
    MDL_STRENGTH_DEFAULT,
    WEIGHTABLE_METRICS,
    RankingConfig,
    resolve_ranking,
    score_from_fvu,
    score_row,
)


class TestResolveRanking:
    def test_default_is_the_engineered_mdl_mode(self) -> None:
        cfg = resolve_ranking()
        assert cfg.mode == 'mdl'
        assert cfg.mdl_strength == MDL_STRENGTH_DEFAULT == 4.5e-3
        assert cfg.effective_weights == {'mdl': 4.5e-3}

    def test_weighted_defaults_to_no_penalty_at_all(self) -> None:
        cfg = resolve_ranking('weighted')
        assert cfg.effective_weights == {}

    def test_the_pre_014_ranking_is_one_weight(self) -> None:
        cfg = resolve_ranking('weighted', weights={'n_nodes': 0.05})
        assert cfg.effective_weights == {'n_nodes': 0.05}

    def test_pareto_defaults(self) -> None:
        cfg = resolve_ranking('pareto')
        assert cfg.metrics == ('fvu', 'n_nodes') and cfg.tie_break == 'fvu'
        assert cfg.effective_weights == {}

    @pytest.mark.parametrize("mode, knob", [
        ('mdl', {'weights': {'n_nodes': 1.0}}),
        ('mdl', {'metrics': ('fvu',)}),
        ('mdl', {'tie_break': 'fvu'}),
        ('weighted', {'mdl_strength': 1e-3}),
        ('weighted', {'metrics': ('fvu',)}),
        ('weighted', {'tie_break': 'fvu'}),
        ('pareto', {'mdl_strength': 1e-3}),
        ('pareto', {'weights': {'n_nodes': 1.0}}),
    ])
    def test_a_knob_of_another_mode_raises_instead_of_lying_dormant(self, mode: str, knob: dict) -> None:
        with pytest.raises(ValueError, match="belongs to ranking_mode"):
            resolve_ranking(mode, **knob)

    def test_fvu_is_not_weightable(self) -> None:
        assert 'fvu' not in WEIGHTABLE_METRICS
        with pytest.raises(ValueError, match="base term"):
            resolve_ranking('weighted', weights={'fvu': 1.0})

    @pytest.mark.parametrize("bad", [
        dict(mode='nope'),
        dict(mode='weighted', weights={'n_transcendental': 1.0}),
        dict(mode='weighted', weights={'n_nodes': float('nan')}),
        dict(mode='mdl', mdl_strength=-1.0),
        dict(mode='pareto', metrics=('n_nodes',)),
        dict(mode='pareto', metrics=('fvu', 'fvu')),
        dict(mode='pareto', metrics=('fvu', 'nope')),
        dict(mode='pareto', tie_break='nope'),
    ])
    def test_unknown_or_invalid_values_raise(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            resolve_ranking(**bad)

    def test_tie_break_outside_the_declared_set_is_allowed(self) -> None:
        cfg = resolve_ranking('pareto', metrics=('fvu', 'n_nodes'), tie_break='mdl')
        assert cfg.tie_break == 'mdl'

    @pytest.mark.parametrize("cfg", [
        resolve_ranking(),
        resolve_ranking('mdl', mdl_strength=1e-2),
        resolve_ranking('weighted', weights={'n_nodes': 0.05, 'mdl': 1e-3}),
        resolve_ranking('pareto', metrics=('fvu', 'mdl'), tie_break='n_nodes'),
    ])
    def test_as_dict_round_trips_and_carries_only_its_modes_knobs(self, cfg: RankingConfig) -> None:
        d = cfg.as_dict()
        assert set(d) <= {'mode', 'mdl_strength', 'weights', 'metrics', 'tie_break'}
        assert RankingConfig.from_dict(d) == cfg
        with pytest.raises(ValueError):
            RankingConfig.from_dict({**d, 'stray': 1})


class TestScoreRow:
    def test_is_the_frozen_scorer_plus_the_two_count_addends(self) -> None:
        row = {'fvu': 0.01, 'expression': ['*', '<constant>', 'pow', 'x1', '2'],
               'constant_count': 2, 'log_prob': -3.0, 'mdl': 12000.0}
        w = {'n_nodes': 0.05, 'n_constants': 0.1, 'neg_log_prob': 0.01, 'mdl': 4.5e-3}
        assert score_row(row, w) == score_from_fvu(0.01, 5, 2, -3.0, 0.05, 0.1, 0.01, 12000.0, 4.5e-3)
        # the two registry-only metrics are plain addends on top
        assert score_row(row, {'n_constant_placeholders': 1.0}) == score_from_fvu(0.01, 5, 2, -3.0, 0.0, 0.0, 0.0) + 1.0
        assert score_row(row, {'n_typed_literals': 2.0}) == score_from_fvu(0.01, 5, 2, -3.0, 0.0, 0.0, 0.0) + 2.0

    def test_no_weights_is_exactly_log10_fvu(self) -> None:
        row = {'fvu': 0.001, 'expression': ['x1'], 'constant_count': 0, 'log_prob': None, 'mdl': None}
        assert score_row(row, {}) == score_from_fvu(0.001, 1, 0, None, 0.0, 0.0, 0.0) == pytest.approx(-3.0)

    def test_unpriced_row_under_a_live_mdl_weight_is_worst(self) -> None:
        row = {'fvu': 0.001, 'expression': ['x1'], 'constant_count': 0, 'log_prob': None, 'mdl': None}
        assert score_row(row, {'mdl': 4.5e-3}) == float('inf')


# --- the spec's two replacement invariants (RANKING_SPEC.md section 5) ---------------------------
import json  # noqa: E402
import os  # noqa: E402

from simplipy import SimpliPyEngine  # noqa: E402
from simplipy.engine import Mode  # noqa: E402

_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "emitted_skeletons_t7.json")


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


def _mu(engine: SimpliPyEngine, tokens: list[str]) -> float:
    return float(engine.complexity(tokens, certified=True, mode=Mode.f64, canon='default'))


class TestModeOneVersusModeTwo:
    """Mode 2 at the engineered count weights approximates Mode 1 on the EMITTED spelling, where
    every constant is the same flat placeholder. Not equality (owner: close, not exact): agreement
    to the measured emitted-spelling tolerance, so a broken pricer or a mis-scaled weight is caught
    without asserting the two modes coincide."""

    #: mu ~= 4,444 * n_nodes + 59,589 * n_constants + 8,073 mB on 8,476 T7 candidates (R^2 0.9969);
    #: at 4.5e-3 per bit that is 0.020 per node and 0.268 per constant.
    NODE_WEIGHT = 4.444 * MDL_STRENGTH_DEFAULT
    CONSTANT_WEIGHT = 59.589 * MDL_STRENGTH_DEFAULT

    def test_pair_inversion_rate_on_the_emitted_spelling(self, engine: SimpliPyEngine) -> None:
        with open(_FIXTURE) as fh:
            expressions = json.load(fh)["expressions"]
        assert len(expressions) >= 200
        rows = []
        for tokens in expressions:
            rows.append({'fvu': 1.0, 'expression': tokens, 'log_prob': None,
                         'constant_count': sum(t == '<constant>' for t in tokens),
                         'mdl': _mu(engine, tokens)})
        # identical fvu on every row, so the two scores ARE the two penalties
        mode1 = resolve_ranking('mdl').effective_weights
        mode2 = resolve_ranking('weighted', weights={
            'n_nodes': self.NODE_WEIGHT, 'n_constants': self.CONSTANT_WEIGHT}).effective_weights
        s1 = np.array([score_row(r, mode1) for r in rows])
        s2 = np.array([score_row(r, mode2) for r in rows])
        assert np.isfinite(s1).all() and np.isfinite(s2).all()
        d1 = np.sign(s1[:, None] - s1[None, :])
        d2 = np.sign(s2[:, None] - s2[None, :])
        strict = (d1 != 0) & (d2 != 0)
        inversions = float(((d1 != d2) & strict).sum() / max(int(strict.sum()), 1))
        # measured 1.02% on 8,476 candidates; a broken pricer or a 1000x unit slip lands far above
        assert inversions <= 0.03, f"Mode 1 vs Mode 2 pair inversion {inversions:.3%}"
        # and they are NOT identical -- Mode 1 prices what counts cannot
        assert inversions > 0.0 or not np.allclose(s1 - s1.mean(), s2 - s2.mean())


class TestPrecisionMonotonicity:
    """Coarse-graining a constant must never RAISE its Mode 1 penalty. This is the property the
    constant coarse-graining plan depends on, and the one a count-based Mode 2 provably lacks."""

    @pytest.mark.parametrize("skeleton", [
        ['*', '{c}', 'x1'],
        ['+', 'sin', 'x1', '{c}'],
        ['pow', 'x1', '{c}'],
        ['*', 'exp', '*', '{c}', 'x1', '{c}'],
    ])
    def test_coarser_constants_never_cost_more(self, engine: SimpliPyEngine, skeleton: list[str]) -> None:
        ladder = ['3.141592653589793', '3.14159265', '3.1416', '3.14', '3']
        prices = []
        for spelled in ladder:
            tokens = [spelled if t == '{c}' else t for t in skeleton]
            prices.append(score_row({'fvu': 1.0, 'expression': tokens, 'log_prob': None,
                                     'constant_count': 0, 'mdl': _mu(engine, tokens)},
                                    resolve_ranking('mdl').effective_weights))
        assert all(b <= a for a, b in zip(prices, prices[1:])), (skeleton, ladder, prices)
        assert prices[-1] < prices[0], "17 digits must cost more than an integer"

    def test_realizing_a_placeholder_is_a_discount(self, engine: SimpliPyEngine) -> None:
        # a printed f64 REPLACES the flat 67,000 mB placeholder: mu goes down, not up
        assert _mu(engine, ['*', '3.141592653589793', 'x1']) < _mu(engine, ['*', '<constant>', 'x1'])
