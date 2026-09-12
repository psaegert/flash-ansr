"""`refiner_typed_spans`: what happens to a literal the model predicted in a TYPED position.

Post-processing now hands the refiner the candidate as the model stated it (see
`test_realized_emission.py`): a literal where it spelled a number, a `<constant>` where it masked
one. So `refine_scope='fittable'` already keeps a spelled exponent verbatim -- freezing is structural,
not a step. The policy exists for the arms that depart from it: `'refine'` thaws the typed literals
back into slots (the pre-2026-09-12 behaviour, kept as a control), `'freeze_then_free'` adds a
duplicate that refits them from the prediction, and `'combinations'` enumerates subsets (deferred).
"""
import numpy as np
import pytest
from simplipy import SimpliPyEngine

import flash_ansr.flash_ansr as harness
from flash_ansr.refine import (
    DEFAULT_TYPED_SPAN_POLICY, TYPED_SPAN_POLICIES, TYPED_ROLES,
    literal_token, refinement_slots, thaw_typed_literals, typed_literal_sites, typed_thaw_subsets)

N_VARIABLES = 4


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-4-3', install=True)


def test_the_default_freezes_and_offers_the_duplicate() -> None:
    assert DEFAULT_TYPED_SPAN_POLICY == 'freeze_then_free'
    assert TYPED_SPAN_POLICIES == ('refine', 'freeze', 'freeze_then_free', 'combinations')


def test_typed_roles_are_the_ones_mask_fittable_keeps() -> None:
    from simplipy import masking
    for role in masking.Role:
        kept = masking.mask_fittable('2', role) is None
        assert kept == (role in TYPED_ROLES), role


class TestLiteralToken:
    """The spelling a prediction is realized as -- it must read back EXACTLY."""

    @pytest.mark.parametrize("value,token", [
        (2.0, '2'), (-2.0, '-2'), (3.0, '3'), (0.0, '0'),          # integral -> the canon's spelling
        (0.5, '0.5'), (2.0000001, '2.0000001'), (1e-20, '1e-20'),  # everything else -> repr
    ])
    def test_spelling(self, value: float, token: str) -> None:
        assert literal_token(value) == token

    def test_round_trips_float64(self) -> None:
        from flash_ansr.refine import literal_value
        for value in (1.4142135623730951, -0.7, 1 / 3, 2.0, 1e300):
            assert literal_value(literal_token(value)) == value

    @pytest.mark.parametrize("value", [float('inf'), float('-inf'), float('nan')])
    def test_non_finite_has_no_spelling(self, value: float) -> None:
        # `inf`/`nan` are reserved numeric spellings simplipy's role walk refuses (H-007)
        with pytest.raises(ValueError, match="literal spelling"):
            literal_token(value)


class TestFreezingIsStructural:
    """On a REALIZED emission the doctrine needs no extra step: the scope already knows the roles."""

    def test_a_spelled_exponent_is_not_a_slot_and_a_spelled_coefficient_is(self, engine: SimpliPyEngine) -> None:
        # the model masked the shift and spelled the exponent -- the <mask_fittable> shape
        assert refinement_slots(['pow', '-', 'x1', '<constant>', '2'], engine, 'fittable') == [3]
        # and where it spelled a coefficient, that IS fittable, seeded at its prediction
        assert refinement_slots(['*', '3.7', 'pow', 'x1', '2'], engine, 'fittable') == [1]

    def test_a_structurally_spelled_rational_exponent_is_typed_throughout(self, engine: SimpliPyEngine) -> None:
        assert refinement_slots(['pow', 'x1', '/', '3', '2'], engine, 'fittable') == []
        assert typed_literal_sites(['pow', 'x1', '/', '3', '2'], engine) == [(3, 3.0), (4, 2.0)]


class TestThaw:
    def test_thaw_restores_the_placeholder_and_reports_the_value(self, engine: SimpliPyEngine) -> None:
        thawed, values = thaw_typed_literals(['pow', '-', 'x1', '<constant>', '2'], engine)
        assert thawed == ['pow', '-', 'x1', '<constant>', '<constant>']
        assert values == {4: 2.0}

    def test_a_subset_thaws_only_what_it_names(self, engine: SimpliPyEngine) -> None:
        thawed, values = thaw_typed_literals(['pow', 'x1', '/', '3', '2'], engine, subset=[4])
        assert thawed == ['pow', 'x1', '/', '3', '<constant>']
        assert values == {4: 2.0}

    def test_nothing_typed_means_nothing_to_thaw(self, engine: SimpliPyEngine) -> None:
        assert thaw_typed_literals(['*', '2.5', 'sin', 'x1'], engine) == (['*', '2.5', 'sin', 'x1'], {})

    def test_specials_are_symbolic_and_stay(self, engine: SimpliPyEngine) -> None:
        assert typed_literal_sites(['pow', 'x1', 'np.pi'], engine) == []
        assert thaw_typed_literals(['pow', 'x1', 'np.pi'], engine) == (['pow', 'x1', 'np.pi'], {})


class TestThawSubsets:
    def test_freeze_and_refine_ask_for_no_duplicate(self, engine: SimpliPyEngine) -> None:
        assert typed_thaw_subsets(['pow', 'x1', '2'], engine, 'refine') == []
        assert typed_thaw_subsets(['pow', 'x1', '2'], engine, 'freeze') == []

    def test_freeze_then_free_asks_for_exactly_one(self, engine: SimpliPyEngine) -> None:
        assert typed_thaw_subsets(['pow', 'x1', '/', '3', '2'], engine, 'freeze_then_free') == [(3, 4)]

    def test_combinations_enumerates_every_non_empty_subset(self, engine: SimpliPyEngine) -> None:
        # deferred policy: 2**k - 1 duplicates, which is why it is not run
        assert sorted(typed_thaw_subsets(['pow', 'x1', '/', '3', '2'], engine, 'combinations')) == \
            [(3,), (3, 4), (4,)]

    def test_no_typed_literal_means_no_duplicate(self, engine: SimpliPyEngine) -> None:
        assert typed_thaw_subsets(['*', '2.5', 'sin', 'x1'], engine, 'combinations') == []


class TestWorker:
    """End to end through the refine worker: what the freeze buys, and what the duplicate adds."""

    @staticmethod
    def _payload(engine, expression, X, y, *, typed_spans, typed_frozen, p0=None):
        from flash_ansr.scoring import resolve_ranking
        return {
            "simplipy_engine": engine, "X": X, "y": y.reshape(-1, 1), "n_variables": N_VARIABLES,
            "expression": list(expression), "raw_beam": [7, 8, 9], "beam": [7, 8, 9],
            "raw_beam_decoded": " ".join(expression), "log_prob": -1.0,
            "constant_count": sum(t == "<constant>" for t in expression), "p0": p0,
            "n_restarts": 4, "method": "curve_fit_lm", "p0_noise": "normal", "p0_noise_kwargs": None,
            "refine_scope": "fittable", "constant_ladder": None, "converge_error": "ignore",
            "numpy_errors": "ignore", "y_variance": float(np.var(y)),
            "ranking_weights": resolve_ranking("mdl", mdl_strength=1e-2).effective_weights,
            "complexity": None, "seed": 0,
            "typed_spans": typed_spans, "typed_frozen": typed_frozen,
        }

    @staticmethod
    def _square_data(shift=0.7, n=96, seed=0, lo=-5.0, hi=5.0):
        """y = (x - shift)**2. On a domain STRADDLING the shift the base changes sign, which is
        where a free exponent is fatal: the first step off the even integer makes
        `negative ** non-integer` nan on every point below it."""
        from flash_ansr.utils import pad_input_set
        rng = np.random.default_rng(seed)
        X = pad_input_set(rng.uniform(lo, hi, size=(n, 1)), N_VARIABLES)
        return X, (X[:, 0] - shift) ** 2

    def test_a_frozen_exponent_is_not_a_fitted_constant(self, engine: SimpliPyEngine) -> None:
        X, y = self._square_data()
        result, warning = harness._refine_candidate_worker(self._payload(
            engine, ["pow", "-", "x1", "<constant>", "2"], X, y,
            typed_spans="freeze", typed_frozen=1))
        assert warning is None and result is not None
        assert len(result["fits"][0][0]) == 1               # ONE fitted constant: the shift
        assert result["fvu"] < 1e-30                        # and it finds it exactly
        assert np.asarray(result["fits"][0][0]).ravel() == pytest.approx([0.7], abs=1e-9)
        assert result["thawed"] is None                     # 'freeze' asks for no duplicate

    def test_thawing_loses_the_candidate_when_the_base_changes_sign(self, engine: SimpliPyEngine) -> None:
        """Why the freeze is not a ranking tweak: it decides whether the candidate EXISTS."""
        X, y = self._square_data()
        result, _warning = harness._refine_candidate_worker(self._payload(
            engine, ["pow", "-", "x1", "<constant>", "<constant>"], X, y,
            typed_spans="refine", typed_frozen=0))
        assert result is None                               # every restart failed
        # ... and a nearly correct seed does not rescue it either
        result, _warning = harness._refine_candidate_worker(self._payload(
            engine, ["pow", "-", "x1", "<constant>", "<constant>"], X, y,
            typed_spans="refine", typed_frozen=0, p0=[0.5, 2.0]))
        assert result is None

    def test_freeze_then_free_returns_a_duplicate_seeded_at_the_prediction(self, engine: SimpliPyEngine) -> None:
        X, y = self._square_data()
        result, warning = harness._refine_candidate_worker(self._payload(
            engine, ["pow", "-", "x1", "<constant>", "2"], X, y,
            typed_spans="freeze_then_free", typed_frozen=1, p0=[0.7]))
        assert warning is None and result is not None
        variants = result["thawed"]
        assert variants is not None and len(variants) == 1
        child = variants[0]
        assert child["typed_thaw"] == "4"                   # the exponent's token index
        assert child["typed_frozen"] == 0                   # the duplicate is on the shipped path
        assert child["expression"] == ["pow", "-", "x1", "<constant>", "<constant>"]
        assert len(child["fits"][0][0]) == 2                # shift AND exponent are fitted now
        assert child["fvu"] < 1e-20
        assert child["constants_emitted"] == pytest.approx([0.7, 2.0])

    def test_a_float_exponent_is_where_the_duplicate_earns_its_keep(self, engine: SimpliPyEngine) -> None:
        """A power law whose exponent is genuinely fractional: the domain is positive, so the free
        exponent is safe, and the duplicate can move it off a wrong prediction."""
        from flash_ansr.utils import pad_input_set
        rng = np.random.default_rng(0)
        X = pad_input_set(rng.uniform(0.5, 5.0, size=(96, 1)), N_VARIABLES)
        y = 2.0 * X[:, 0] ** 1.5
        # the model predicted the exponent 2 (wrong; the law is 1.5)
        result, _warning = harness._refine_candidate_worker(self._payload(
            engine, ["*", "<constant>", "pow", "x1", "2"], X, y,
            typed_spans="freeze_then_free", typed_frozen=1))
        assert result is not None
        assert result["fvu"] > 1e-3                          # frozen at the wrong exponent: a bad fit
        child = (result["thawed"] or [None])[0]
        assert child is not None
        assert child["fvu"] < result["fvu"]                  # thawed, it can reach the real exponent
        assert child["fvu"] < 1e-6

    def test_the_ladder_child_of_a_thawed_duplicate_keeps_the_mark(self, engine: SimpliPyEngine) -> None:
        """`typed_thaw` travels in the PAYLOAD, so the constant ladder's clone of the duplicate carries it.
        Set on the result after the fit, the ladder child was already cloned without it and read as a
        plain candidate's child -- which mis-attributed 26 of 60 rank-0 answers on 2026-09-12."""
        from flash_ansr.utils import pad_input_set
        from flash_ansr.spelling import ConstantLadderConfig
        rng = np.random.default_rng(0)
        X = pad_input_set(rng.uniform(0.5, 5.0, size=(96, 1)), N_VARIABLES)
        y = 2.0 * X[:, 0] ** 1.5
        payload = self._payload(engine, ["*", "<constant>", "pow", "x1", "2"], X, y,
                                typed_spans="freeze_then_free", typed_frozen=1)
        payload["constant_ladder"] = ConstantLadderConfig.from_mapping(True)
        result, _warning = harness._refine_candidate_worker(payload)
        assert result is not None
        child = (result["thawed"] or [None])[0]
        assert child is not None and child["typed_thaw"] == "4"
        grandchild = child["respelled"]                      # the ladder respells 1.5000... -> 3/2, 2.000... -> 2
        assert grandchild is not None, "the ladder produced no variant for the thawed duplicate"
        assert grandchild["typed_thaw"] == "4" and grandchild["spelling"]

    def test_a_candidate_with_no_typed_literal_spawns_no_duplicate(self, engine: SimpliPyEngine) -> None:
        from flash_ansr.utils import pad_input_set
        rng = np.random.default_rng(0)
        X = pad_input_set(rng.uniform(0.5, 3.0, size=(64, 1)), N_VARIABLES)
        y = 2.0 * X[:, 0]
        result, _warning = harness._refine_candidate_worker(self._payload(
            engine, ["*", "<constant>", "x1"], X, y,
            typed_spans="freeze_then_free", typed_frozen=0, p0=[2.0]))
        assert result is not None and result["thawed"] is None
