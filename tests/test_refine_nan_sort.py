"""A non-finite restart loss must never win the fit selection.

`sorted(key=loss)` is a partial no-op on a list containing NaN: NaN compares False against
everything, so a leading NaN stays at index 0 AND the finite entries behind it are left
unordered. Every consumer downstream reads `_all_constants_values[0]`, so a divergent restart
was published as the answer while an exact fit sat in the same list.
"""
import numpy as np

from flash_ansr.refine import Refiner, fit_sort_key


def _fit(value: float, loss: float) -> tuple[np.ndarray, np.ndarray, float]:
    return (np.array([value], dtype=float), np.array([[0.0]], dtype=float), loss)


class TestFitSortKey:
    def test_nan_sinks_below_every_finite_loss(self) -> None:
        fits = [_fit(1.0, np.nan), _fit(2.0, 3.0), _fit(3.0, np.nan), _fit(4.0, 1e-9), _fit(5.0, 0.5)]
        ordered = sorted(fits, key=fit_sort_key)
        assert [f[2] for f in ordered[:3]] == [1e-9, 0.5, 3.0]
        assert all(np.isnan(f[2]) for f in ordered[3:])

    def test_infinite_loss_also_sinks(self) -> None:
        ordered = sorted([_fit(1.0, np.inf), _fit(2.0, 7.0)], key=fit_sort_key)
        assert ordered[0][2] == 7.0

    def test_min_selects_the_finite_optimum(self) -> None:
        # The failure shape at inference.py:_best_constants and _compile_results_pure.
        fits = [_fit(1.0, np.nan), _fit(2.0, 0.0), _fit(3.0, 5.0)]
        assert min(fits, key=fit_sort_key)[0].item() == 2.0

    def test_all_non_finite_is_ordered_not_crashed(self) -> None:
        ordered = sorted([_fit(1.0, np.nan), _fit(2.0, np.inf)], key=fit_sort_key)
        assert len(ordered) == 2


class TestAssignFits:
    def _refiner(self) -> Refiner:
        r = Refiner.__new__(Refiner)
        r.constants_symbols = ['c0']
        r._all_constants_values = []
        r.loss = np.inf
        r.valid_fit = False
        return r

    def test_leading_nan_does_not_become_the_best_fit(self) -> None:
        r = self._refiner()
        r._assign_fits([_fit(1.0, np.nan), _fit(2.0, 0.0), _fit(3.0, np.nan)])
        assert r.loss == 0.0
        assert r.valid_fit is True
        assert r._all_constants_values[0][0].item() == 2.0

    def test_valid_fit_describes_index_zero(self) -> None:
        # Previously `valid_fit = any(isfinite(...))` could be True while index 0 held NaN,
        # so `if not refiner.valid_fit` passed and the caller then read a NaN loss.
        r = self._refiner()
        r._assign_fits([_fit(1.0, np.nan), _fit(2.0, 4.0)])
        assert r.valid_fit is True
        assert np.isfinite(r._all_constants_values[0][-1])

    def test_all_non_finite_is_not_a_valid_fit(self) -> None:
        r = self._refiner()
        r._assign_fits([_fit(1.0, np.nan), _fit(2.0, np.inf)])
        assert r.valid_fit is False

    def test_empty_input_leaves_infinite_loss(self) -> None:
        r = self._refiner()
        r._assign_fits([])
        assert r.loss == np.inf
        assert r.valid_fit is False

    def test_arity_mismatch_still_filtered(self) -> None:
        r = self._refiner()
        r._assign_fits([(np.array([1.0, 2.0]), None, 0.0), _fit(3.0, 9.0)])
        assert len(r._all_constants_values) == 1
        assert r.loss == 9.0
