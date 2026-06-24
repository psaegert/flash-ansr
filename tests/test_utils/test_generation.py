import unittest

from flash_ansr.utils.generation import (
    _FULL_CAP_MIN_VRAM_GB,
    _SMALL_CARD_BATCH_CAP,
    _spill_over_budget,
    suggest_batch_size,
    suggest_batch_size_dims,
)


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


# Representative model sizes -> the MEASURED 4090 cap (the lookup in suggest_batch_size).
_MODEL_CAPS = [
    (1e9, 128),     # ~1B
    (1.2e8, 512),   # ~120M
    (2e7, 1024),    # ~20M
    (3e6, 2048),    # ~3M
]
# A card at/above the gate reports ~25.8 GiB via total_memory/1e9; below: 20 GB ~21.5, 16 GB ~17.2, 12 GB ~11.
_BIG_CARD_VRAM = 25.76
_SMALL_CARD_VRAMS = [21.5, 17.18, 11.0]


class TestSuggestBatchSize(unittest.TestCase):
    def test_big_card_uses_measured_caps(self) -> None:
        # >= 24 GiB: with a large `choices`, the chunk equals the measured per-model cap.
        for n_params, cap in _MODEL_CAPS:
            got = suggest_batch_size(32768, n_params, _BIG_CARD_VRAM)
            self.assertEqual(got, cap, msg=f"n_params={n_params:g} on a 24+ GiB card")
            self.assertTrue(_is_power_of_two(got))

    def test_small_card_falls_to_conservative_default(self) -> None:
        # < 24 GiB: every model size collapses to the conservative cap regardless of the measured lookup.
        for vram in _SMALL_CARD_VRAMS:
            for n_params, _ in _MODEL_CAPS:
                got = suggest_batch_size(32768, n_params, vram)
                self.assertEqual(got, _SMALL_CARD_BATCH_CAP, msg=f"vram={vram} n_params={n_params:g}")
                self.assertTrue(_is_power_of_two(got))

    def test_gate_boundary_is_inclusive(self) -> None:
        # Exactly at the threshold uses the full cap; just below falls to the default.
        self.assertEqual(suggest_batch_size(32768, 1e9, _FULL_CAP_MIN_VRAM_GB), 128)
        self.assertEqual(suggest_batch_size(32768, 1e9, _FULL_CAP_MIN_VRAM_GB - 0.01), _SMALL_CARD_BATCH_CAP)

    def test_choices_clamps_below_cap(self) -> None:
        # `choices` smaller than the cap -> the largest power-of-2 <= choices, on any card.
        self.assertEqual(suggest_batch_size(100, 3e6, _BIG_CARD_VRAM), 64)   # min(2048, 100) -> pow2 64
        self.assertEqual(suggest_batch_size(100, 1e9, 17.18), 64)            # min(64(default), 100) -> 64
        self.assertEqual(suggest_batch_size(40, 1e9, 17.18), 32)             # min(64, 40) -> pow2 32
        self.assertEqual(suggest_batch_size(1, 1e9, _BIG_CARD_VRAM), 1)

    def test_small_card_never_exceeds_big_card(self) -> None:
        # The gate must be conservative: a smaller card never returns a LARGER chunk than a big card.
        for n_params, _ in _MODEL_CAPS:
            big = suggest_batch_size(32768, n_params, _BIG_CARD_VRAM)
            for vram in _SMALL_CARD_VRAMS:
                self.assertLessEqual(suggest_batch_size(32768, n_params, vram), big)


class TestSpillOverBudget(unittest.TestCase):
    def test_under_budget_is_false(self) -> None:
        self.assertFalse(_spill_over_budget(10 * 1024**3, 100 * 1024**3, 0.9))

    def test_over_budget_is_true(self) -> None:
        self.assertTrue(_spill_over_budget(95 * 1024**3, 100 * 1024**3, 0.9))

    def test_exact_fraction_is_not_over(self) -> None:
        # strict '>' -> sitting exactly at fraction*avail does not trip.
        self.assertFalse(_spill_over_budget(90, 100, 0.9))

    def test_nonpositive_avail_never_blocks(self) -> None:
        # A degenerate / unknown VRAM reading must not block a decode.
        self.assertFalse(_spill_over_budget(10**12, 0, 0.9))
        self.assertFalse(_spill_over_budget(10**12, -5.0, 0.9))


class TestSuggestBatchSizeDims(unittest.TestCase):
    _DIMS = dict(n_layers=8, n_heads=8, head_dim=64, max_len=64)

    def test_unknown_vram_uses_floor(self) -> None:
        # free_bytes None (CPU / unknown card) -> a conservative power-of-2 floor (<= 128), never extrapolate.
        got = suggest_batch_size_dims(32768, static_decode=True, free_bytes=None, **self._DIMS)
        self.assertLessEqual(got, 128)
        self.assertTrue(_is_power_of_two(got))

    def test_tiny_budget_returns_one(self) -> None:
        got = suggest_batch_size_dims(32768, static_decode=True, free_bytes=1000, **self._DIMS)
        self.assertEqual(got, 1)

    def test_generous_budget_is_power_of_two_within_choices(self) -> None:
        got = suggest_batch_size_dims(4096, static_decode=True, free_bytes=24 * 1024**3, **self._DIMS)
        self.assertTrue(_is_power_of_two(got))
        self.assertLessEqual(got, 4096)
        self.assertGreaterEqual(got, 1)

    def test_static_cap_at_least_dynamic(self) -> None:
        # The dynamic per-row overhead (4.2) is larger than static (1.6), so for the SAME budget the
        # dynamic chunk is never larger than the static one.
        free = 24 * 1024**3
        static = suggest_batch_size_dims(32768, static_decode=True, free_bytes=free, **self._DIMS)
        dynamic = suggest_batch_size_dims(32768, static_decode=False, free_bytes=free, **self._DIMS)
        self.assertLessEqual(dynamic, static)


if __name__ == "__main__":
    unittest.main()
