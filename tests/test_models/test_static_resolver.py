"""CPU unit tests for the collapsed static-decode resolver (multi-chunk regime predicate, 2026-06-24).

The old per-(scale,c) enable table + scale-class/c-band + hardware gate were replaced by one runtime
predicate: with `static_decode=None` (the deployed default), static engages iff the model is capable AND
the global kill-switch is on AND the decode is MULTI-chunk (the resolved static batch < choices). These
tests drive `_resolve_static_decode` via a lightweight stub (no real model / GPU).
"""
import types
import unittest
from unittest import mock

from flash_ansr.model import flash_ansr_model as fam


def _stub(*, capable=True):
    s = types.SimpleNamespace()
    s.supports_static_decode = lambda: ((True, "") if capable else (False, "incapable config"))
    return s


def _resolve(stub, static_decode, choices=None, static_batch=None):
    return fam.FlashANSRModel._resolve_static_decode(
        stub, static_decode, choices=choices, static_batch=static_batch)


class TestShipsEnabled(unittest.TestCase):
    def test_kill_switch_default_on(self) -> None:
        # static-decode now ships ENABLED by default (multi-chunk regime); flip False to force dynamic.
        self.assertIs(fam._DEPLOYED_STATIC_ENABLED, True)


class TestExplicit(unittest.TestCase):
    def test_explicit_true_capable(self) -> None:
        self.assertTrue(_resolve(_stub(), True, choices=1024, static_batch=256))

    def test_explicit_true_incapable_warns_and_falls_back(self) -> None:
        with self.assertWarns(UserWarning):
            self.assertFalse(_resolve(_stub(capable=False), True, choices=1024, static_batch=256))

    def test_explicit_false(self) -> None:
        self.assertFalse(_resolve(_stub(), False, choices=1024, static_batch=64))


class TestRegimePredicate(unittest.TestCase):
    def test_multichunk_enables_static(self) -> None:
        # static_batch < choices -> multi-chunk -> static
        self.assertTrue(_resolve(_stub(), None, choices=1024, static_batch=256))
        self.assertTrue(_resolve(_stub(), None, choices=262144, static_batch=256))
        self.assertTrue(_resolve(_stub(), None, choices=512, static_batch=64))

    def test_singlechunk_stays_dynamic(self) -> None:
        # static_batch >= choices -> single-chunk -> dynamic (excludes the measured single-chunk losses)
        self.assertFalse(_resolve(_stub(), None, choices=64, static_batch=64))
        self.assertFalse(_resolve(_stub(), None, choices=512, static_batch=512))
        self.assertFalse(_resolve(_stub(), None, choices=64, static_batch=256))

    def test_unknown_batch_or_choices_is_dynamic(self) -> None:
        self.assertFalse(_resolve(_stub(), None, choices=1024, static_batch=None))  # CPU / direct caller
        self.assertFalse(_resolve(_stub(), None, choices=None, static_batch=256))

    def test_incapable_is_dynamic(self) -> None:
        self.assertFalse(_resolve(_stub(capable=False), None, choices=1024, static_batch=256))

    def test_kill_switch_off_forces_dynamic(self) -> None:
        with mock.patch.object(fam, "_DEPLOYED_STATIC_ENABLED", False):
            self.assertFalse(_resolve(_stub(), None, choices=1024, static_batch=256))


if __name__ == "__main__":
    unittest.main()
