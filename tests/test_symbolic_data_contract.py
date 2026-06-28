"""Contract test for the flash-ansr <-> symbolic-data seam (flash-ansr 0.7).

flash-ansr 0.7 carved the expression/data layer into the ``symbolic-data`` package. This
freezes the seam: flash-ansr re-exports the symbols it needs from ``symbolic_data``, the
deprecated ``flash_ansr.expressions.*`` paths still resolve (emitting a DeprecationWarning),
expression-token normalization now lives in ``simplipy``, and ``FlashANSRDataset`` keeps
its ``skeleton_pool`` constructor seam.
"""
import importlib
import inspect
import sys
import warnings

import pytest


def test_top_level_reexports_symbolic_data():
    import flash_ansr
    import symbolic_data

    assert flash_ansr.SkeletonPool is symbolic_data.SkeletonPool
    assert flash_ansr.NoValidSampleFoundError is symbolic_data.NoValidSampleFoundError


@pytest.mark.parametrize(
    "mod,name",
    [
        ("skeleton_pool", "SkeletonPool"),
        ("skeleton_sampling", "SkeletonSampler"),
        ("support_sampling", "SupportSampler"),
        ("distributions", "get_distribution"),
        ("prior_factory", "build_prior_callable"),
        ("holdout", "HoldoutManager"),
        ("token_ops", "apply_variable_mapping"),
        ("compilation", "codify"),
    ],
)
def test_deprecated_expressions_shims_resolve(mod, name):
    """Old deep imports still resolve to the symbolic_data object (back-compat)."""
    shim = importlib.import_module(f"flash_ansr.expressions.{mod}")
    target = importlib.import_module(f"symbolic_data.{mod}")
    assert getattr(shim, name) is getattr(target, name)


def test_expressions_shim_emits_deprecation_warning():
    """Importing a carved submodule fresh emits a DeprecationWarning."""
    sys.modules.pop("flash_ansr.expressions.skeleton_pool", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("flash_ansr.expressions.skeleton_pool")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_normalization_moved_to_simplipy():
    sys.modules.pop("flash_ansr.expressions.normalization", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from flash_ansr.expressions.normalization import normalize_skeleton, normalize_expression
    import simplipy

    assert normalize_skeleton is simplipy.normalize_skeleton
    assert normalize_expression is simplipy.normalize_expression
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_flash_ansr_dataset_keeps_skeleton_pool_seam():
    from flash_ansr.data import FlashANSRDataset

    assert "skeleton_pool" in inspect.signature(FlashANSRDataset.__init__).parameters


def test_data_cli_dropped_model_cli_kept(capsys):
    """The skeleton-pool data CLI moved to symbolic-data; model commands remain."""
    from flash_ansr.__main__ import main

    with pytest.raises(SystemExit):
        main(["generate-skeleton-pool"])
    err = capsys.readouterr().err
    assert "invalid choice" in err
    for kept in ("train", "benchmark", "install", "remove"):
        assert kept in err
