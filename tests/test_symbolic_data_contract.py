"""Contract test for the flash-ansr <-> symbolic-data seam.

flash-ansr carved the expression/data layer into ``symbolic-data`` (0.7) and removed the
deprecated ``flash_ansr.expressions`` shim (0.8). This freezes the remaining seam: flash-ansr
re-exports the few symbols it needs from ``symbolic_data``, expression-token normalization lives
in ``simplipy``, and ``FlashANSRDataset`` keeps its ``skeleton_pool`` constructor seam.
"""
import inspect

import pytest


def test_top_level_reexports_symbolic_data():
    import flash_ansr
    import symbolic_data

    assert flash_ansr.SkeletonPool is symbolic_data.SkeletonPool
    assert flash_ansr.NoValidSampleFoundError is symbolic_data.NoValidSampleFoundError


def test_expressions_shim_is_gone():
    """The deprecated shim package was removed in 0.8 (import the data layer from symbolic_data)."""
    with pytest.raises(ModuleNotFoundError):
        import flash_ansr.expressions  # noqa: F401


def test_normalization_lives_in_simplipy():
    import simplipy

    assert callable(simplipy.normalize_skeleton)
    assert callable(simplipy.normalize_expression)


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
