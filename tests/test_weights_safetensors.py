"""Weights are safetensors. One format, one code path, and the numbers survive the round trip."""
import os

import pytest
import torch

from flash_ansr.utils.weights import (
    LEGACY_WEIGHTS_FILENAME,
    WEIGHTS_FILENAME,
    convert_legacy_weights,
    load_weights,
    save_weights,
)


def _module():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.LayerNorm(5), torch.nn.Linear(5, 2))


class TestRoundTrip:
    def test_values_survive_exactly(self, tmp_path):
        src = _module()
        save_weights(src, str(tmp_path))
        dst = _module()
        with torch.no_grad():                       # perturb so a no-op load would be caught
            for p in dst.parameters():
                p.add_(1.0)
        load_weights(dst, str(tmp_path))
        for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
            assert torch.equal(a, b)

    def test_writes_the_expected_filename(self, tmp_path):
        save_weights(_module(), str(tmp_path))
        assert os.path.exists(tmp_path / WEIGHTS_FILENAME)
        assert not os.path.exists(tmp_path / LEGACY_WEIGHTS_FILENAME)

    def test_key_mismatch_is_refused(self, tmp_path):
        save_weights(_module(), str(tmp_path))
        with pytest.raises(RuntimeError):
            load_weights(torch.nn.Linear(6, 5), str(tmp_path))


class TestRefusals:
    def test_missing_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=WEIGHTS_FILENAME):
            load_weights(_module(), str(tmp_path))

    def test_a_legacy_checkpoint_is_named_in_the_error(self, tmp_path):
        # Pickles are not read any more; the error has to point at the way out.
        torch.save(_module().state_dict(), tmp_path / LEGACY_WEIGHTS_FILENAME)
        with pytest.raises(FileNotFoundError, match="convert-weights"):
            load_weights(_module(), str(tmp_path))


class TestConverter:
    def test_converts_and_leaves_the_pickle_alone(self, tmp_path):
        src = _module()
        torch.save(src.state_dict(), tmp_path / LEGACY_WEIGHTS_FILENAME)
        convert_legacy_weights(str(tmp_path))
        assert os.path.exists(tmp_path / LEGACY_WEIGHTS_FILENAME), "the .pt must be left in place"
        dst = _module()
        load_weights(dst, str(tmp_path))
        for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
            assert torch.equal(a, b)

    def test_refuses_to_clobber_without_overwrite(self, tmp_path):
        torch.save(_module().state_dict(), tmp_path / LEGACY_WEIGHTS_FILENAME)
        convert_legacy_weights(str(tmp_path))
        with pytest.raises(FileExistsError):
            convert_legacy_weights(str(tmp_path))
        convert_legacy_weights(str(tmp_path), overwrite=True)

    def test_non_tensor_entries_are_refused_not_dropped(self, tmp_path):
        torch.save({"w": torch.zeros(2), "step": 7}, tmp_path / LEGACY_WEIGHTS_FILENAME)
        with pytest.raises(ValueError, match="non-tensor"):
            convert_legacy_weights(str(tmp_path))

    def test_nothing_to_convert_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=LEGACY_WEIGHTS_FILENAME):
            convert_legacy_weights(str(tmp_path))
