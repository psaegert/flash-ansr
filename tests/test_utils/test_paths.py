"""Tests for the project-root path helpers, including the FLASH_ANSR_ROOT override.

The override is the pre-split hook that lets a separate repo importing ``flash_ansr`` anchor asset
lookups at its own tree. These tests pin both the override behaviour
and that the default (env unset) is unchanged for the 80+ existing get_path call sites.
"""
import os

import pytest

from flash_ansr.utils.paths import (
    ROOT_ENV_VAR,
    _DEFAULT_ROOT,
    get_path,
    get_root,
    normalize_path_preserve_leading_dot,
    substitute_root_path,
)


class TestGetRoot:
    def test_default_is_source_checkout_root(self, monkeypatch):
        monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
        root = get_root()
        assert os.path.isabs(root)
        assert root == _DEFAULT_ROOT
        # the default root is the repo root: it contains src/flash_ansr
        assert os.path.isdir(os.path.join(root, "src", "flash_ansr"))

    def test_env_override_takes_precedence(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        assert get_root() == os.path.abspath(str(tmp_path))

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(ROOT_ENV_VAR, "")
        assert get_root() == _DEFAULT_ROOT


class TestGetPath:
    def test_default_resolves_under_repo_root(self, monkeypatch):
        monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
        assert get_path("configs") == os.path.join(_DEFAULT_ROOT, "configs")
        assert get_path("a", "b", filename="c.txt") == os.path.join(_DEFAULT_ROOT, "a", "b", "c.txt")

    def test_honours_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        root = os.path.abspath(str(tmp_path))
        assert get_path("models", "x") == os.path.join(root, "models", "x")

    def test_rejects_non_string_args(self):
        with pytest.raises(TypeError):
            get_path("models", 3)  # type: ignore[arg-type]

    def test_create_makes_directories(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        directory = get_path("a", "b", create=True)
        assert os.path.isdir(directory)

    def test_create_with_filename_makes_parent_only(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        file_path = get_path("c", filename="f.txt", create=True)
        assert os.path.isdir(os.path.dirname(file_path))
        assert not os.path.exists(file_path)  # the file itself is not created


class TestSubstituteRootPath:
    def test_replaces_placeholder_with_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        root = os.path.abspath(str(tmp_path))
        assert substitute_root_path("{{ROOT}}/models/x") == f"{root}/models/x"

    def test_no_placeholder_is_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        assert substitute_root_path("no/placeholder/here") == "no/placeholder/here"


class TestNormalizePreserveLeadingDot:
    def test_preserves_leading_dot(self):
        assert normalize_path_preserve_leading_dot(f".{os.sep}a{os.sep}b") == os.path.join(".", "a", "b")

    def test_normalizes_without_leading_dot(self):
        assert normalize_path_preserve_leading_dot(f"a{os.sep}b{os.sep}") == os.path.join("a", "b")
