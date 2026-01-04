from pathlib import Path
from typing import Any

import pytest

from flash_ansr.model import manage


def _install_get_path(tmp_path: Path):
    def fake_get_path(*segments: str, filename: str | None = None, create: bool = False) -> str:  # type: ignore[override]
        assert segments[0] == 'models'
        base = tmp_path / 'models'
        if create:
            base.mkdir(parents=True, exist_ok=True)
        if len(segments) > 1:
            target = base / segments[1]
        else:
            target = base
        if create:
            target.mkdir(parents=True, exist_ok=True)
        if filename:
            target = target / filename
        return str(target)

    return fake_get_path


def test_install_model_downloads_to_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any) -> None:
    download_args: dict[str, Any] = {}

    def fake_snapshot_download(*, repo_id: str, repo_type: str, local_dir: str) -> None:  # type: ignore[override]
        download_args['repo_id'] = repo_id
        download_args['repo_type'] = repo_type
        download_args['local_dir'] = local_dir
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(manage, 'snapshot_download', fake_snapshot_download)
    monkeypatch.setattr(manage, 'get_path', _install_get_path(tmp_path))

    manage.install_model('owner/model', verbose=True)

    captured = capsys.readouterr()
    assert 'Installing model owner/model' in captured.out
    assert download_args == {
        'repo_id': 'owner/model',
        'repo_type': 'model',
        'local_dir': str(tmp_path / 'models' / 'owner/model'),
    }
    assert Path(download_args['local_dir']).exists()


def test_remove_model_deletes_package_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any) -> None:
    package_dir = tmp_path / 'models' / 'foo'
    package_dir.mkdir(parents=True, exist_ok=True)

    def fake_get_path(*segments: str, **_: Any) -> str:  # type: ignore[override]
        assert segments[0] == 'models'
        return str(package_dir)

    monkeypatch.setattr(manage, 'get_path', fake_get_path)

    manage.remove_model('foo', verbose=True, force_remove=True)

    captured = capsys.readouterr()
    assert 'Removing' in captured.out
    assert not package_dir.exists()


def test_remove_model_raises_if_paths_conflict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package_dir = tmp_path / 'models' / 'foo'
    package_dir.mkdir(parents=True, exist_ok=True)
    external_dir = tmp_path / 'external'
    external_dir.mkdir()

    def fake_get_path(*segments: str, **_: Any) -> str:  # type: ignore[override]
        assert segments[0] == 'models'
        return str(package_dir)

    monkeypatch.setattr(manage, 'get_path', fake_get_path)

    with pytest.raises(ValueError):
        manage.remove_model(str(external_dir), verbose=True, force_remove=True)
