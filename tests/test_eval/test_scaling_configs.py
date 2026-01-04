from pathlib import Path

import pytest

import yaml

SCALING_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "scaling"


@pytest.mark.parametrize("config_path", sorted(SCALING_CONFIG_DIR.glob("*.yaml")))
def test_scaling_configs_define_required_sections(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    experiments = config.get("experiments")
    entries = experiments.values() if experiments else [config]

    assert entries, f"config {config_path} has no run entries"

    for entry in entries:
        run = entry.get("run", entry)
        assert "data_source" in run, f"Missing data_source in {config_path}"
        assert "model_adapter" in run, f"Missing model_adapter in {config_path}"
        assert run["data_source"].get("type"), f"data_source.type missing in {config_path}"
        assert run["model_adapter"].get("type"), f"model_adapter.type missing in {config_path}"
