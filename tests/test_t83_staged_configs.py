"""The T8.3 staged runs (owner 2026-11-13) are T8.1-120M plus exactly the changes of their stage.

Three cumulative 120M runs on SimpliPy 0.14 test T8.3's changes before the 0.15 run: S1 the training mechanics
(gated decoder self-attention, margin cross-entropy), S2 adds the units augmentation and real-valued unused input
columns, S3 the family-coherence prior and the complexity spread prompt. Each run is compared with the one before it
(S1 with T8-120M), so a config must not change anything else. Read as YAML only.
"""
import pytest
import yaml

from flash_ansr import get_path
from flash_ansr.model.flash_ansr_model import FRESH_RUN_SETTINGS
from flash_ansr.train.train import FRESH_RUN_TRAIN_SETTINGS

FILES = ["model.yaml", "train.yaml", "dataset_train.yaml", "dataset_val.yaml", "catalog_train.yaml", "tokenizer.yaml"]

# What each pool's bare name resolved to on 2026-11-13 (HF snapshot b798ad3d): the versions T8 and T8.1 held out.
PINNED_VERSIONS = {"fastsrb": 2}


def _load(config: str, name: str) -> dict:
    with open(get_path("configs", config, name)) as f:
        return yaml.safe_load(f)


def _without(mapping: dict, *keys: str) -> dict:
    return {key: value for key, value in mapping.items() if key not in keys}


@pytest.mark.parametrize("name", FILES)
def test_stage_1_is_t81_plus_its_two_changes(name: str):
    base, stage = _load("v25.0-T8.1-120M", name), _load("v25.0-T8.3-S1-120M", name)
    if name == "model.yaml":
        assert stage["decoder_self_attn_gate"] is True and stage["decoder_cross_attn_gate"] is False
        assert _without(stage, "decoder_self_attn_gate", "decoder_cross_attn_gate") == base
    elif name == "train.yaml":
        assert stage["cross_entropy"] == "margin" and "cross_entropy" not in base
        assert _without(stage, "cross_entropy") == base
    elif name == "catalog_train.yaml":
        pinned = [f"{pool}@{PINNED_VERSIONS.get(pool, 1)}" for pool in base["holdout_pools"]]
        assert stage["holdout_pools"] == pinned
        assert _without(stage, "holdout_pools") == _without(base, "holdout_pools")
    else:
        assert stage == base


def test_stage_1_is_a_valid_fresh_run_on_simplipy_014():
    model, train = _load("v25.0-T8.3-S1-120M", "model.yaml"), _load("v25.0-T8.3-S1-120M", "train.yaml")
    for key, value in FRESH_RUN_SETTINGS.items():
        assert model[key] == value, key
    for key, value in FRESH_RUN_TRAIN_SETTINGS.items():
        assert train[key] == value, key
    assert "legacy_train_settings" not in train and "legacy_model_settings" not in train
    assert model["simplipy_engine"] == _load("v25.0-T8.3-S1-120M", "catalog_train.yaml")["simplipy_engine"] == "acj-5-4-llm"
    # T8-120M's schedule, so its checkpoints line up step for step
    with open(get_path("configs", "v25.0-T8-120M", "train.yaml")) as f:
        t8 = yaml.safe_load(f)
    for key in ("steps", "batch_size", "lr_schedule", "optimizer", "z_loss_weight"):
        assert train.get(key) == t8.get(key), key
