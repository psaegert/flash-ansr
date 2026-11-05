"""The v25.0-T8.3 configs state the owner's T8.3 rulings (2026-11-03) and stay internally consistent.

Read as YAML only: the configs name SimpliPy 0.15's engine, which this environment may not have, so nothing here
builds a model or a catalog. The end-to-end smoke ran on the 0.14 engine with the new operators removed.
"""
import pytest
import yaml

from flash_ansr import get_path
from flash_ansr.data.units import validate_units_block
from flash_ansr.model.flash_ansr_model import FRESH_RUN_SETTINGS
from flash_ansr.train.train import FRESH_RUN_TRAIN_SETTINGS

NEW_OPERATORS = {"floor", "ceil", "erf", "erfc", "harmonic", "max", "min", "mod", "atan2", "besselj", "bessely",
                 "besseli", "besselk"}


def _load(name: str, size: str = "120M") -> dict:
    with open(get_path("configs", f"v25.0-T8.3-{size}", name)) as f:
        return yaml.safe_load(f)


def test_the_model_gates_self_attention_only_and_carries_every_fix():
    model = _load("model.yaml")
    assert model["decoder_self_attn_gate"] is True and model["decoder_cross_attn_gate"] is False
    for key, value in FRESH_RUN_SETTINGS.items():
        assert model[key] == value, key
    assert model["simplipy_engine"] == "acj2-5-4-llm"


def test_training_uses_the_margin_cross_entropy():
    train = _load("train.yaml")
    for key, value in FRESH_RUN_TRAIN_SETTINGS.items():
        assert train[key] == value, key
    assert "legacy_train_settings" not in train and "legacy_model_settings" not in train


def test_the_training_stream_carries_arm_uh():
    block = validate_units_block(_load("dataset_train.yaml")["units_block"])
    assert block == {"p_identity": 0.5, "log2_scale_max": 100, "log2_literal_max": 100, "max_redraws": 400}
    assert "units_block" not in _load("dataset_val.yaml")


def test_the_catalog_has_the_new_operators_and_the_coherence_prior():
    catalog = _load("catalog_train.yaml")
    weights = catalog["operator_weights"]
    assert NEW_OPERATORS <= set(weights) and all(weights[op] == 1 for op in NEW_OPERATORS)
    affinity = catalog["family_affinity"]
    assert affinity["concentration"] == 0.35
    placed = [op for ops in affinity["families"].values() for op in ops] + list(affinity["independent"])
    assert len(placed) == len(set(placed))
    # The function nodes of the depth prior: the unary operators and the binary ones with a literal slot, except the
    # transparent neg/inv. Each must sit in exactly one family or be independent (symbolic-data refuses otherwise).
    binary = {"+", "-", "*", "/", "pow", "rootn", "max", "min", "mod", "atan2", "besselj", "bessely", "besseli",
              "besselk"}
    slotted = set(catalog["typed_slots"])
    functions = {op for op, w in weights.items() if w > 0 and (op not in binary or op in slotted)} - {"neg", "inv"}
    assert set(placed) == functions
    assert catalog["simplipy_engine"] == "acj2-5-4-llm"


def test_every_catalog_operator_is_in_the_vocabulary():
    operators = set(_load("tokenizer.yaml")["operators"])
    assert set(_load("catalog_train.yaml")["operator_weights"]) <= operators


def test_the_holdouts_pin_srbfs_next_release_and_keep_t81s():
    pools = _load("catalog_train.yaml")["holdout_pools"]
    with open(get_path("configs", "v25.0-T8.1-120M", "catalog_train.yaml")) as f:
        previous = yaml.safe_load(f)["holdout_pools"]
    assert set(previous) <= {pool.split("@")[0] for pool in pools}
    # the release's 158 sets plus the five it retires, each at an exact version
    assert len(pools) == len(set(pools)) == 158 + 5
    assert all(pool.count("@") == 1 and pool.split("@")[1].isdigit() for pool in pools)


@pytest.mark.parametrize("name", ["model.yaml", "train.yaml", "dataset_train.yaml", "dataset_val.yaml",
                                  "catalog_train.yaml", "tokenizer.yaml"])
def test_every_file_parses(name: str):
    assert isinstance(_load(name), dict)
