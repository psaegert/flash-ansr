"""A fresh training run carries every fix whose default is legacy (FRESH_RUN_SETTINGS).

The fixes default to the legacy behaviour so that old checkpoints load bit-identically; the trainer refuses to
start a NEW model without them. The v25.0-T8 series was trained without two of them because nothing checked,
so the T8 configs are the case that must fail here, and the T8.1 configs the case that must pass.
"""
import ast
import inspect
import textwrap
from unittest import mock

import pytest
import torch
import yaml

from flash_ansr import FlashANSRModel, get_path
from flash_ansr.model.flash_ansr_model import FRESH_RUN_SETTINGS
from flash_ansr.utils.config_io import load_config

# Keys from_config reads with a default that is a real default, not a setting kept for old checkpoints.
PLAIN_DEFAULTS = {
    "decoder_data_mode", "decoder_block_cross_attn_norm", "decoder_cross_attn_kv_norm", "decoder_use_rope_cross_attn",
    "decoder_use_xsa_self_attn", "decoder_block_norm_position", "encoder_block_norm_position", "optional_condition",
    "null_memory_init_seed", "outlier_head", "head_pre_logits_norm", "decoder_data_norm",
}
T8_1_SIZES = ("3M", "20M", "120M")


def _defaulted_keys() -> set[str]:
    """Every key FlashANSRModel.from_config reads as ``config_.get(key, default)``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(FlashANSRModel.from_config)))
    return {
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "config_"
        and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
    }


def test_every_defaulted_model_key_is_classified():
    # A new key read with a default is a decision: a fix whose default only serves old checkpoints goes into
    # FRESH_RUN_SETTINGS (and the trainer enforces it), anything else into PLAIN_DEFAULTS above.
    keys = _defaulted_keys()
    assert keys, "the parse found no defaulted keys: the check would pass vacuously"
    unclassified = keys - set(FRESH_RUN_SETTINGS) - PLAIN_DEFAULTS
    assert not unclassified, f"classify these from_config defaults: {sorted(unclassified)}"
    assert set(FRESH_RUN_SETTINGS) <= keys, "every registered fix is read by from_config"


def _model(config_dir: str, engine: str = "acj-4-3") -> FlashANSRModel:
    # The engine is irrelevant to what is checked here; the suite's small one keeps the test fast.
    config = load_config(get_path("configs", config_dir, "model.yaml"))
    config["simplipy_engine"] = engine
    return FlashANSRModel.from_config(config).eval()


def _padding_change(model: FlashANSRModel, n: int = 16, rows: int = 1024) -> float:
    """Relative change of the encoder memory when an n-point set is zero-padded to ``rows`` behind a row mask,
    as every training batch pads it, against the same set unpadded, as inference passes it."""
    d = model.encoder_max_n_variables
    x = torch.randn(1, n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    padded = torch.zeros(1, rows, d, dtype=torch.float64)
    padded[:, :n] = x
    mask = torch.zeros(1, rows, dtype=torch.bool)
    mask[:, :n] = True
    with torch.no_grad():
        tight = model._create_memory(x)
        wide = model._create_memory(padded, mask)
    return float((tight - wide).norm() / tight.norm())


def test_t8_is_refused_and_depends_on_padding():
    model = _model("v25.0-T8-3M")
    assert set(model.fresh_run_deviations()) == {"encoder_mask_query_norms", "sanitize_input_num"}
    assert _padding_change(model) > 1e-2


@pytest.mark.parametrize("size", T8_1_SIZES)
def test_t8_1_configs_state_every_fix(size):
    config = yaml.safe_load(open(get_path("configs", f"v25.0-T8.1-{size}", "model.yaml")))
    # Stated in the file, not inherited from a default that happens to agree today.
    assert {key: config.get(key) for key in FRESH_RUN_SETTINGS} == FRESH_RUN_SETTINGS


def test_t8_1_carries_every_fix_and_ignores_padding():
    model = _model("v25.0-T8.1-3M")
    assert model.fresh_run_deviations() == {}
    assert _padding_change(model) < 1e-6


def _trainer(tmp_path, pop=(), legacy=None):
    import shutil
    from flash_ansr.train import Trainer

    cfg_dir = tmp_path / "cfg"
    shutil.copytree(get_path("configs", "test"), cfg_dir)
    model_yaml = cfg_dir / "model.yaml"
    model_cfg = yaml.safe_load(model_yaml.read_text())
    for key in pop:
        model_cfg.pop(key)
    model_yaml.write_text(yaml.safe_dump(model_cfg))
    if legacy is not None:
        train_yaml = cfg_dir / "train.yaml"
        train_cfg = yaml.safe_load(train_yaml.read_text())
        train_cfg["legacy_model_settings"] = legacy
        train_yaml.write_text(yaml.safe_dump(train_cfg))
    return Trainer.from_config(str(cfg_dir / "train.yaml"))


def _run(trainer, **kwargs):
    with mock.patch("wandb.init"), mock.patch("wandb.log"):
        return trainer.run(project_name="neural-symbolic-regression-test", entity="psaegert", name="pytest-fresh",
                           verbose=False, steps=1, device="cpu", preprocess=False, checkpoint_interval=None,
                           checkpoint_directory=None, wandb_mode="disabled", **kwargs)


def test_trainer_refuses_a_fresh_run_that_inherits_a_legacy_default(tmp_path):
    trainer = _trainer(tmp_path, pop=("encoder_mask_query_norms",))
    with pytest.raises(ValueError, match="encoder_mask_query_norms: True"):
        _run(trainer)


def test_a_legacy_setting_needs_a_reason(tmp_path):
    with pytest.raises(ValueError, match="needs a reason"):
        _trainer(tmp_path, pop=("sanitize_input_num",), legacy={"sanitize_input_num": ""})._check_fresh_run_settings()
    _trainer(tmp_path / "b", pop=("sanitize_input_num",),
             legacy={"sanitize_input_num": "control arm reproducing T8"})._check_fresh_run_settings()


def test_the_fixed_test_config_passes_the_check(tmp_path):
    _trainer(tmp_path)._check_fresh_run_settings()


def test_a_resumed_run_keeps_the_settings_it_started_with(tmp_path):
    # Resuming a legacy run (a T8 chain on a cluster, say) must not be refused: it continues the model it has.
    trainer = _trainer(tmp_path, pop=("encoder_mask_query_norms", "sanitize_input_num"))
    with mock.patch.object(type(trainer), "_load_checkpoint", return_value=1):
        _run(trainer, resume_from=str(tmp_path / "checkpoint_1"))
