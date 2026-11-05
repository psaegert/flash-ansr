"""The margin form of the cross-entropy (Barron, Gnanapragasam & Tschannen 2026, arXiv:2610.31642).

Same value as PyTorch's fused cross-entropy, but the target logit keeps its gradient past the fp32 precision
cliff (margin ~16.6), where the fused form rounds softmax_t to 1 and drops it ("Softmax Collapse").
"""
from unittest import mock

import pytest
import torch
import yaml

from flash_ansr import get_path
from flash_ansr.train.train import FRESH_RUN_TRAIN_SETTINGS, Trainer, margin_cross_entropy

IGNORE = 0


def _logits_and_labels(seed=0, n=64, vocab=40, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, vocab, generator=g) * scale
    labels = torch.randint(1, vocab, (n,), generator=g)
    labels[::7] = IGNORE
    return logits, labels


def test_the_value_is_the_cross_entropy():
    logits, labels = _logits_and_labels()
    ours = margin_cross_entropy(logits, labels, IGNORE)
    fused = torch.nn.functional.cross_entropy(logits, labels, ignore_index=IGNORE, reduction="none")
    torch.testing.assert_close(ours, fused, rtol=1e-6, atol=1e-6)


def test_moderate_margins_give_the_fused_gradient():
    logits, labels = _logits_and_labels(seed=1)
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    margin_cross_entropy(a, labels, IGNORE).sum().backward()
    torch.nn.functional.cross_entropy(b, labels, ignore_index=IGNORE, reduction="sum").backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_the_target_keeps_its_gradient_past_the_fp32_cliff():
    # The target leads both rivals by 20 > 16.6: the fused form drops the target's gradient, ours keeps it.
    logits = torch.tensor([[20.0, 0.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0])
    torch.nn.functional.cross_entropy(logits, labels, reduction="sum").backward()
    assert logits.grad[0, 0].item() == 0.0 and logits.grad[0, 1].item() > 0.0    # collapsed, does not sum to 0
    logits.grad = None
    margin_cross_entropy(logits, labels, ignore_index=-100).sum().backward()
    expected = torch.exp(torch.tensor(-20.0))                                      # softmax_k for each rival
    torch.testing.assert_close(logits.grad[0, 1:], torch.full((2,), expected.item()), rtol=1e-5, atol=0.0)
    torch.testing.assert_close(logits.grad[0, 0], -2 * expected, rtol=1e-5, atol=0.0)
    assert abs(logits.grad.sum().item()) <= 1e-6 * logits.grad.abs().sum().item()


def test_the_value_at_a_large_margin_is_not_rounded_to_zero():
    loss = margin_cross_entropy(torch.tensor([[30.0, 0.0]]), torch.tensor([0]), ignore_index=-100)
    torch.testing.assert_close(loss, torch.exp(torch.tensor([-30.0])), rtol=1e-6, atol=0.0)
    assert torch.nn.functional.cross_entropy(torch.tensor([[30.0, 0.0]]), torch.tensor([0])).item() == 0.0


def test_ignored_positions_have_no_loss_and_no_gradient():
    logits, labels = _logits_and_labels(seed=2)
    logits.requires_grad_(True)
    loss = margin_cross_entropy(logits, labels, IGNORE)
    ignored = labels == IGNORE
    assert ignored.any() and (loss[ignored] == 0).all()
    loss.sum().backward()
    assert (logits.grad[ignored] == 0).all()


def test_half_precision_logits_are_computed_in_fp32():
    logits, labels = _logits_and_labels(seed=3)
    ours = margin_cross_entropy(logits.to(torch.bfloat16), labels, IGNORE)
    assert ours.dtype == torch.float32
    fused = torch.nn.functional.cross_entropy(logits.to(torch.bfloat16).float(), labels, ignore_index=IGNORE, reduction="none")
    torch.testing.assert_close(ours, fused, rtol=1e-6, atol=1e-6)


def _trainer(tmp_path, cross_entropy=None, legacy=None):
    import shutil
    cfg_dir = tmp_path / "cfg"
    shutil.copytree(get_path("configs", "test"), cfg_dir)
    train_yaml = cfg_dir / "train.yaml"
    cfg = yaml.safe_load(train_yaml.read_text())
    if cross_entropy is None:
        cfg.pop("cross_entropy", None)
    else:
        cfg["cross_entropy"] = cross_entropy
    if legacy is not None:
        cfg["legacy_train_settings"] = legacy
    train_yaml.write_text(yaml.safe_dump(cfg))
    return Trainer.from_config(str(train_yaml))


def test_a_fresh_run_must_use_the_margin_form(tmp_path):
    assert FRESH_RUN_TRAIN_SETTINGS["cross_entropy"] == "margin"
    trainer = _trainer(tmp_path)
    assert trainer.cross_entropy == "softmax"                 # the legacy default: old configs reproduce
    with pytest.raises(ValueError, match="cross_entropy: 'margin'"):
        trainer._check_fresh_run_settings()
    _trainer(tmp_path / "b", cross_entropy="margin")._check_fresh_run_settings()
    with pytest.raises(ValueError, match="needs a reason"):
        _trainer(tmp_path / "c", legacy={"cross_entropy": ""})._check_fresh_run_settings()
    _trainer(tmp_path / "d", legacy={"cross_entropy": "control arm reproducing T8"})._check_fresh_run_settings()


def test_an_unknown_form_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cross_entropy must be"):
        _trainer(tmp_path, cross_entropy="sparsemax")


def test_a_training_step_runs_with_the_margin_form(tmp_path):
    trainer = _trainer(tmp_path, cross_entropy="margin")
    with mock.patch("wandb.init"), mock.patch("wandb.log"):
        trainer.run(project_name="neural-symbolic-regression-test", entity="psaegert", name="pytest-margin-ce",
                    verbose=False, steps=1, device="cpu", preprocess=False, checkpoint_interval=None,
                    checkpoint_directory=None, wandb_mode="disabled")
