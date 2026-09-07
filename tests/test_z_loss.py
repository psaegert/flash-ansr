"""z-loss (`_z_loss`, train.yaml ``z_loss_weight``): the restoring force on the loss-flat
shared-logit-offset direction that cross-entropy cannot see. Weight 0.0 keeps the trainer
bit-for-bit as without the term; it is only ever added when the weight is positive.

Also the logit-scale alarms (`_logit_scale`): log Z and mean |logit| over the supervised
positions, logged every step so a drift is visible long before it costs loss.
"""
import math

import torch

from flash_ansr.train.train import _logit_scale, _z_loss


class TestZLoss:
    def test_uniform_logits_give_squared_log_vocab(self) -> None:
        logits = torch.zeros(7, 337)
        valid = torch.ones(7, dtype=torch.bool)
        expected = math.log(337.0) ** 2
        assert torch.isclose(_z_loss(logits, valid), torch.tensor(expected), rtol=1e-6)

    def test_shared_offset_is_penalized(self) -> None:
        # The exact pathology: a shared offset leaves CE unchanged but must move the z-loss.
        logits = torch.randn(11, 64)
        valid = torch.ones(11, dtype=torch.bool)
        shifted = logits - 100.0
        ce = torch.nn.functional.cross_entropy(logits, torch.zeros(11, dtype=torch.long))
        ce_shifted = torch.nn.functional.cross_entropy(shifted, torch.zeros(11, dtype=torch.long))
        assert torch.isclose(ce, ce_shifted, rtol=1e-5)
        assert _z_loss(shifted, valid) > _z_loss(logits, valid) + 1e3

    def test_only_supervised_positions_count(self) -> None:
        logits = torch.zeros(4, 16)
        logits[2] += 50.0  # a masked position must not contribute
        valid = torch.tensor([True, True, False, True])
        expected = math.log(16.0) ** 2
        assert torch.isclose(_z_loss(logits, valid), torch.tensor(expected), rtol=1e-6)

    def test_fp32_result_from_bf16_logits(self) -> None:
        # The offsets this loss exists to shrink sit exactly where bf16 resolution dies.
        logits = (torch.randn(5, 32) - 90.0).to(torch.bfloat16)
        valid = torch.ones(5, dtype=torch.bool)
        out = _z_loss(logits, valid)
        assert out.dtype == torch.float32
        assert torch.isfinite(out)

    def test_gradient_pulls_the_offset_back(self) -> None:
        logits = (torch.randn(6, 20) - 40.0).requires_grad_(True)
        valid = torch.ones(6, dtype=torch.bool)
        _z_loss(logits, valid).backward()
        # log Z < 0 here, so d/dlogit of log^2 Z is negative: the step raises every logit.
        assert logits.grad is not None
        assert (logits.grad < 0).all()


class TestLogitScale:
    def test_sums_and_count_over_supervised_positions(self) -> None:
        logits = torch.zeros(3, 8)
        logits[1] += 2.0
        valid = torch.tensor([True, True, False])
        log_z_sum, absmean_sum, n = _logit_scale(logits, valid)
        assert n == 2
        assert math.isclose(log_z_sum, math.log(8.0) + (2.0 + math.log(8.0)), rel_tol=1e-6)
        assert math.isclose(absmean_sum, 0.0 + 2.0, rel_tol=1e-6)

    def test_empty_selection(self) -> None:
        assert _logit_scale(torch.zeros(2, 4), torch.zeros(2, dtype=torch.bool)) == (0.0, 0.0, 0)

    def test_no_gradient_and_fp32_from_bf16(self) -> None:
        logits = (torch.randn(4, 16) * 300).to(torch.bfloat16).requires_grad_(True)
        log_z_sum, absmean_sum, n = _logit_scale(logits, torch.ones(4, dtype=torch.bool))
        assert n == 4 and math.isfinite(log_z_sum) and absmean_sum > 100
        assert logits.grad is None
