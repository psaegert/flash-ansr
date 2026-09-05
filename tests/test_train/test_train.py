import contextlib
import datetime
import unittest
from typing import Any

from unittest import mock

import torch

from flash_ansr import get_path
from flash_ansr.train import Trainer


class TestTrain(unittest.TestCase):
    # The datasets build their generative source directly from the migrated configs; no on-disk
    # skeleton-pool scaffolding is needed any more.
    @mock.patch('wandb.init')
    @mock.patch('wandb.log')
    def test_train(self, mock_log, mock_init):
        trainer = Trainer.from_config(get_path('configs', 'test', 'train.yaml'))

        steps = 2
        device = 'cpu'

        trainer.run(
            project_name='neural-symbolic-regression-test',
            entity='psaegert',
            name=f'pytest-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}',
            verbose=True,
            steps=steps,
            device=device,
            preprocess=False,
            checkpoint_interval=None,
            checkpoint_directory=None,
            wandb_mode="disabled",
            validate_size=10,
            validate_interval=1)


class _DummyDataset:
    def collate(self, micro_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            'input_ids': torch.tensor([[1, 2]], dtype=torch.long),
            'x_tensors': torch.zeros((1, 1, 1), dtype=torch.float32),
            'y_tensors': torch.zeros((1, 1, 1), dtype=torch.float32),
            'labels': torch.tensor([[1]], dtype=torch.long),
            'data_attn_mask': torch.ones((1, 2, 2), dtype=torch.bool),
        }


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = {"<pad>": 0}
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.embed = torch.nn.Embedding(10, 4)
        self.output = torch.nn.Linear(4, 6)

    def forward(self, input_ids: torch.Tensor, data_tensor: torch.Tensor, input_num: Any = None, data_attn_mask: Any = None, condition_mask: Any = None) -> torch.Tensor:  # noqa: ANN401
        embedded = self.embed(input_ids.long())
        return self.output(embedded)


def _build_dummy_trainer() -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.model = _DummyModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.amp_dtype = torch.bfloat16
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.lr_scheduler = None
    trainer.batch_size = 1
    trainer.train_dataset = _DummyDataset()
    trainer.val_dataset = None  # type: ignore[assignment]
    trainer.gradient_accumulation_steps = 1
    trainer.config = {}
    trainer.device = torch.device('cpu')
    trainer.metrics_ignore_index = -100
    trainer.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
    trainer.outlier_loss_weight = 0.0
    trainer.outlier_pos_weight = 1.0
    trainer.residual_loss_weight = 0.0
    trainer.residual_scale = 'none'
    trainer.total_pflops = 0.0
    trainer.encoder_parameters = sum(p.numel() for p in trainer.model.encoder.parameters() if p.requires_grad)
    trainer.decoder_parameters = sum(p.numel() for p in trainer.model.decoder.parameters() if p.requires_grad)
    trainer._prompt_token_ids = {"complexity": None}
    trainer._apply_prompt_mask = lambda batch: None
    trainer._update_total_pflops = lambda **_: None
    trainer._log_metrics = lambda *args, **kwargs: None
    return trainer


def _make_dummy_batch() -> dict[str, Any]:
    return {
        'x_tensors': [torch.zeros((1, 1, 1), dtype=torch.float32)],
        'y_tensors': [torch.zeros((1, 1, 1), dtype=torch.float32)],
        'input_ids': [[1, 2]],
    }


def _grads(trainer: Trainer) -> list[torch.Tensor]:
    return [p.grad.detach().clone() for p in trainer.model.parameters() if p.grad is not None]


def test_gradients_accumulate_across_calls_when_not_stepping() -> None:
    """do_optimizer_step=False must leave gradients in place for the next call."""
    trainer = _build_dummy_trainer()
    batch = _make_dummy_batch()

    with mock.patch('torch.autocast', lambda *args, **kwargs: contextlib.nullcontext()):
        trainer._train_step(batch=batch, step=0, preprocess=False, do_optimizer_step=False)
        after_one = _grads(trainer)
        trainer._train_step(batch=batch, step=1, preprocess=False, do_optimizer_step=False)
        after_two = _grads(trainer)

    assert after_one, "no gradients after the first accumulation call"
    assert len(after_one) == len(after_two)
    # Same batch twice, so the accumulated gradient should be twice the single-call gradient.
    for one, two in zip(after_one, after_two):
        assert torch.allclose(two, one * 2, atol=1e-5), "gradients were cleared between calls"


def test_no_parameter_update_when_not_stepping() -> None:
    """do_optimizer_step=False must not move the parameters."""
    trainer = _build_dummy_trainer()
    before = [p.detach().clone() for p in trainer.model.parameters()]

    with mock.patch('torch.autocast', lambda *args, **kwargs: contextlib.nullcontext()):
        trainer._train_step(batch=_make_dummy_batch(), step=0, preprocess=False, do_optimizer_step=False)

    for was, now in zip(before, trainer.model.parameters()):
        assert torch.equal(was, now), "parameters moved despite do_optimizer_step=False"


def test_step_clears_gradients_for_the_next_cycle() -> None:
    """A stepping call must leave gradients zeroed so the next cycle starts clean."""
    trainer = _build_dummy_trainer()

    with mock.patch('torch.autocast', lambda *args, **kwargs: contextlib.nullcontext()):
        trainer._train_step(batch=_make_dummy_batch(), step=0, preprocess=False, do_optimizer_step=True)

    for grad in _grads(trainer):
        assert torch.count_nonzero(grad) == 0, "gradients survived an optimizer step"
