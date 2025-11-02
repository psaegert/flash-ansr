import contextlib
import datetime
import shutil
import unittest
from collections import Counter
from typing import Any

from unittest import mock

import torch

from flash_ansr import SkeletonPool, get_path
from flash_ansr.train import Trainer


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.val_skeleton_save_dir = get_path('data', 'test', 'skeleton_pool_val')

        # Create a skeleton pool
        pool = SkeletonPool.from_config(get_path('configs', 'test', 'skeleton_pool_val.yaml'))
        pool.create(size=10)
        pool.save(
            self.val_skeleton_save_dir,
            config=get_path('configs', 'test', 'skeleton_pool_val.yaml'))

    def tearDown(self) -> None:
        shutil.rmtree(self.val_skeleton_save_dir)

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
            'prompt_metadata': micro_batch.get('prompt_metadata', []),
        }


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = {"<pad>": 0}
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.embed = torch.nn.Embedding(10, 4)
        self.output = torch.nn.Linear(4, 6)

    def forward(self, input_ids: torch.Tensor, data_tensor: torch.Tensor, input_num: Any = None, data_attn_mask: Any = None) -> torch.Tensor:  # noqa: ANN401
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
    trainer.total_pflops = 0.0
    trainer.encoder_parameters = sum(p.numel() for p in trainer.model.encoder.parameters() if p.requires_grad)
    trainer.decoder_parameters = sum(p.numel() for p in trainer.model.decoder.parameters() if p.requires_grad)
    trainer._prompt_token_ids = {"complexity": None}
    trainer.prompt_combo_counts = Counter()
    trainer.prompt_total_samples = 0
    trainer._apply_prompt_mask = lambda batch: None
    trainer._update_total_pflops = lambda **_: None
    trainer._log_metrics = lambda *args, **kwargs: None
    return trainer


def _make_dummy_batch() -> dict[str, Any]:
    return {
        'x_tensors': [torch.zeros((1, 1, 1), dtype=torch.float32)],
        'y_tensors': [torch.zeros((1, 1, 1), dtype=torch.float32)],
        'input_ids': [[1, 2]],
        'prompt_metadata': [
            {'allowed_terms': [], 'include_terms': [], 'exclude_terms': []},
        ],
    }


def test_train_step_updates_prompt_statistics_when_preprocess_enabled() -> None:
    trainer = _build_dummy_trainer()
    spy = mock.Mock()
    trainer._update_prompt_statistics = spy  # type: ignore[assignment]

    with mock.patch('torch.autocast', lambda *args, **kwargs: contextlib.nullcontext()):
        trainer._train_step(batch=_make_dummy_batch(), step=0, preprocess=True)

    spy.assert_called_once()
    call_batch = spy.call_args.args[0]
    assert call_batch['prompt_metadata'][0] == {'allowed_terms': [], 'include_terms': [], 'exclude_terms': []}


def test_train_step_skips_prompt_statistics_when_preprocess_disabled() -> None:
    trainer = _build_dummy_trainer()
    spy = mock.Mock()
    trainer._update_prompt_statistics = spy  # type: ignore[assignment]

    with mock.patch('torch.autocast', lambda *args, **kwargs: contextlib.nullcontext()):
        trainer._train_step(batch=_make_dummy_batch(), step=0, preprocess=False)

    spy.assert_not_called()
