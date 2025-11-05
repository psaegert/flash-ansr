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
        trainer = Trainer.from_config(get_path('configs', 'test', 'train_no_numeric_head.yaml'))

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

        trainer.train_dataset.shutdown()
        trainer.val_dataset.shutdown()

    @mock.patch('wandb.init')
    @mock.patch('wandb.log')
    def test_train_numeric_head_mdn_initialises(self, mock_log, mock_init):
        trainer = Trainer.from_config(get_path('configs', 'test', 'train_numeric_head.yaml'))
        assert trainer.model.use_numeric_head is True
        assert trainer.model.numeric_head_kind == 'mdn'
        trainer.train_dataset.shutdown()
        trainer.val_dataset.shutdown()

    @mock.patch('wandb.init')
    @mock.patch('wandb.log')
    def test_train_numeric_head_deterministic_initialises(self, mock_log, mock_init):
        trainer = Trainer.from_config(get_path('configs', 'test', 'train_numeric_head_deterministic.yaml'))
        assert trainer.model.use_numeric_head is True
        assert trainer.model.numeric_head_kind == 'deterministic'
        trainer.train_dataset.shutdown()
        trainer.val_dataset.shutdown()


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
        logits = self.output(embedded)
        return logits, None


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


def _run_train_step_for_config(config_filename: str) -> float:
    trainer = Trainer.from_config(get_path('configs', 'test', config_filename))
    trainer.device = torch.device('cpu')
    trainer.model.to(trainer.device)
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.gradient_accumulation_steps = 1

    tokenizer = trainer.model.tokenizer
    input_tokens = [
        tokenizer['<bos>'],
        tokenizer['<expression>'],
        tokenizer['*'],
        tokenizer['x1'],
        tokenizer['<constant>'],
        tokenizer['</expression>'],
        tokenizer['<eos>'],
    ]

    support_points = 4
    n_variables = trainer.model.encoder_max_n_variables - 1
    x_support = torch.zeros((support_points, n_variables), dtype=torch.float32)
    x_support[:, 0] = torch.linspace(-1.0, 1.0, steps=support_points)
    y_support = (2.0 * x_support[:, 0]).unsqueeze(-1)
    data_attn_mask = torch.ones((1, support_points), dtype=torch.bool)

    raw_batch = {
        'input_ids': [input_tokens],
        'x_tensors': [x_support],
        'y_tensors': [y_support],
        'constants': [torch.tensor([2.0], dtype=torch.float32)],
        'data_attn_mask': data_attn_mask,
        'prompt_metadata': [{}],
    }

    trainer.train_dataset._ensure_numeric_channel(raw_batch)
    collated_batch = trainer.train_dataset.collate(raw_batch, device=trainer.device)
    trainer._apply_prompt_mask = lambda batch: None

    metrics_spy = mock.Mock()
    trainer._log_metrics = metrics_spy  # type: ignore[assignment]

    with mock.patch('torch.autocast', lambda *args, **kwargs: contextlib.nullcontext()):
        trainer._train_step(batch=collated_batch, step=0, preprocess=False)

    metrics_spy.assert_called_once()
    call_args = metrics_spy.call_args.args
    numeric_loss_per_token = call_args[6]
    numeric_loss_per_constant = call_args[7]
    numeric_fraction = call_args[8]
    numeric_count = call_args[9]
    sigma_mean = call_args[10]
    sigma_min = call_args[11]
    sigma_max = call_args[12]
    sigma_clamp_fraction = call_args[13]
    sigma_count = call_args[14]

    assert numeric_loss_per_token is not None
    assert numeric_loss_per_token >= 0.0
    assert numeric_loss_per_constant is not None
    assert numeric_loss_per_constant >= 0.0
    assert numeric_fraction is not None and numeric_fraction >= 0.0
    assert numeric_count is not None and numeric_count >= 0

    if 'deterministic' in config_filename:
        assert sigma_mean is None
        assert sigma_min is None
        assert sigma_max is None
        assert sigma_clamp_fraction is None
        assert sigma_count is None
    else:
        assert sigma_mean is not None and sigma_mean > 0.0
        assert sigma_min is not None and sigma_min > 0.0
        assert sigma_max is not None and sigma_max >= sigma_min
        assert sigma_clamp_fraction is not None and 0.0 <= sigma_clamp_fraction <= 1.0
        assert sigma_count is not None and sigma_count > 0

    trainer.train_dataset.shutdown()
    if trainer.val_dataset is not None:
        trainer.val_dataset.shutdown()
    return float(numeric_loss_per_token)


def test_train_step_logs_numeric_loss_for_constant_tokens_mdn() -> None:
    numeric_loss = _run_train_step_for_config('train_numeric_head.yaml')
    assert numeric_loss >= 0.0


def test_train_step_logs_numeric_loss_for_constant_tokens_deterministic() -> None:
    numeric_loss = _run_train_step_for_config('train_numeric_head_deterministic.yaml')
    assert numeric_loss >= 0.0


def test_apply_prompt_mask_sets_ignore_indices() -> None:
    trainer = _build_dummy_trainer()
    trainer.device = torch.device('cpu')

    labels = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    prompt_mask = torch.tensor([[False, True, False, True, False]], dtype=torch.bool)

    batch = {
        'labels': labels.clone(),
        'prompt_mask': prompt_mask,
    }

    trainer._apply_prompt_mask(batch)

    updated_labels = batch['labels']
    assert updated_labels.shape == labels.shape
    assert updated_labels[0, 0] == trainer.metrics_ignore_index
    assert updated_labels[0, 1] == 11
    assert updated_labels[0, 2] == trainer.metrics_ignore_index
    assert updated_labels[0, 3] == 13
    assert batch['prompt_mask'].dtype == torch.bool
