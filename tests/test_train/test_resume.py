import os
import tempfile
from unittest import mock
import copy
import random

import torch

from flash_ansr.train import Trainer
from flash_ansr import get_path


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):  # pragma: no cover - not used
        return self.linear(x)


def _build_resume_trainer(with_scheduler: bool = True, scaler_enabled: bool = True) -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.model = _TinyModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.amp_dtype = torch.float16 if scaler_enabled else torch.bfloat16
    trainer.scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    trainer.lr_scheduler = (
        torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lambda step: 1.0) if with_scheduler else None
    )
    trainer.batch_size = 1
    trainer.train_dataset = None  # type: ignore[assignment]
    trainer.val_dataset = None  # type: ignore[assignment]
    trainer.gradient_accumulation_steps = 1
    trainer.config = {}
    trainer.device = torch.device("cpu")
    trainer.metrics_ignore_index = -100
    trainer.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
    trainer.total_pflops = 0.0
    trainer.encoder_parameters = 1
    trainer.decoder_parameters = 1
    trainer._prompt_token_ids = {"complexity": None}
    trainer.prompt_combo_counts = {}
    trainer.prompt_total_samples = 0
    return trainer


def test_load_checkpoint_reconstructs_scheduler_and_scaler_absent() -> None:
    trainer = _build_resume_trainer(with_scheduler=True, scaler_enabled=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoint_5")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model and optimizer state
        torch.save(trainer.model.state_dict(), os.path.join(ckpt_dir, "state_dict.pt"))
        torch.save(trainer.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        # Persist training metadata but omit scheduler/scaler states
        torch.save({"step": 5, "total_pflops": 123.0}, os.path.join(ckpt_dir, "training_state.pt"))

        # Change model weights to verify they are restored
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(1.0)

        resume_step = trainer._load_checkpoint(ckpt_dir, device="cpu")

        # Scheduler state missing: it should be reconstructed at last_epoch=step-1
        assert resume_step == 5
        assert trainer.lr_scheduler is not None
        assert trainer.lr_scheduler.last_epoch == 4
        assert trainer.total_pflops == 123.0
        # Model weights should match the checkpoint contents
        loaded_state = torch.load(os.path.join(ckpt_dir, "state_dict.pt"), map_location="cpu")
        for name, param in trainer.model.state_dict().items():
            assert torch.allclose(param, loaded_state[name])


def test_run_training_uses_step_offset_for_resume_step_override() -> None:
    trainer = _build_resume_trainer(with_scheduler=False, scaler_enabled=False)
    trainer.worker_preprocess = False
    trainer.num_workers = None
    trainer.decoder_max_seq_len = 8

    class _IterDataset:
        def iterate(self, steps: int, batch_size: int, preprocess: bool, preprocess_in_worker: bool,
                    num_workers: int | None, max_seq_len: int):
            for _ in range(steps):
                yield {"x_tensors": [], "y_tensors": [], "input_ids": [], "data_attn_mask": [], "labels": []}

    trainer.train_dataset = _IterDataset()

    trainer._setup_training_state = lambda device, verbose=False: None  # type: ignore[assignment]
    train_step_spy = mock.Mock()
    trainer._train_step = train_step_spy  # type: ignore[assignment]
    trainer._validate_step = lambda *args, **kwargs: None  # type: ignore[assignment]
    trainer._save_checkpoint = lambda *args, **kwargs: None  # type: ignore[assignment]

    trainer.run_training(
        steps=5,
        preprocess=False,
        device="cpu",
        compile_mode=None,
        checkpoint_interval=None,
        checkpoint_directory=None,
        validate_interval=None,
        validate_size=None,
        validate_batch_size=1,
        preprocess_in_worker=None,
        resume_from=None,
        resume_step=3,
        verbose=False,
    )

    # Should have processed only the remaining 2 steps and start at global steps 3 and 4
    called_steps = [call.args[1] for call in train_step_spy.call_args_list]
    assert called_steps == [3, 4]


def test_resume_matches_baseline_loss_curve() -> None:
    torch.manual_seed(0)
    random.seed(0)

    total_steps = 64
    half_steps = total_steps // 2

    # Materialize real training batches once for reuse to make runs comparable.
    base_trainer = Trainer.from_config(get_path('configs', 'test', 'train.yaml'))
    base_trainer.worker_preprocess = False
    batches: list[dict[str, torch.Tensor]] = []
    for batch in base_trainer.train_dataset.iterate(
            steps=total_steps,
            batch_size=base_trainer.batch_size,
            preprocess=False,
            preprocess_in_worker=False,
            num_workers=base_trainer.num_workers,
            max_seq_len=base_trainer.decoder_max_seq_len):
        batches.append(copy.deepcopy(batch))

    def _run_training_with_batches(step_count: int, batches_slice: list[dict[str, torch.Tensor]],
                                   *, resume_from: str | None = None) -> list[float]:
        # Keep initial weights deterministic across runs.
        torch.manual_seed(0)
        random.seed(0)

        trainer = Trainer.from_config(get_path('configs', 'test', 'train.yaml'))
        trainer.worker_preprocess = False

        def _iter_batches(*_args, **_kwargs):  # type: ignore[override]
            for b in batches_slice:
                yield copy.deepcopy(b)

        trainer.train_dataset.iterate = _iter_batches  # type: ignore[assignment]

        logged_losses: list[float] = []
        trainer._log_metrics = lambda step, logits, labels, ce_loss, total_loss, total_gradient_norm: logged_losses.append(ce_loss)  # type: ignore[assignment]
        trainer._validate_step = lambda *args, **kwargs: None  # type: ignore[assignment]

        trainer.run_training(
            steps=step_count,
            preprocess=False,
            device="cpu",
            compile_mode=None,
            checkpoint_interval=None,
            checkpoint_directory=None,
            validate_interval=None,
            validate_size=None,
            validate_batch_size=1,
            preprocess_in_worker=False,
            resume_from=resume_from,
            resume_step=None,
            verbose=False,
        )

        return logged_losses

    # Baseline: full uninterrupted run
    baseline_losses = _run_training_with_batches(total_steps, batches)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, f"checkpoint_{half_steps}")

        # First half: run and write checkpoint manually using trainer's saver to preserve metadata
        torch.manual_seed(0)
        random.seed(0)
        first_half_trainer = Trainer.from_config(get_path('configs', 'test', 'train.yaml'))
        first_half_trainer.worker_preprocess = False

        def _iter_first_half(*_args, **_kwargs):  # type: ignore[override]
            for b in batches[:half_steps]:
                yield copy.deepcopy(b)

        first_half_trainer.train_dataset.iterate = _iter_first_half  # type: ignore[assignment]
        first_half_trainer._log_metrics = lambda *args, **kwargs: None  # type: ignore[assignment]
        first_half_trainer._validate_step = lambda *args, **kwargs: None  # type: ignore[assignment]

        # Run half the steps and force a checkpoint at the end
        first_half_trainer.run_training(
            steps=half_steps,
            preprocess=False,
            device="cpu",
            compile_mode=None,
            checkpoint_interval=None,
            checkpoint_directory=None,
            validate_interval=None,
            validate_size=None,
            validate_batch_size=1,
            preprocess_in_worker=False,
            resume_from=None,
            resume_step=None,
            verbose=False,
        )
        # Manually save checkpoint at half_steps
        first_half_trainer._save_checkpoint(step=half_steps, checkpoint_directory=tmpdir)

        # Resume for the second half using the saved checkpoint
        resumed_losses = _run_training_with_batches(total_steps, batches[half_steps:], resume_from=ckpt_dir)

    # The resumed loss trajectory should match the baseline's second half
    assert len(resumed_losses) == len(baseline_losses[half_steps:])
    for resumed, expected in zip(resumed_losses, baseline_losses[half_steps:]):
        torch.testing.assert_close(torch.tensor(resumed), torch.tensor(expected), rtol=5e-3, atol=1e-2)
