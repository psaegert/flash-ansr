"""Training orchestration logic for Flash-ANSR models.

This module provides the `Trainer` class, which encapsulates the full
training/validation loop as well as utilities for configuration driven
initialisation and experiment logging.
"""

import os
import copy
from typing import Any, Literal

import torch

from torch.optim.lr_scheduler import LRScheduler
from torch import nn

from tqdm import tqdm

from flash_ansr.model import FlashANSRModel
from flash_ansr.data import FlashANSRDataset
from flash_ansr.utils import load_config, save_config, substitute_root_path, unfold_config
from flash_ansr.eval.token_prediction import correct_token_predictions_at_k, reciprocal_rank
from flash_ansr.train.utils import OptimizerFactory, pw_linear_schedule

import wandb


torch.set_float32_matmul_precision('high')


class Trainer:
    """Manage end-to-end training for a ``FlashANSRModel``.

    Notes
    -----
    Parameters mirror the components required to run training while being
    flexible enough to support configuration driven instantiation via
    :meth:`Trainer.from_config`.
    """

    def __init__(
            self,
            model: FlashANSRModel,
            optimizer: torch.optim.Optimizer,
            amp_dtype: torch.dtype,
            scaler: torch.amp.GradScaler,
            lr_scheduler: LRScheduler | None,
            batch_size: int,
            train_dataset: FlashANSRDataset,
            val_dataset: FlashANSRDataset,
            gradient_accumulation_steps: int = 1,
            config: dict[str, Any] = None) -> None:
        """Create a fully configured trainer instance.

        Parameters
        ----------
        model : FlashANSRModel
            The model to optimise.
        optimizer : torch.optim.Optimizer
            Optimiser responsible for updating model parameters.
        amp_dtype : torch.dtype
            Mixed precision dtype used with autocast.
        scaler : torch.amp.GradScaler
            Gradient scaler used for automatic mixed precision.
        lr_scheduler : LRScheduler or None
            Optional learning rate scheduler.
        batch_size : int
            Number of samples processed per training step.
        train_dataset : FlashANSRDataset
            Dataset providing training batches.
        val_dataset : FlashANSRDataset
            Dataset providing validation batches.
        gradient_accumulation_steps : int, default=1
            Number of micro-batches per optimizer step.
        config : dict[str, Any] or None, optional
            Trainer configuration metadata (used for logging).
        """

        self.model = model
        self.optimizer = optimizer
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.config = config or {}
        self.device = torch.device("cpu")  # Updated during ``run``

        # Metrics and Loss Functions
        self.metrics_ignore_index = self.model.tokenizer["<pad>"]
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.metrics_ignore_index)

        self.total_pflops = 0.0
        self.encoder_parameters = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        self.decoder_parameters = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Trainer":
        """Instantiate a trainer from a config dictionary or YAML file path.

        Parameters
        ----------
        config : dict[str, Any] or str
            Dictionary of trainer settings or path to a YAML configuration file.

        Returns
        -------
        Trainer
            Trainer populated using the provided configuration.
        """
        config_ = load_config(config)

        if "trainer" in config_.keys():
            config_ = config_["trainer"]

        # Handle relative model paths
        if isinstance(config, str) and isinstance(config_["model"], str) and config_["model"].startswith('.'):
            config_["model"] = os.path.join(os.path.dirname(config), config_["model"])

        print(f'Creating model from {config_["model"]}')
        model = FlashANSRModel.from_config(config_["model"])

        print(f'Loading optimizer with config {config_["optimizer"]}')
        optimizer = OptimizerFactory.get_optimizer(config_['optimizer']['name'], params=model.parameters(), **config_['optimizer'].get('kwargs', {}))

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

        print(f'Loading lr_scheduler with config {config_["lr_schedule"]}')
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: pw_linear_schedule(step, config_['lr_schedule'])  # Pass the step and the points that define the piecewise linear schedule
        )

        print(f'Loading train_dataset with config {config_["train_dataset"]}')
        train_dataset = FlashANSRDataset.from_config(config_["train_dataset"])

        print(f'Loading val_dataset with config {config_["val_dataset"]}')
        if isinstance(config_["val_dataset"], str):
            resolved_config_path = substitute_root_path(config_["val_dataset"])
            if os.path.isfile(resolved_config_path):
                val_dataset = FlashANSRDataset.from_config(resolved_config_path)
            elif os.path.isdir(resolved_config_path):
                _, val_dataset = FlashANSRDataset.load(directory=resolved_config_path)
            else:
                raise ValueError(f"val_dataset must be a file or directory, not {resolved_config_path}")
        elif isinstance(config_["val_dataset"], dict):
            val_dataset = FlashANSRDataset.from_config(config_["val_dataset"])
        else:
            raise ValueError(f"val_dataset must be a dict or path to a file or directory, not {config_['val_dataset']}")

        return cls(
            model=model,
            optimizer=optimizer,
            amp_dtype=amp_dtype,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            batch_size=config_['batch_size'],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            gradient_accumulation_steps=config_.get("gradient_accumulation_steps", 1),
            config=config_
        )

    def _setup_training_state(self, device: str, verbose: bool) -> None:
        """Move the model to ``device`` and reset training state.

        Parameters
        ----------
        device : str
            Torch device string used for training (for example ``"cuda"``).
        verbose : bool
            If ``True`` emit human readable status messages.
        """
        self.device = torch.device(device)
        self.model.to(self.device)

        self.max_set_size = self.train_dataset.skeleton_pool.n_support_prior_config['kwargs']['max_value']
        self.total_pflops = 0.0

        if verbose:
            print(f"Padding sequences to max set size of {self.max_set_size}")

            config_value = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
            if config_value:
                print(f"PYTORCH_CUDA_ALLOC_CONF is set to: '{config_value}'")
            else:
                print("PYTORCH_CUDA_ALLOC_CONF is not set.")

    def run_training(
            self,
            steps: int,
            preprocess: bool = False,
            device: str = "cpu",
            compile_mode: str | None = None,
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            validate_size: int | None = None,
            validate_batch_size: int = 128,
            verbose: bool = False) -> FlashANSRModel:
        """Execute the core training loop.

        Parameters
        ----------
        steps : int
            Number of optimisation steps to perform.
        preprocess : bool, default=False
            Whether to run dataset preprocessing before each batch.
        device : str, default='cpu'
            Target device identifier passed to :func:`torch.device`.
        compile_mode : str or None, optional
            Torch.compile mode for model compilation.
        checkpoint_interval : int or None, optional
            Save checkpoints every ``n`` steps when provided.
        checkpoint_directory : str or None, optional
            Directory for persisted checkpoints.
        validate_interval : int or None, optional
            Frequency (in steps) to run validation.
        validate_size : int or None, optional
            Limit the number of validation examples processed.
        validate_batch_size : int, default=128
            Batch size used for validation passes.
        verbose : bool, default=False
            If ``True`` display progress bars and additional logs.

        Returns
        -------
        FlashANSRModel
            The trained model instance.
        """

        try:
            if compile_mode is not None:
                self.model = torch.compile(self.model, mode=compile_mode)

            self._setup_training_state(device, verbose=verbose)

            pbar = tqdm(range(steps), disable=not verbose, smoothing=0, desc="Training")
            for step, batch in enumerate(self.train_dataset.iterate(steps=steps, batch_size=self.batch_size, preprocess=preprocess)):
                self._train_step(batch, step, preprocess, do_optimizer_step=True)

                pbar.update(1)

                if validate_interval is not None and ((step + 1) % validate_interval == 0 or step == steps - 1):
                    self._validate_step(step + 1, validate_size, validate_batch_size, preprocess, verbose)

                if checkpoint_interval is not None and checkpoint_directory is not None and (step + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(step + 1, checkpoint_directory)

            pbar.close()
            return self.model

        except Exception:
            self.train_dataset.shutdown()
            self.val_dataset.shutdown()
            raise

    def run(
            self,
            project_name: str,
            entity: str,
            name: str,
            steps: int,
            preprocess: bool = False,
            device: str = "cpu",
            compile_mode: str | None = None,
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            validate_size: int | None = None,
            validate_batch_size: int = 128,
            wandb_mode: Literal['online', 'offline', 'disabled'] = 'online',
            wandb_watch_log: Literal['gradients', 'parameters', 'all'] | None = None,
            wandb_watch_log_freq: int = 1000,
            verbose: bool = False) -> FlashANSRModel:
        """Train the model while managing the experiment lifecycle via W&B.

        Parameters
        ----------
        project_name : str
            Name of the Weights & Biases project.
        entity : str
            W&B entity (team or user) under which to log the run.
        name : str
            Run name displayed in W&B.
        steps : int
            Number of training iterations to execute.
        preprocess : bool, default=False
            Whether to preprocess dataset batches.
        device : str, default='cpu'
            Torch device string (for example ``"cpu"`` or ``"cuda"``).
        compile_mode : str or None, optional
            Torch.compile mode, if supported.
        checkpoint_interval : int or None, optional
            Step interval for checkpointing.
        checkpoint_directory : str or None, optional
            Directory where checkpoints are written when enabled.
        validate_interval : int or None, optional
            Step cadence for validation.
        validate_size : int or None, optional
            Maximum number of validation samples, if limited.
        validate_batch_size : int, default=128
            Batch size to use during validation.
        wandb_mode : {'online', 'offline', 'disabled'}, default='online'
            W&B initialisation mode.
        wandb_watch_log : {'gradients', 'parameters', 'all'} or None, optional
            W&B ``watch`` setting controlling what to log.
        wandb_watch_log_freq : int, default=1000
            Frequency (steps) for W&B gradient/parameter logging.
        verbose : bool, default=False
            When ``True`` print progress information to stdout.

        Returns
        -------
        FlashANSRModel
            The trained model instance.
        """

        if verbose:
            print(f"Training model ({self.model.n_params:,} parameters) for {steps:,} steps on device {device}")

        wandb_config = unfold_config(copy.deepcopy(self.config))
        wandb_config.update({"steps": steps, "device": device, "verbose": verbose})

        with wandb.init(config=wandb_config, project=project_name, entity=entity, name=name, mode=wandb_mode):  # type: ignore
            if wandb_mode != 'disabled':
                wandb.watch(self.model, log=wandb_watch_log, log_freq=wandb_watch_log_freq)  # type: ignore
                if verbose and wandb_watch_log is not None:
                    print(f'Watching model with wandb log={wandb_watch_log} at frequency {wandb_watch_log_freq}')

            return self.run_training(
                steps=steps,
                preprocess=preprocess,
                device=device,
                compile_mode=compile_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_directory=checkpoint_directory,
                validate_interval=validate_interval,
                validate_size=validate_size,
                validate_batch_size=validate_batch_size,
                verbose=verbose
            )

    def _update_total_pflops(self, encoder_tokens: int, decoder_tokens: int, batch_size: int) -> None:
        """Accumulate an estimate of total pico floating-point operations.

        Parameters
        ----------
        encoder_tokens : int
            Number of encoder tokens processed in the current step.
        decoder_tokens : int
            Number of decoder tokens processed in the current step.
        batch_size : int
            Size of the current batch (prior to micro batching).
        """
        self.total_pflops += 6 * (self.encoder_parameters * encoder_tokens + self.decoder_parameters * decoder_tokens) * batch_size * 1e-15

    def _train_step(self, batch: dict[str, torch.Tensor], step: int, preprocess: bool, do_optimizer_step: bool = True) -> None:
        """Perform a single optimisation step with optional gradient accumulation.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batched tensors produced by the training dataset.
        step : int
            Zero-indexed global training step.
        preprocess : bool
            Whether to preprocess batch data prior to collation.
        do_optimizer_step : bool, default=True
            When ``False`` skip the optimiser update (useful for gradient
            accumulation across micro-batches).
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_ce_loss = 0.0
        total_loss = 0.0

        # Split the batch into micro-batches to support gradient accumulation
        for acc_step in range(self.gradient_accumulation_steps):
            micro_batch_size = len(batch['x_tensors']) // self.gradient_accumulation_steps
            micro_batch = {k: v[acc_step * micro_batch_size:(acc_step + 1) * micro_batch_size] for k, v in batch.items()}
            micro_batch = self.train_dataset.collate(micro_batch, device=self.device)

            data_tensor = torch.cat([micro_batch['x_tensors'], micro_batch['y_tensors']], dim=-1)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits = self.model(micro_batch['input_ids'], data_tensor, input_num=micro_batch.get('input_num', None), data_attn_mask=micro_batch['data_attn_mask'].to(self.device))
                flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                flat_labels = micro_batch['labels'].reshape(-1)
                ce_loss = self.cross_entropy_loss(flat_logits, flat_labels)

                # Force every parameter to contribute to the loss so that gradient
                # tracking tools (e.g. wandb.watch) do not encounter ``None`` gradients
                # for frozen or unused tensors.
                param_sum = sum(p.sum() for p in self.model.parameters())
                zero_loss = 0.0 * param_sum

                loss = ce_loss / self.gradient_accumulation_steps + zero_loss

            # If the loss is nan or inf, stop the training
            if not torch.isfinite(loss):
                raise ValueError(f"Loss is {loss.item()}, stopping training")

            self.scaler.scale(loss).backward()
            total_ce_loss += ce_loss.item()
            total_loss += loss.item() * self.gradient_accumulation_steps

        # Perform the optimizer step
        self.scaler.unscale_(self.optimizer)

        self._update_total_pflops(encoder_tokens=data_tensor.shape[1], decoder_tokens=micro_batch['input_ids'].shape[1], batch_size=len(batch['x_tensors']))

        total_gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        if do_optimizer_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Log metrics and update scheduler after the optimizer step
            self._log_metrics(step, flat_logits, flat_labels, total_ce_loss, total_loss, total_gradient_norm)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _validate_step(self, step: int, size: int | None, batch_size: int, preprocess: bool, verbose: bool) -> None:
        """Evaluate the model on the validation split and log aggregate metrics.

        Parameters
        ----------
        step : int
            Global training step at which validation is triggered.
        size : int or None
            Optional limit on the number of validation examples.
        batch_size : int
            Number of samples per validation batch.
        preprocess : bool
            Whether to preprocess validation batches.
        verbose : bool
            If ``True`` display a validation progress bar.
        """
        self.model.eval()

        val_ce_loss = 0.0
        val_mrr = 0.0
        val_acc_at_1 = 0.0
        total_items = 0
        total_batches = 0

        with torch.no_grad():
            if size is None:
                steps = len(self.val_dataset) // batch_size
            else:
                steps = size // batch_size

            pbar = tqdm(total=steps, leave=False, position=1, disable=not verbose, desc="Validating", smoothing=0.0)
            for batch in self.val_dataset.iterate(size=size, batch_size=batch_size, preprocess=preprocess):
                batch = self.val_dataset.collate(batch, device=self.device)
                data_tensor = torch.cat([batch['x_tensors'], batch['y_tensors']], dim=-1)

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits = self.model(batch['input_ids'], data_tensor, input_num=batch.get('input_num', None), data_attn_mask=batch['data_attn_mask'].to(self.device))
                    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                    flat_labels = batch['labels'].reshape(-1)
                    ce_loss = self.cross_entropy_loss(flat_logits, flat_labels)

                    # Accumulate metrics for each batch
                    val_ce_loss += ce_loss.item() * flat_labels.shape[0]
                    total_items += flat_labels.shape[0]

                # Filter out ignored indices for metric calculation
                valid_indices = flat_labels != self.metrics_ignore_index
                if valid_indices.any():
                    val_mrr += reciprocal_rank(flat_logits[valid_indices], flat_labels[valid_indices])
                    val_acc_at_1 += correct_token_predictions_at_k(flat_logits[valid_indices], flat_labels[valid_indices], k=1)
                    total_batches += 1

                pbar.update(1)
            pbar.close()

        # Calculate average metrics
        avg_val_ce_loss = val_ce_loss / total_items if total_items > 0 else 0.0
        avg_val_mrr = val_mrr / total_batches if total_batches > 0 else 0.0
        avg_val_acc_at_1 = val_acc_at_1 / total_batches if total_batches > 0 else 0.0

        # Log averaged validation metrics
        self._log_validation_metrics(step, avg_val_ce_loss, avg_val_mrr, avg_val_acc_at_1)

    def _save_checkpoint(self, step: int, checkpoint_directory: str) -> None:
        """Persist model weights, optimiser state, and config for ``step``.

        Parameters
        ----------
        step : int
            Training step associated with the persisted checkpoint.
        checkpoint_directory : str
            Directory into which the checkpoint will be written.
        """
        save_directory = os.path.join(checkpoint_directory, f"checkpoint_{step}")
        self.model.save(directory=save_directory, errors='ignore')
        torch.save(self.optimizer.state_dict(), os.path.join(save_directory, "optimizer.pt"))
        save_config(
            load_config(self.config, resolve_paths=True),
            directory=save_directory,
            filename='train.yaml',
            reference='relative',
            recursive=True,
            resolve_paths=True)
        print(f"Checkpoint saved at {save_directory}")

    def _log_metrics(self, step: int, logits: torch.Tensor, labels: torch.Tensor, ce_loss: float, total_loss: float, total_gradient_norm: torch.Tensor) -> None:
        """Submit training metrics for the current batch to Weights & Biases.

        Parameters
        ----------
        step : int
            Global training step the metrics correspond to.
        logits : torch.Tensor
            Model logits for the current batch.
        labels : torch.Tensor
            Ground-truth labels aligned with ``logits``.
        ce_loss : float
            Cross-entropy loss for the batch.
        total_loss : float
            Total loss (including auxiliary terms) for the batch.
        total_gradient_norm : torch.Tensor
            Gradient norm measured after clipping.
        """

        log_data = {
            "total_gradient_norm": total_gradient_norm.item(),
            "train_ce_loss": ce_loss,
            "train_loss": total_loss,
            "lr": self.optimizer.param_groups[0]['lr'],
            "train_mean_reciprocal_rank": reciprocal_rank(logits, labels, ignore_index=self.metrics_ignore_index),
            "train_correct_token_predictions_at_1": correct_token_predictions_at_k(logits, labels, k=1, ignore_index=self.metrics_ignore_index),
            "total_pflops": self.total_pflops,
        }
        wandb.log(log_data, step=step)  # type: ignore

    def _log_validation_metrics(self, step: int, val_ce_loss: float, val_mrr: float, val_acc_at_1: float) -> None:
        """Submit aggregated validation metrics to Weights & Biases.

        Parameters
        ----------
        step : int
            Global training step the metrics correspond to.
        val_ce_loss : float
            Mean validation cross-entropy loss.
        val_mrr : float
            Mean reciprocal rank on the validation set.
        val_acc_at_1 : float
            Accuracy at one token prediction on the validation set.
        """
        log_data = {
            "val_ce_loss": val_ce_loss,
            "val_mean_reciprocal_rank": val_mrr,
            "val_correct_token_predictions_at_1": val_acc_at_1,
        }
        wandb.log(log_data, step=step)  # type: ignore
