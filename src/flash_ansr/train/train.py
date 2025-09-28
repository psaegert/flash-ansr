import os
import copy
from typing import Any, Literal

import torch
import torch_optimizer

from torch.optim.lr_scheduler import LRScheduler
from torch import nn

from tqdm import tqdm

from flash_ansr.model import FlashANSRModel
from flash_ansr.data import FlashANSRDataset
from flash_ansr.utils import load_config, save_config, substitute_root_path, unfold_config
from flash_ansr.eval.token_prediction import correct_token_predictions_at_k, reciprocal_rank
from flash_ansr.train.scheduler import LRSchedulerFactory

import wandb

torch.set_float32_matmul_precision('high')


class OptimizerFactory():
    @staticmethod
    def get_optimizer(name: str, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        if hasattr(torch.optim, name):
            return getattr(torch.optim, name)(*args, **kwargs)
        if hasattr(torch_optimizer, name):
            return getattr(torch_optimizer, name)(*args, **kwargs)

        raise NotImplementedError(f"Optimizer {name} not found in torch.optim, torch_optimizer")


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([example[k] for example in batch]) for k in batch[0]}


class Trainer():
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
        self.device = torch.device("cpu")  # Will be set in run method

        # Metrics and Loss Functions
        self.metrics_ignore_index = self.model.tokenizer["<pad>"]
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.metrics_ignore_index)

        self.total_pflops = 0.0
        self.encoder_parameters = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        self.decoder_parameters = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Trainer":
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

        print(f'Loading lr_scheduler with config {config_["lr_scheduler"]}')
        lr_scheduler = LRSchedulerFactory.get_scheduler(config_['lr_scheduler']['name'], optimizer, **config_['lr_scheduler'].get('kwargs', {})) if config_["lr_scheduler"] else None

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
        """Sets up the model, device, and initial training state."""
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
        """Core training loop."""

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

        if verbose:
            print(f"Training model ({self.model.n_params:,} parameters) for {steps:,} steps on device {device}")

        wandb_config = unfold_config(copy.deepcopy(self.config))
        wandb_config.update({"steps": steps, "device": device, "verbose": verbose})

        with wandb.init(config=wandb_config, project=project_name, entity=entity, name=name, mode=wandb_mode):  # type: ignore
            if wandb_mode != 'disabled':
                wandb.watch(self.model, log=wandb_watch_log, log_freq=wandb_watch_log_freq)  # type: ignore
            if verbose:
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
        self.total_pflops += 6 * (self.encoder_parameters * encoder_tokens + self.decoder_parameters * decoder_tokens) * batch_size * 1e-15

    def _train_step(self, batch: dict[str, torch.Tensor], step: int, preprocess: bool, do_optimizer_step: bool = True) -> None:
        """Performs a single training step, including gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad()

        total_ce_loss = 0.0
        total_loss = 0.0

        for acc_step in range(self.gradient_accumulation_steps):
            micro_batch_size = len(batch['x_tensors']) // self.gradient_accumulation_steps
            micro_batch = {k: v[acc_step * micro_batch_size:(acc_step + 1) * micro_batch_size] for k, v in batch.items()}
            micro_batch = self.train_dataset.collate(micro_batch, device=self.device)

            data_tensor = torch.cat([micro_batch['x_tensors'], micro_batch['y_tensors']], dim=-1)

            # Pad the data tensor to the max set size (B, M, D) -> (B, max_set_size, D)
            if data_tensor.shape[1] < self.max_set_size:
                pad_size = self.max_set_size - data_tensor.shape[1]
                pad_tensor = torch.zeros((data_tensor.shape[0], pad_size, data_tensor.shape[2]), device=data_tensor.device, dtype=data_tensor.dtype)
                data_tensor = torch.cat([data_tensor, pad_tensor], dim=1)
                data_attn_mask = torch.cat([
                    torch.ones((data_tensor.shape[0], data_tensor.shape[1] - pad_size), device=data_tensor.device, dtype=torch.bool),
                    torch.zeros((data_tensor.shape[0], pad_size), device=data_tensor.device, dtype=torch.bool)], dim=1)
            else:
                data_attn_mask = torch.ones((data_tensor.shape[0], data_tensor.shape[1]), device=data_tensor.device, dtype=torch.bool)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits = self.model(micro_batch['input_ids'], data_tensor, input_num=micro_batch.get('input_num', None), data_attn_mask=data_attn_mask)
                flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                flat_labels = micro_batch['labels'].reshape(-1)
                ce_loss = self.cross_entropy_loss(flat_logits, flat_labels)

                param_sum = sum(p.sum() for p in self.model.parameters())   # HACK: Avoids None parameter gradients for wandb.watch
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

            # Log metrics and update scheduler
            self._log_metrics(step, flat_logits, flat_labels, total_ce_loss, total_loss, total_gradient_norm)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _validate_step(self, step: int, size: int | None, batch_size: int, preprocess: bool, verbose: bool) -> None:
        """Performs a single validation step."""
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

            pbar = tqdm(total=steps, leave=False, position=1, disable=not verbose, desc="Validating")
            for batch in self.val_dataset.iterate(size=size, batch_size=batch_size, preprocess=preprocess):
                batch = self.val_dataset.collate(batch, device=self.device)
                data_tensor = torch.cat([batch['x_tensors'], batch['y_tensors']], dim=-1)

                # Pad the data tensor to the max set size (B, M, D) -> (B, max_set_size, D)
                if data_tensor.shape[1] < self.max_set_size:
                    pad_size = self.max_set_size - data_tensor.shape[1]
                    pad_tensor = torch.zeros((data_tensor.shape[0], pad_size, data_tensor.shape[2]), device=data_tensor.device, dtype=data_tensor.dtype)
                    data_tensor = torch.cat([data_tensor, pad_tensor], dim=1)
                    data_attn_mask = torch.cat([
                        torch.ones((data_tensor.shape[0], data_tensor.shape[1] - pad_size), device=data_tensor.device, dtype=torch.bool),
                        torch.zeros((data_tensor.shape[0], pad_size), device=data_tensor.device, dtype=torch.bool)], dim=1)
                else:
                    data_attn_mask = torch.ones((data_tensor.shape[0], data_tensor.shape[1]), device=data_tensor.device, dtype=torch.bool)

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits = self.model(batch['input_ids'], data_tensor, input_num=batch.get('input_num', None), data_attn_mask=data_attn_mask)
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
        """Saves the model and training configuration."""
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
        """Logs a batch's training metrics to wandb."""

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
        """Logs validation metrics to wandb."""
        log_data = {
            "val_ce_loss": val_ce_loss,
            "val_mean_reciprocal_rank": val_mrr,
            "val_correct_token_predictions_at_1": val_acc_at_1,
        }
        wandb.log(log_data, step=step)  # type: ignore
