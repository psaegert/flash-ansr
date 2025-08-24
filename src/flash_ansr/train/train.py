import os
import copy
from typing import Any, Literal

import torch
import torch_optimizer
import schedulefree
import numpy as np

from torch.optim.lr_scheduler import LRScheduler
from torch import nn

from tqdm import tqdm

from flash_ansr.model import FlashANSRModel
from flash_ansr.data import FlashANSRDataset
from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.eval.token_prediction import correct_token_predictions_at_k, reciprocal_rank
from flash_ansr.train.scheduler import LRSchedulerFactory
from flash_ansr.train.loss import ContrastiveLoss

import wandb


class OptimizerFactory():
    @staticmethod
    def get_optimizer(name: str, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        if hasattr(torch.optim, name):
            return getattr(torch.optim, name)(*args, **kwargs)
        if hasattr(torch_optimizer, name):
            return getattr(torch_optimizer, name)(*args, **kwargs)
        if hasattr(schedulefree, name):
            return getattr(schedulefree, name)(*args, **kwargs)

        raise NotImplementedError(f"Optimizer {name} not found in torch.optim, torch_optimizer, or schedulefree")


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
        self.contrastive_loss_fn = ContrastiveLoss()

        # Training state
        self.cumulative_training_tokens = 0
        self.cumulative_training_pflops = 0.0
        self.cumulative_parameter_distance = 0.0
        self.last_gradients_list: torch.Tensor | None = None
        self.last_parameters_list: torch.Tensor | None = None
        self.initial_parameters_list: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Trainer":
        config_ = load_config(config)

        if "trainer" in config_.keys():
            config_ = config_["trainer"]

        # Handle relative model paths
        if isinstance(config, str) and isinstance(config_["model"], str) and config_["model"].startswith('.'):
            config_["model"] = os.path.join(os.path.dirname(config), config_["model"])

        print(f'Loading model from {config_["model"]}')
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

    def _setup_training_state(self, device: str) -> None:
        """Sets up the model, device, and initial training state."""
        self.device = torch.device(device)
        self.model.to(self.device)
        self.flops_per_token = self.model.n_params * 6
        self.cumulative_training_pflops = 0.0
        self.cumulative_training_tokens = 0
        self.last_parameters_list = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        self.initial_parameters_list = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        self.cumulative_parameter_distance = 0.0
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

    def run_training(
            self,
            steps: int,
            preprocess: bool = False,
            device: str = "cpu",
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            validate_size: int | None = None,
            validate_batch_size: int = 128,
            verbose: bool = False) -> FlashANSRModel:
        """Core training loop."""
        self._setup_training_state(device)

        # Initial validation
        if validate_interval is not None:
            self._validate_step(0, validate_size, validate_batch_size, preprocess, verbose)

        pbar = tqdm(range(steps), disable=not verbose, smoothing=0, desc="Training")
        for step, batch in enumerate(self.train_dataset.iterate(steps=steps, batch_size=self.batch_size, preprocess=preprocess)):
            self._train_step(batch, step, preprocess)

            pbar.update(1)

            if validate_interval is not None and ((step + 1) % validate_interval == 0 or step == steps - 1):
                self._validate_step(step + 1, validate_size, validate_batch_size, preprocess, verbose)

            if checkpoint_interval is not None and checkpoint_directory is not None and (step + 1) % checkpoint_interval == 0:
                self._save_checkpoint(step + 1, checkpoint_directory)

        pbar.close()
        return self.model

    def run(
            self,
            project_name: str,
            entity: str,
            name: str,
            steps: int,
            preprocess: bool = False,
            device: str = "cpu",
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            validate_size: int | None = None,
            validate_batch_size: int = 128,
            wandb_mode: Literal['online', 'offline', 'disabled'] = 'online',
            verbose: bool = False) -> FlashANSRModel:

        if verbose:
            print(f"Training model ({self.model.n_params:,} parameters) for {steps:,} steps on device {device}")

        wandb_config = copy.deepcopy(self.config)
        wandb_config.update({"steps": steps, "device": device, "verbose": verbose})

        with wandb.init(config=wandb_config, project=project_name, entity=entity, name=name, mode=wandb_mode):  # type: ignore
            return self.run_training(
                steps=steps,
                preprocess=preprocess,
                device=device,
                checkpoint_interval=checkpoint_interval,
                checkpoint_directory=checkpoint_directory,
                validate_interval=validate_interval,
                validate_size=validate_size,
                validate_batch_size=validate_batch_size,
                verbose=verbose
            )

    def _train_step(self, batch: dict[str, torch.Tensor], step: int, preprocess: bool) -> None:
        """Performs a single training step, including gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_ce_loss = 0.0
        total_loss = 0.0

        for acc_step in range(self.gradient_accumulation_steps):
            micro_batch_size = len(batch['x_tensors']) // self.gradient_accumulation_steps
            micro_batch = {k: v[acc_step * micro_batch_size:(acc_step + 1) * micro_batch_size] for k, v in batch.items()}
            micro_batch = self.train_dataset.collate(micro_batch, device=self.device)

            data_tensor = torch.cat([micro_batch['x_tensors'], micro_batch['y_tensors']], dim=-1)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits = self.model.forward(micro_batch['input_ids'], data_tensor, input_num=micro_batch.get('input_num', None))
                flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                flat_labels = micro_batch['labels'].reshape(-1)
                ce_loss = self.cross_entropy_loss(flat_logits, flat_labels)
                loss = ce_loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            total_ce_loss += ce_loss.item()
            total_loss += loss.item() * self.gradient_accumulation_steps

            # Update cumulative metrics
            tokens = logits.shape[1] * micro_batch['x_tensors'].shape[0]
            self.cumulative_training_tokens += tokens
            self.cumulative_training_pflops += (self.flops_per_token * tokens) * 1e-15

        # Perform the optimizer step
        self.scaler.unscale_(self.optimizer)
        gradient_norms = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Log metrics and update scheduler
        self._log_metrics(step, flat_logits, flat_labels, total_ce_loss, total_loss, gradient_norms)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _validate_step(self, step: int, size: int | None, batch_size: int, preprocess: bool, verbose: bool) -> None:
        """Performs a single validation step."""
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        val_ce_loss = 0.0
        total_items = 0
        final_flat_logits_list: list[torch.Tensor] = []
        final_flat_labels_list: list[torch.Tensor] = []

        with torch.no_grad():
            if size is None:
                steps = len(self.val_dataset) * batch_size // batch_size
            else:
                steps = size // batch_size

            pbar = tqdm(total=steps, leave=False, position=1, disable=not verbose, desc="Validating")
            for batch in self.val_dataset.iterate(size=size, batch_size=batch_size, preprocess=preprocess):
                batch = self.val_dataset.collate(batch, device=self.device)
                data_tensor = torch.cat([batch['x_tensors'], batch['y_tensors']], dim=-1)

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits = self.model.forward(batch['input_ids'], data_tensor, input_num=batch.get('input_num', None))
                    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                    flat_labels = batch['labels'].reshape(-1)
                    ce_loss = self.cross_entropy_loss(flat_logits, flat_labels)

                val_ce_loss += ce_loss.item() * flat_labels.shape[0]
                total_items += flat_labels.shape[0]

                final_flat_logits_list.append(flat_logits)
                final_flat_labels_list.append(flat_labels)
                pbar.update(1)
            pbar.close()

        if total_items > 0:
            val_ce_loss /= total_items

        final_flat_logits = torch.cat(final_flat_logits_list, dim=0)
        final_flat_labels = torch.cat(final_flat_labels_list, dim=0)

        # Log validation metrics
        self._log_validation_metrics(step, final_flat_logits, final_flat_labels, val_ce_loss)

    def _save_checkpoint(self, step: int, checkpoint_directory: str) -> None:
        """Saves the model and training configuration."""
        save_directory = os.path.join(checkpoint_directory, f"checkpoint_{step}")
        self.model.save(directory=save_directory, errors='ignore')
        save_config(
            load_config(self.config, resolve_paths=True),
            directory=save_directory,
            filename='train.yaml',
            reference='relative',
            recursive=True,
            resolve_paths=True)
        print(f"Checkpoint saved at {save_directory}")

    def _log_metrics(self, step: int, logits: torch.Tensor, labels: torch.Tensor, ce_loss: float, total_loss: float, gradient_norms: torch.Tensor) -> None:
        """Logs a batch's training metrics to wandb."""
        new_gradients_list = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        if self.last_gradients_list is None:
            gradient_cosine_similarity = np.nan
        else:
            gradient_cosine_similarity = torch.nn.functional.cosine_similarity(self.last_gradients_list, new_gradients_list, dim=0).item()
        self.last_gradients_list = new_gradients_list

        new_parameters_list = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        self.cumulative_parameter_distance += (new_parameters_list - self.last_parameters_list).norm().item()
        self.last_parameters_list = new_parameters_list

        log_data = {
            "gradient_cosine_similarity": gradient_cosine_similarity,
            "max_gradient_norm": torch.max(gradient_norms).item(),
            "total_parameter_distance": self.cumulative_parameter_distance,
            "effective_parameter_distance": (new_parameters_list - self.initial_parameters_list).norm().item(),
            "train_ce_loss": ce_loss,
            "train_loss": total_loss,
            "lr": self.optimizer.param_groups[0]['lr'],
            "cumulative_training_tokens": self.cumulative_training_tokens,
            "cumulative_training_pflops": self.cumulative_training_pflops,
            "train_mean_reciprocal_rank": reciprocal_rank(logits, labels, ignore_index=self.metrics_ignore_index),
            "train_correct_token_predictions_at_1": correct_token_predictions_at_k(logits, labels, k=1, ignore_index=self.metrics_ignore_index),
            "train_correct_token_predictions_at_5": correct_token_predictions_at_k(logits, labels, k=5, ignore_index=self.metrics_ignore_index),
        }
        wandb.log(log_data, step=step)  # type: ignore

    def _log_validation_metrics(self, step: int, logits: torch.Tensor, labels: torch.Tensor, val_ce_loss: float) -> None:
        """Logs validation metrics to wandb."""
        log_data = {
            "val_ce_loss": val_ce_loss,
            "val_mean_reciprocal_rank": reciprocal_rank(logits, labels, ignore_index=self.metrics_ignore_index),
            "val_correct_token_predictions_at_1": correct_token_predictions_at_k(logits, labels, k=1, ignore_index=self.metrics_ignore_index),
            "val_correct_token_predictions_at_5": correct_token_predictions_at_k(logits, labels, k=5, ignore_index=self.metrics_ignore_index),
            "val_correct_token_predictions_at_10": correct_token_predictions_at_k(logits, labels, k=10, ignore_index=self.metrics_ignore_index),
        }
        wandb.log(log_data, step=step)  # type: ignore
