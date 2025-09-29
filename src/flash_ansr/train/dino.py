import os
import copy
import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch import nn

from tqdm import tqdm

from flash_ansr.model.dino import DINOEncoder
from flash_ansr.data import FlashANSRDataset
from flash_ansr.utils import load_config, save_config, substitute_root_path, unfold_config
from flash_ansr.train.utils import OptimizerFactory
from flash_ansr.train.scheduler import LRSchedulerFactory

import wandb

torch.set_float32_matmul_precision('high')


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([example[k] for example in batch]) for k in batch[0]}


class DINOTrainer():
    def __init__(
            self,
            dino_encoder: DINOEncoder,
            optimizer: torch.optim.Optimizer,
            amp_dtype: torch.dtype,
            scaler: torch.amp.GradScaler,
            lr_scheduler: LRScheduler | None,
            batch_size: int,
            train_dataset: FlashANSRDataset,
            val_dataset: FlashANSRDataset,
            gradient_accumulation_steps: int = 1,
            config: dict[str, Any] = None) -> None:

        self.config = config or {}
        self.student = dino_encoder
        self.optimizer = optimizer
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cpu")  # Will be set in run method

        # DINO specific parameters from config
        self.dino_config = {
            'weight_decay': self.config.get('dino_weight_decay', {}),
            'student_temp': self.config.get('dino_temperature_student', 0.1),
            'teacher_temp': self.config.get('dino_temperature_teacher', {}),
            'teacher_lambda': self.config.get('dino_teacher_lambda', {}),
            'center_momentum': self.config.get('dino_center_momentum', 0.9),
            'views': self.config.get('dino_views', 8),
        }
        self.total_steps = self.config.get("steps", 1)

        # Create EMA teacher model
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # DINO head output dimension for centering
        dino_head_output_dim = self.student.dino_head[-1].out_features
        self.center = torch.zeros(1, 1, dino_head_output_dim)

        # Metrics and Loss Functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.total_pflops = 0.0
        self.parameters = sum(p.numel() for p in self.student.parameters() if p.requires_grad)

    @staticmethod
    def _cosine_scheduler(current_step: int, total_steps: int, start_val: float, end_val: float) -> float:
        """Cosine schedule from start_val to end_val."""
        if current_step < total_steps:
            return end_val - (end_val - start_val) * (1 + math.cos(math.pi * current_step / total_steps)) / 2
        return end_val

    @staticmethod
    def _linear_scheduler(current_step: int, total_steps: int, start_val: float, end_val: float) -> float:
        """Linear schedule from start_val to end_val."""
        if current_step < total_steps:
            return start_val + (end_val - start_val) * current_step / total_steps
        return end_val

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "DINOTrainer":
        config_ = load_config(config)

        if "trainer" in config_.keys():
            config_ = config_["trainer"]

        # Handle relative model paths
        if isinstance(config, str) and isinstance(config_["dino_encoder"], str) and config_["dino_encoder"].startswith('.'):
            config_["dino_encoder"] = os.path.join(os.path.dirname(config), config_["dino_encoder"])

        print(f'Creating DINOEncoder from {config_["dino_encoder"]}')
        dino_encoder = DINOEncoder.from_config(config_["dino_encoder"])

        print(f'Loading optimizer with config {config_["optimizer"]}')
        optimizer = OptimizerFactory.get_optimizer(config_['optimizer']['name'], params=dino_encoder.parameters(), **config_['optimizer'].get('kwargs', {}))

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
            dino_encoder=dino_encoder,
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
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.center = self.center.to(self.device)

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
            device: str = "cpu",
            compile_mode: str | None = None,
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            validate_size: int | None = None,
            validate_batch_size: int = 128,
            verbose: bool = False) -> DINOEncoder:
        """Core training loop."""

        try:
            if compile_mode is not None:
                print(f"Compiling student model with mode: {compile_mode}")
                self.student = torch.compile(self.student, mode=compile_mode)
                # Teacher is not compiled as it's only for inference

            self._setup_training_state(device, verbose=verbose)

            pbar = tqdm(range(steps), disable=not verbose, smoothing=0, desc="Training")
            for step, batch in enumerate(self.train_dataset.iterate(steps=steps, batch_size=self.batch_size, preprocess=False, n_per_equation=self.dino_config['views'])):
                self._train_step(batch, step, do_optimizer_step=True)
                pbar.update(1)

                if validate_interval is not None and ((step + 1) % validate_interval == 0 or step == steps - 1):
                    self._validate_step(step + 1, validate_size, validate_batch_size, verbose)

                if checkpoint_interval is not None and checkpoint_directory is not None and (step + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(step + 1, checkpoint_directory)

            pbar.close()
            # Return the unwrapped student model if compiled
            return self.student._orig_mod if hasattr(self.student, '_orig_mod') else self.student

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
            verbose: bool = False) -> DINOEncoder:

        if verbose:
            print(f"Training DINOEncoder ({self.student.n_params:,} parameters) for {steps:,} steps on device {device}")

        wandb_config = unfold_config(copy.deepcopy(self.config))
        wandb_config.update({"steps": steps, "device": device, "verbose": verbose})

        with wandb.init(config=wandb_config, project=project_name, entity=entity, name=name, mode=wandb_mode):  # type: ignore
            if wandb_mode != 'disabled':
                wandb.watch(self.student, log=wandb_watch_log, log_freq=wandb_watch_log_freq)  # type: ignore
                if verbose and wandb_watch_log is not None:
                    print(f'Watching DINOEncoder with wandb log={wandb_watch_log} at frequency {wandb_watch_log_freq}')

            return self.run_training(
                steps=steps,
                device=device,
                compile_mode=compile_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_directory=checkpoint_directory,
                validate_interval=validate_interval,
                validate_size=validate_size,
                validate_batch_size=validate_batch_size,
                verbose=verbose
            )

    def _update_total_pflops(self, encoder_tokens: int, batch_size: int) -> None:
        self.total_pflops += 6 * (self.parameters * encoder_tokens) * batch_size * 1e-15

    def _train_step(self, batch: dict[str, torch.Tensor], step: int, do_optimizer_step: bool = True) -> None:
        """Performs a single training step, including DINO loss, EMA updates, and gradient accumulation."""
        self.student.train()

        # Schedule weight decay
        wd_config = self.dino_config['weight_decay']
        wd = self._cosine_scheduler(step, self.total_steps, wd_config['initial'], wd_config['final'])
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = wd

        # Schedule teacher temperature
        teacher_temp_config = self.dino_config['teacher_temp']
        teacher_temp = self._linear_scheduler(step, self.total_steps, teacher_temp_config['initial'], teacher_temp_config['final'])
        student_temp = self.dino_config['student_temp']

        total_dino_loss = 0.0

        # We zero grads once before the accumulation loop
        self.optimizer.zero_grad()

        for acc_step in range(self.gradient_accumulation_steps):
            micro_batch_size = len(batch['x_tensors']) // self.gradient_accumulation_steps
            start_idx = acc_step * micro_batch_size
            end_idx = (acc_step + 1) * micro_batch_size
            micro_batch = {k: v[start_idx:end_idx] for k, v in batch.items()}
            micro_batch = self.train_dataset.collate(micro_batch, device=self.device)

            data_tensor = torch.cat([micro_batch['x_tensors'], micro_batch['y_tensors']], dim=-1)

            # Pad the data tensor to the max set size (B, M, D) -> (B, max_set_size, D)
            B, M, D = data_tensor.shape
            if M < self.max_set_size:
                pad_size = self.max_set_size - M
                pad_tensor = torch.zeros((B, pad_size, D), device=data_tensor.device, dtype=data_tensor.dtype)
                data_tensor = torch.cat([data_tensor, pad_tensor], dim=1)
                data_attn_mask = torch.cat([
                    torch.ones((B, M), device=data_tensor.device, dtype=torch.bool),
                    torch.zeros((B, pad_size), device=data_tensor.device, dtype=torch.bool)], dim=1)
            else:
                data_attn_mask = torch.ones((B, M), device=data_tensor.device, dtype=torch.bool)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                student_logits: torch.Tensor = self.student(data_tensor, data_attn_mask=data_attn_mask)

                with torch.no_grad():
                    teacher_logits: torch.Tensor = self.teacher(data_tensor, data_attn_mask=data_attn_mask)

                # --- START: Vectorized Multi-View Loss Calculation ---
                B, S, K = student_logits.shape

                # 1. Generate integer labels from input_ids to identify equation groups
                # `labels` will be a tensor of shape (B,) where samples from the same equation have the same integer label.
                _, labels = torch.unique(micro_batch['input_ids'], dim=0, return_inverse=True)

                # 2. Create masks
                # `positive_mask` [B, B]: True if sample i and j are from the same equation
                positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
                # `identity_mask` [B, B]: True for diagonal elements (i == j)
                identity_mask = torch.eye(B, device=labels.device, dtype=torch.bool)
                # `valid_pair_mask`: True for pairs from the same equation but are not the same view
                valid_pair_mask = positive_mask & ~identity_mask

                # 3. Calculate student log-probabilities and teacher probabilities
                student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
                teacher_probs = F.softmax((teacher_logits - self.center) / teacher_temp, dim=-1)

                # 4. Compute pairwise cross-entropy loss matrix [B, B]
                # The 'bsk,csk->bc' einsum computes the dot product for each student view 'b'
                # against each teacher view 'c', summed over sequence 's' and feature 'k' dimensions.
                # This efficiently calculates the cross-entropy for all B*B pairs.
                pairwise_loss_matrix = -torch.einsum('bsk,csk->bc', student_log_probs, teacher_probs) / S

                # 5. Select valid pairs using the mask and compute the final mean loss
                # This averages the loss only over the valid (i, j) pairs where i!=j
                # and i and j are views of the same equation.
                dino_loss = pairwise_loss_matrix[valid_pair_mask].mean()
                # --- END: Vectorized Multi-View Loss Calculation ---

                # HACK: Avoids None parameter gradients for wandb.watch
                param_sum = sum(p.sum() for p in self.student.parameters() if p.grad is not None or p.requires_grad)

                loss = dino_loss / self.gradient_accumulation_steps + (0.0 * param_sum)

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is {loss.item()}, stopping training")

            self.scaler.scale(loss).backward()
            total_dino_loss += dino_loss.item()

            # Update center with momentum using the detached teacher outputs
            with torch.no_grad():
                batch_center = torch.mean(teacher_logits.detach(), dim=[0, 1], keepdim=True)
                self.center = self.center * self.dino_config['center_momentum'] + batch_center * (1 - self.dino_config['center_momentum'])

        # After accumulating gradients, perform optimizer step
        self.scaler.unscale_(self.optimizer)
        total_gradient_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 2.0)

        if do_optimizer_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update for the teacher model
            teacher_lambda_config = self.dino_config['teacher_lambda']
            teacher_lambda = self._cosine_scheduler(step, self.total_steps, teacher_lambda_config['initial'], teacher_lambda_config['final'])
            with torch.no_grad():
                for student_p, teacher_p in zip(self.student.parameters(), self.teacher.parameters()):
                    teacher_p.data.mul_(teacher_lambda).add_(student_p.data, alpha=1 - teacher_lambda)

            self._update_total_pflops(encoder_tokens=data_tensor.shape[1], batch_size=len(batch['x_tensors']))

            # Log metrics and update scheduler
            self._log_metrics(step, total_dino_loss, total_dino_loss, total_gradient_norm, wd, teacher_temp, teacher_lambda)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _validate_step(self, step: int, size: int | None, batch_size: int, verbose: bool) -> None:
        """Performs a single validation step using a standard cross-entropy loss."""
        self.student.eval()

        # Schedule teacher temperature
        teacher_temp_config = self.dino_config['teacher_temp']
        teacher_temp = self._linear_scheduler(step, self.total_steps, teacher_temp_config['initial'], teacher_temp_config['final'])
        student_temp = self.dino_config['student_temp']

        total_dino_loss = 0.0

        with torch.no_grad():
            steps = (size or len(self.val_dataset)) // batch_size
            pbar = tqdm(total=steps, leave=False, position=1, disable=not verbose, desc="Validating")

            for batch in self.val_dataset.iterate(size=size, batch_size=batch_size, preprocess=False, n_per_equation=self.dino_config['views']):
                batch = self.val_dataset.collate(batch, device=self.device)
                data_tensor = torch.cat([batch['x_tensors'], batch['y_tensors']], dim=-1)

                # Pad the data tensor to the max set size
                B, M, D = data_tensor.shape
                if M < self.max_set_size:
                    pad_size = self.max_set_size - M
                    pad_tensor = torch.zeros((B, pad_size, D), device=data_tensor.device, dtype=data_tensor.dtype)
                    data_tensor = torch.cat([data_tensor, pad_tensor], dim=1)
                    data_attn_mask = torch.cat([
                        torch.ones((B, M), device=data_tensor.device, dtype=torch.bool),
                        torch.zeros((B, pad_size), device=data_tensor.device, dtype=torch.bool)], dim=1)
                else:
                    data_attn_mask = torch.ones((B, M), device=data_tensor.device, dtype=torch.bool)

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    student_logits: torch.Tensor = self.student(data_tensor, data_attn_mask=data_attn_mask)

                    with torch.no_grad():
                        teacher_logits: torch.Tensor = self.teacher(data_tensor, data_attn_mask=data_attn_mask)

                    # --- START: Vectorized Multi-View Loss Calculation ---
                    B, S, K = student_logits.shape
                    _, labels = torch.unique(batch['input_ids'], dim=0, return_inverse=True)

                    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
                    identity_mask = torch.eye(B, device=labels.device, dtype=torch.bool)
                    valid_pair_mask = positive_mask & ~identity_mask

                    student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
                    teacher_probs = F.softmax((teacher_logits - self.center) / teacher_temp, dim=-1)

                    pairwise_loss_matrix = -torch.einsum('bsk,csk->bc', student_log_probs, teacher_probs) / S

                    dino_loss = pairwise_loss_matrix[valid_pair_mask].mean()
                    # --- END: Vectorized Multi-View Loss Calculation ---

                total_dino_loss += dino_loss.item()
                pbar.update(1)
            pbar.close()

        avg_dino_loss = total_dino_loss / (steps or 1)

        self._log_validation_metrics(step, avg_dino_loss)

    def _save_checkpoint(self, step: int, checkpoint_directory: str) -> None:
        """Saves the student model and training configuration."""
        save_directory = os.path.join(checkpoint_directory, f"checkpoint_{step}")

        # Ensure we save the unwrapped model if using torch.compile
        model_to_save = self.teacher._orig_mod if hasattr(self.teacher, '_orig_mod') else self.teacher

        model_to_save.save(directory=save_directory, errors='ignore')
        torch.save(self.optimizer.state_dict(), os.path.join(save_directory, "optimizer.pt"))
        save_config(
            load_config(self.config, resolve_paths=True),
            directory=save_directory,
            filename='train.yaml',
            reference='relative',
            recursive=True,
            resolve_paths=True)
        print(f"Checkpoint saved at {save_directory}")

    def _log_metrics(self, step: int, ce_loss: float, total_loss: float, total_gradient_norm: torch.Tensor, wd: float, teacher_temp: float, teacher_lambda: float) -> None:
        """Logs a batch's training metrics to wandb."""
        log_data = {
            "total_gradient_norm": total_gradient_norm.item(),
            "train_dino_loss": ce_loss,
            "train_loss": total_loss,
            "lr": self.optimizer.param_groups[0]['lr'],
            "total_pflops": self.total_pflops,
            "weight_decay": wd,
            "teacher_temp": teacher_temp,
            "teacher_lambda": teacher_lambda,
        }
        wandb.log(log_data, step=step)  # type: ignore

    def _log_validation_metrics(self, step: int, val_ce_loss: float) -> None:
        """Logs validation metrics to wandb."""
        log_data = {
            "val_ce_loss": val_ce_loss,
        }
        wandb.log(log_data, step=step)  # type: ignore
