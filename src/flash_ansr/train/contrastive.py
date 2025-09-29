import os
import copy
from typing import Any, Literal

import torch
from torch.optim.lr_scheduler import LRScheduler

from tqdm import tqdm

from flash_ansr.model.contrastive_encoder import ContrastiveEncoder
from flash_ansr.data import FlashANSRDataset
from flash_ansr.utils import load_config, save_config, substitute_root_path, unfold_config
from flash_ansr.train.utils import OptimizerFactory
from flash_ansr.train.loss import ContrastiveLoss
from flash_ansr.train.scheduler import LRSchedulerFactory

import wandb

torch.set_float32_matmul_precision('high')


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([example[k] for example in batch]) for k in batch[0]}


def schedule(points: list[tuple[int, float]], step: int) -> float:
    """Schedules a value based on piecewise linear interpolation between given points."""
    if step <= points[0][0]:
        return points[0][1]
    for i in range(1, len(points)):
        if step <= points[i][0]:
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            return y0 + (y1 - y0) * (step - x0) / (x1 - x0)
    return points[-1][1]


class ContrastiveTrainer():
    def __init__(
            self,
            contrastive_encoder: ContrastiveEncoder,
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
        self.contrastive_encoder = contrastive_encoder
        self.optimizer = optimizer
        self.amp_dtype = amp_dtype
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cpu")  # Will be set in run method

        self.contrastive_config = {
            'temperature': self.config.get('contrastive_temperature', [(0, 0.5)]),
            'views': self.config.get('contrastive_views', 8),
        }
        self.total_steps = self.config.get("steps", 1)

        # Metrics and Loss Functions
        self.contrastive_loss = ContrastiveLoss()  # Use defaults and schedule the margin/temperature during training

        self.total_pflops = 0.0
        self.parameters = sum(p.numel() for p in self.contrastive_encoder.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "ContrastiveTrainer":
        config_ = load_config(config)

        if "trainer" in config_.keys():
            config_ = config_["trainer"]

        # Handle relative model paths
        if isinstance(config, str) and isinstance(config_["contrastive_encoder"], str) and config_["contrastive_encoder"].startswith('.'):
            config_["contrastive_encoder"] = os.path.join(os.path.dirname(config), config_["contrastive_encoder"])

        print(f'Creating ContrastiveEncoder from {config_["contrastive_encoder"]}')
        contrastive_encoder = ContrastiveEncoder.from_config(config_["contrastive_encoder"])

        print(f'Loading optimizer with config {config_["optimizer"]}')
        optimizer = OptimizerFactory.get_optimizer(config_['optimizer']['name'], params=contrastive_encoder.parameters(), **config_['optimizer'].get('kwargs', {}))

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
            contrastive_encoder=contrastive_encoder,
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
        self.contrastive_encoder.to(self.device)

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
            verbose: bool = False) -> ContrastiveEncoder:
        """Core training loop."""

        try:
            if compile_mode is not None:
                print(f"Compiling contrastive encoder model with mode: {compile_mode}")
                self.contrastive_encoder = torch.compile(self.contrastive_encoder, mode=compile_mode)

            self._setup_training_state(device, verbose=verbose)

            pbar = tqdm(range(steps), disable=not verbose, smoothing=0, desc="Training")
            for step, batch in enumerate(self.train_dataset.iterate(steps=steps, batch_size=self.batch_size, preprocess=False, n_per_equation=self.contrastive_config['views'])):
                self._train_step(batch, step, do_optimizer_step=True)
                pbar.update(1)

                if validate_interval is not None and ((step + 1) % validate_interval == 0 or step == steps - 1):
                    self._validate_step(step + 1, validate_size, validate_batch_size, verbose)

                if checkpoint_interval is not None and checkpoint_directory is not None and (step + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(step + 1, checkpoint_directory)

            pbar.close()
            return self.contrastive_encoder

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
            verbose: bool = False) -> ContrastiveEncoder:

        if verbose:
            print(f"Training ContrastiveEncoder ({self.contrastive_encoder.n_params:,} parameters) for {steps:,} steps on device {device}")

        wandb_config = unfold_config(copy.deepcopy(self.config))
        wandb_config.update({"steps": steps, "device": device, "verbose": verbose})

        with wandb.init(config=wandb_config, project=project_name, entity=entity, name=name, mode=wandb_mode):  # type: ignore
            if wandb_mode != 'disabled':
                wandb.watch(self.contrastive_encoder, log=wandb_watch_log, log_freq=wandb_watch_log_freq)  # type: ignore
                if verbose and wandb_watch_log is not None:
                    print(f'Watching ContrastiveEncoder with wandb log={wandb_watch_log} at frequency {wandb_watch_log_freq}')

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
        self.contrastive_encoder.train()

        # Schedule temperature
        self.contrastive_loss.temperature = schedule(self.contrastive_config['temperature'], step)

        total_contrastive_loss = 0.0

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
                # features shape is (B, S, D) where S is the number of queries
                features: torch.Tensor = self.contrastive_encoder(data_tensor, data_attn_mask=data_attn_mask)

                # ### START MODIFICATION ###
                # Implement the token-to-token contrastive loss strategy.

                # Get the shape components
                B, S, D = features.shape
                labels = micro_batch['expression_ids']

                # 1. Reshape features from (B, S, D) to (S * B, D)
                # We permute to group all 0-th tokens, all 1st tokens, etc. together.
                features_reshaped = features.permute(1, 0, 2).reshape(S * B, D)

                # 2. Expand labels from (B,) to (S * B,) to match the reshaped features.
                # [l_1, l_2] -> [[l_1, l_2], [l_1, l_2], ...] (S times) -> [l_1, l_2, l_1, l_2, ...]
                labels_reshaped = labels.unsqueeze(0).expand(S, -1).reshape(S * B)

                # 3. Compute loss on the reshaped tensors
                contrastive_loss = self.contrastive_loss(features_reshaped, labels_reshaped)
                # ### END MODIFICATION ###

                # HACK: Avoids None parameter gradients for wandb.watch
                param_sum = sum(p.sum() for p in self.contrastive_encoder.parameters() if p.grad is not None or p.requires_grad)

                loss = contrastive_loss / self.gradient_accumulation_steps + (0.0 * param_sum)

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is {loss.item()}, stopping training")

            self.scaler.scale(loss).backward()
            total_contrastive_loss += contrastive_loss.item()

        # After accumulating gradients, perform optimizer step
        self.scaler.unscale_(self.optimizer)
        total_gradient_norm = torch.nn.utils.clip_grad_norm_(self.contrastive_encoder.parameters(), 2.0)

        if do_optimizer_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._update_total_pflops(encoder_tokens=data_tensor.shape[1], batch_size=len(batch['x_tensors']))

            # Log metrics and update scheduler
            self._log_metrics(step, total_contrastive_loss, total_contrastive_loss, total_gradient_norm)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _validate_step(self, step: int, size: int | None, batch_size: int, verbose: bool) -> None:
        self.contrastive_encoder.eval()

        # Schedule temperature
        self.contrastive_loss.temperature = schedule(self.contrastive_config['temperature'], step)

        total_contrastive_loss = 0.0

        with torch.no_grad():
            steps = (size or len(self.val_dataset)) // batch_size
            pbar = tqdm(total=steps, leave=False, position=1, disable=not verbose, desc="Validating")

            for batch in self.val_dataset.iterate(size=size, batch_size=batch_size, preprocess=False, n_per_equation=self.contrastive_config['views']):
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
                    # features shape is (B, S, D)
                    features: torch.Tensor = self.contrastive_encoder(data_tensor, data_attn_mask=data_attn_mask)

                    # ### START MODIFICATION ###
                    # Implement the token-to-token contrastive loss strategy.

                    # Get the shape components
                    B, S, D = features.shape
                    labels = batch['expression_ids']

                    # 1. Reshape features from (B, S, D) to (S * B, D)
                    features_reshaped = features.permute(1, 0, 2).reshape(S * B, D)

                    # 2. Expand labels from (B,) to (S * B,)
                    labels_reshaped = labels.unsqueeze(0).expand(S, -1).reshape(S * B)

                    # 3. Compute loss on the reshaped tensors
                    contrastive_loss = self.contrastive_loss(features_reshaped, labels_reshaped)
                    # ### END MODIFICATION ###

                total_contrastive_loss += contrastive_loss.item()
                pbar.update(1)
            pbar.close()

        avg_contrastive_loss = total_contrastive_loss / (steps or 1)

        self._log_validation_metrics(step, avg_contrastive_loss)

    def _save_checkpoint(self, step: int, checkpoint_directory: str) -> None:
        """Saves the model and training configuration."""
        save_directory = os.path.join(checkpoint_directory, f"checkpoint_{step}")

        self.contrastive_encoder.save(directory=save_directory, errors='ignore')
        torch.save(self.optimizer.state_dict(), os.path.join(save_directory, "optimizer.pt"))
        save_config(
            load_config(self.config, resolve_paths=True),
            directory=save_directory,
            filename='train.yaml',
            reference='relative',
            recursive=True,
            resolve_paths=True)
        print(f"Checkpoint saved at {save_directory}")

    def _log_metrics(self, step: int, ce_loss: float, total_loss: float, total_gradient_norm: torch.Tensor) -> None:
        """Logs a batch's training metrics to wandb."""
        log_data = {
            "total_gradient_norm": total_gradient_norm.item(),
            "train_contrastive_loss": ce_loss,
            "train_loss": total_loss,
            "lr": self.optimizer.param_groups[0]['lr'],
            "total_pflops": self.total_pflops,
            "contrastive_temperature": self.contrastive_loss.temperature,
        }
        wandb.log(log_data, step=step)  # type: ignore

    def _log_validation_metrics(self, step: int, val_ce_loss: float) -> None:
        """Logs validation metrics to wandb."""
        log_data = {
            "val_ce_loss": val_ce_loss,
        }
        wandb.log(log_data, step=step)  # type: ignore
