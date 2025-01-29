import os
import time
import copy
from typing import Any, Literal

import torch
import torch_optimizer
import numpy as np

import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch import nn

from tqdm import tqdm

from flash_ansr.models import FlashANSRTransformer
from flash_ansr.data import FlashANSRDataset
from flash_ansr.utils import load_config, save_config, substitute_root_path
from flash_ansr.eval.token_prediction import correct_token_predictions_at_k, reciprocal_rank
from flash_ansr.train.scheduler import LRSchedulerFactory

import wandb


class OptimizerFactory():
    @staticmethod
    def get_optimizer(name: str, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        if hasattr(torch.optim, name):
            return getattr(torch.optim, name)(*args, **kwargs)

        if hasattr(torch_optimizer, name):
            return getattr(torch_optimizer, name)(*args, **kwargs)

        raise NotImplementedError(f"Optimizer {name} not found in torch.optim")


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([example[k] for example in batch]) for k in batch[0]}


class AILWQ_ContrastiveLoss(nn.Module):
    """
    https://github.com/AILWQ/Joint_Supervised_Learning_for_SR/blob/main/loss_function/ContrastiveLoss.py
    """
    def __init__(self, temperature: float = 0.5) -> None:
        super(AILWQ_ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, enc_features: torch.Tensor, labels: list | torch.Tensor) -> torch.Tensor:
        n = len(labels)
        device = enc_features.device

        # Create labels tensor
        if isinstance(labels, list):
            unique_labels_map = {label: i for i, label in enumerate(set(labels))}
            labels_tensor = torch.tensor([unique_labels_map[label] for label in labels], device=enc_features.device)
        else:
            labels_tensor = labels

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(enc_features.unsqueeze(1), enc_features.unsqueeze(0), dim=2)

        # Create masks for the same class and different class (diagonal and where labels are equal)
        mask = torch.ones_like(similarity_matrix, device=device) * (labels_tensor.expand(n, n).eq(labels_tensor.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask, device=device) - mask
        mask_diagonal = torch.ones(n, n, device=device) - torch.eye(n, n, device=device)

        # Positive term
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_diagonal
        sim = mask * similarity_matrix

        # Negative term
        no_sim = similarity_matrix - sim  # [batch_size, batch_size]
        no_sim_sum = torch.sum(no_sim, dim=1)  # [batch_size]
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T  # [batch_size, batch_size]
        sim_sum = no_sim_sum_expend

        if torch.sum(sim_sum) == 0:
            return torch.tensor(0, device=device)

        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n, device=device)

        # compute loss of a batch
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)) + 1e-5)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, temperature: float = 0.5) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss based on cosine similarities.

        Args:
            features (torch.Tensor): A tensor of shape (N, D), where N is the number of samples and D is the feature dimension.
            labels (torch.Tensor): A tensor of shape (N,) with integer class labels for each sample.
            margin (float): Margin value for dissimilar pairs. Default is 0.5.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        # Create labels tensor
        if isinstance(labels, list):
            unique_labels_map = {label: i for i, label in enumerate(set(labels))}
            labels_tensor = torch.tensor([unique_labels_map[label] for label in labels], device=features.device)
        else:
            labels_tensor = labels

        # Normalize features to have unit norm (for cosine similarity)
        features = F.normalize(features, p=2, dim=1)

        # Compute the cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Apply exponentiation with temperature scaling
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)

        # Create a mask to identify positive (same label) and negative (different label) pairs
        labels_tensor = labels_tensor.view(-1, 1)  # Reshape to (N, 1) for broadcasting
        positive_mask = labels_tensor == labels_tensor.T  # Mask for positive pairs
        negative_mask = ~positive_mask  # Mask for negative pairs

        # Row-wise normalization of similarity matrix
        row_sum = similarity_matrix.sum(dim=1, keepdim=True)  # Shape (N, 1)

        # Compute positive loss
        positive_loss = -torch.log((similarity_matrix * positive_mask).sum(dim=1) / row_sum.squeeze()).mean()

        # Compute negative loss
        negative_loss = F.relu(torch.log((similarity_matrix * negative_mask).sum(dim=1) / row_sum.squeeze()) - self.margin).mean()

        # Combine positive and negative losses
        loss = positive_loss + negative_loss

        return loss


class Trainer():
    def __init__(
            self,
            model: FlashANSRTransformer,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: LRScheduler,
            batch_size: int,
            train_dataset: FlashANSRDataset,
            val_dataset: FlashANSRDataset,
            gradient_accumulation_steps: int = 1,
            config: dict[str, Any] = None) -> None:

        self.model = model

        # https://arxiv.org/pdf/2001.08361
        self.flops_per_token = self.model.n_params * 6
        self.cumulative_training_pflops = 0.0
        self.cumulative_training_tokens = 0

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.metrics_ignore_index = self.model.expression_space.tokenizer["<pad>"]
        self.numeric_token_index = self.model.expression_space.tokenizer["<num>"]
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.metrics_ignore_index)
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.contrastive_loss_fn = ContrastiveLoss()

        self.config = config or {}

        self.last_gradients_list: torch.Tensor | None = None
        self.last_parameters_list: torch.Tensor | None = None
        self.initial_parameters_list: torch.Tensor | None = None

        self.cumulative_parameter_distance = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "Trainer":
        config_ = load_config(config)

        if "trainer" in config_.keys():
            config_ = config_["trainer"]

        # If the config is a string, convert relative paths within the config to absolute paths
        if isinstance(config, str) and isinstance(config_["model"], str):
            if config_["model"].startswith('.'):
                config_["model"] = os.path.join(os.path.dirname(config), config_["model"])

        # Support loading existing models
        print(f'Loading model from {config_["model"]}')
        model = FlashANSRTransformer.from_config(config_["model"])

        print(f'Loading optimizer with config {config_["optimizer"]}')
        optimizer = OptimizerFactory.get_optimizer(config_['optimizer']['name'], params=model.parameters(), **config_['optimizer']['kwargs'] if 'kwargs' in config_['optimizer'] else {})

        print(f'Loading lr_scheduler with config {config_["lr_scheduler"]}')
        lr_scheduler = LRSchedulerFactory.get_scheduler(config_['lr_scheduler']['name'], optimizer, **config_['lr_scheduler']['kwargs'] if 'kwargs' in config_['lr_scheduler'] else {})

        print(f'Loading train_dataset with config {config_["train_dataset"]}')
        train_dataset = FlashANSRDataset.from_config(config_["train_dataset"])

        print(f'Loading val_dataset with config {config_["val_dataset"]}')
        # If the value is a file path, load the dataset from the file
        if isinstance(config_["val_dataset"], str):
            resolved_config_path = substitute_root_path(config_["val_dataset"])
            if not os.path.exists(resolved_config_path):
                raise ValueError(f"val_dataset file {resolved_config_path} does not exist")

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
            lr_scheduler=lr_scheduler,
            batch_size=config_['batch_size'],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            gradient_accumulation_steps=config_.get("gradient_accumulation_steps", 1),
            config=config_
        )

    def run_from_config(
            self,
            project_name: str,
            entity: str,
            name: str,
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            wandb_mode: Literal['online', 'offline', 'disabled'] = 'online',
            verbose: bool = False) -> FlashANSRTransformer:
        return self.run(
            project_name=project_name,
            entity=entity,
            name=name,
            numeric_prediction_loss_weight=self.config.get('numeric_prediction_loss_weight', 0),
            contrastive_loss_weight=self.config.get('contrastive_loss_weight', 0),
            contrastive_n_per_class=self.config.get('contrastive_n_per_class', 1),
            contrastive_margin=self.config.get('contrastive_margin', 0.5),
            contrastive_temperature=self.config.get('contrastive_temperature', 0.5),
            steps=self.config["steps"],
            device=self.config["device"],
            checkpoint_interval=checkpoint_interval,
            checkpoint_directory=checkpoint_directory,
            validate_interval=validate_interval,
            validate_size=self.config.get("val_size", None),
            validate_batch_size=self.config["val_batch_size"],
            wandb_mode=wandb_mode,
            verbose=verbose
        )

    def run(
            self,
            project_name: str,
            entity: str,
            name: str,
            numeric_prediction_loss_weight: float,
            contrastive_loss_weight: float,
            contrastive_n_per_class: int,
            contrastive_margin: float,
            contrastive_temperature: float,
            steps: int,
            device: str = "cpu",
            checkpoint_interval: int | None = None,
            checkpoint_directory: str | None = None,
            validate_interval: int | None = None,
            validate_size: int | None = None,
            validate_batch_size: int = 128,
            wandb_mode: Literal['online', 'offline', 'disabled'] = 'online',
            verbose: bool = False) -> FlashANSRTransformer:
        if verbose:
            print(f"Training model ({self.model.n_params:,} parameters) for {steps:,} steps on device {device}")
        torch.cuda.empty_cache()
        wandb_config = copy.deepcopy(self.config)
        for key, value in zip(
                ["steps", "device", "verbose"],
                [steps, device, verbose]):
            if key not in self.config.keys():
                wandb_config[key] = value

        self.contrastive_loss_fn = ContrastiveLoss(margin=contrastive_margin, temperature=contrastive_temperature)

        self.model.to(device)
        self.cumulative_training_pflops = 0.0
        self.cumulative_training_tokens = 0

        self.last_parameters_list = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        self.initial_parameters_list = torch.cat([p.detach().flatten() for p in self.model.parameters()])

        with wandb.init(config=wandb_config, project=project_name, entity=entity, name=name, mode=wandb_mode):  # type: ignore
            pbar = tqdm(range(steps), disable=not verbose, smoothing=0.001, desc="Training")

            # Validate
            if validate_interval is not None:
                _ = self._validate(
                    self.val_dataset,
                    numeric_prediction_loss_weight=numeric_prediction_loss_weight,
                    contrastive_loss_weight=contrastive_loss_weight,
                    contrastive_n_per_class=contrastive_n_per_class,
                    step=0,
                    size=validate_size,
                    batch_size=validate_batch_size,
                    verbose=verbose)

            # Train the model
            for step, batch in enumerate(self.train_dataset.iterate(steps=steps, batch_size=self.batch_size, n_per_equation=contrastive_n_per_class)):
                pbar.set_description("Training")

                # Train
                _ = self._train_batch(
                    batch,
                    numeric_prediction_loss_weight=numeric_prediction_loss_weight,
                    contrastive_loss_weight=contrastive_loss_weight,
                    step=step)

                pbar.update(1)

                if validate_interval is not None and ((step + 1) % validate_interval == 0 or step == steps - 1):
                    # Validate
                    _ = self._validate(
                        self.val_dataset,
                        numeric_prediction_loss_weight=numeric_prediction_loss_weight,
                        contrastive_loss_weight=contrastive_loss_weight,
                        contrastive_n_per_class=contrastive_n_per_class,
                        step=(step + 1),
                        size=validate_size,
                        batch_size=validate_batch_size,
                        verbose=verbose)

                # Save the model
                if checkpoint_interval is not None and checkpoint_directory is not None and (step + 1) % checkpoint_interval == 0:
                    save_directory = os.path.join(checkpoint_directory, f"checkpoint_{(step + 1)}")
                    self.model.save(directory=save_directory, errors='ignore')
                    save_config(
                        load_config(self.config, resolve_paths=True),
                        directory=save_directory,
                        filename='train.yaml',
                        reference='relative',
                        recursive=True,
                        resolve_paths=True)

            pbar.close()

        return self.model

    def _train_batch(self, batch: dict[str, torch.Tensor], numeric_prediction_loss_weight: float, contrastive_loss_weight: float, step: int) -> float:
        self.model.train()

        start_time = time.time()

        self.optimizer.zero_grad()

        accumulated_ce_loss = 0.0
        accumulated_contrastive_loss = 0.0
        accumulated_num_loss = 0.0
        accumulated_loss = 0.0

        micro_batch_size = int(len(batch['x_tensors']) // self.gradient_accumulation_steps)

        for acc_step in range(self.gradient_accumulation_steps):
            micro_batch = {k: v[acc_step * micro_batch_size:(acc_step + 1) * micro_batch_size] for k, v in batch.items()}
            input_ids, x_tensor, y_tensor, labels, constants, skeleton_hashes = FlashANSRDataset.collate_batch(micro_batch, device=self.model.device)

            # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
            pad_length = self.model.encoder_max_n_variables - x_tensor.shape[2] - y_tensor.shape[2]
            if pad_length > 0:
                x_tensor = nn.functional.pad(x_tensor, (0, pad_length, 0, 0, 0, 0), value=0)

            # Concatenate x and y tensors as input to the set transformer
            data_tensor = torch.cat([x_tensor, y_tensor], dim=-1)

            # Forward pass
            logits, num_output = self.model.forward(input_ids, data_tensor, numeric_head=(numeric_prediction_loss_weight > 0))

            flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1)

            # Calculate the loss
            ce_loss: torch.Tensor = self.cross_entropy_loss(flat_logits, flat_labels)

            if contrastive_loss_weight > 0:
                # Use memory (embeddings of encoder) that the model cached during the forward pass
                contrastive_loss = self.contrastive_loss_fn(self.model.memory.reshape(self.model.memory.shape[0], -1), skeleton_hashes)

            # Compare the num_output at the positions where the labels are `<num>`
            num_loss = torch.tensor(0, device=self.model.device, dtype=torch.float32)

            if numeric_prediction_loss_weight > 0:
                n_tokens = 0.0
                for i, (lbl, const) in enumerate(zip(labels, constants)):
                    n_tokens += torch.sum(lbl != self.metrics_ignore_index).item()
                    if len(const) > 0:
                        num_loss += self.mse_loss(num_output[i, 1:, 0][lbl == self.numeric_token_index], const)

                if n_tokens > 0:
                    num_loss /= n_tokens

            loss = (ce_loss + numeric_prediction_loss_weight * num_loss + contrastive_loss_weight * contrastive_loss) / (1 + numeric_prediction_loss_weight + contrastive_loss_weight)

            accumulated_ce_loss += ce_loss.item()
            accumulated_contrastive_loss += contrastive_loss.item()
            accumulated_num_loss += num_loss.item()
            accumulated_loss += loss.item()

            loss.backward()

            self.cumulative_training_tokens += logits.shape[1] * x_tensor.shape[0]
            self.cumulative_training_pflops += (self.flops_per_token * logits.shape[1] * x_tensor.shape[0]) * 1e-15  # PFLOPS per token * number of tokens * batch size

        accumulated_ce_loss /= self.gradient_accumulation_steps
        accumulated_contrastive_loss /= self.gradient_accumulation_steps
        accumulated_num_loss /= self.gradient_accumulation_steps
        accumulated_loss /= self.gradient_accumulation_steps

        try:
            if np.isnan(loss.item()):
                print(f'{ce_loss = }, {contrastive_loss = }, {num_loss = }, {loss = }')
                print(f'{np.sum(np.isnan(x_tensor.cpu().numpy()))} NaNs in x_tensor')
                print(f'{np.sum(np.isnan(y_tensor.cpu().numpy()))} NaNs in y_tensor')
                print(f'{np.sum(np.isnan(logits.cpu().detach().numpy()))} NaNs in logits')
                raise ValueError("Loss is NaN")
        except RuntimeError:
            print(input_ids)
            print(labels)
            print(constants)
            print(f'{ce_loss = }, {contrastive_loss = }, {num_loss = }, {loss = }')
            print(flat_logits)
            print(flat_labels)
            # Any nans in xtensor
            print(f'{np.sum(np.isnan(x_tensor.cpu().numpy()))} NaNs in x_tensor')
            print(f'{np.sum(np.isnan(y_tensor.cpu().numpy()))} NaNs in y_tensor')
            raise

        gradient_norms = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()

        new_gradients_list = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])

        if self.last_gradients_list is None:
            self.last_gradients_list = new_gradients_list
            gradient_sample_cosine_similarity = np.nan
        else:
            if self.last_gradients_list.shape != new_gradients_list.shape:
                gradient_sample_cosine_similarity = np.nan
            else:
                gradient_sample_cosine_similarity = torch.nn.functional.cosine_similarity(self.last_gradients_list, new_gradients_list, dim=0).item()

            self.last_gradients_list = new_gradients_list

        new_parameters_list = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        self.cumulative_parameter_distance += (new_parameters_list - self.last_parameters_list).norm().item()
        self.last_parameters_list = new_parameters_list

        wandb.log({  # type: ignore
            "gradient_sample_cosine_similarity": gradient_sample_cosine_similarity,
            "max_gradient_norm": torch.max(gradient_norms),
            "total_parameter_distance": self.cumulative_parameter_distance,
            "effective_parameter_distance": (new_parameters_list - self.initial_parameters_list).norm().item(),
            "train_ce_loss": accumulated_ce_loss,
            "train_contrastive_loss": accumulated_contrastive_loss,
            "train_num_loss": accumulated_num_loss,
            "train_loss": accumulated_loss,
            "lr": self.optimizer.param_groups[0]['lr'],
            "batch_size": x_tensor.shape[0],
            "steps_per_second": 1 / (time.time() - start_time),
            "train_mean_reciprocal_rank": reciprocal_rank(flat_logits, flat_labels, reduction='mean', ignore_index=self.metrics_ignore_index),
            "train_correct_token_predictions_at_1": correct_token_predictions_at_k(flat_logits, flat_labels, k=1, reduction='mean', ignore_index=self.metrics_ignore_index),
            "train_correct_token_predictions_at_5": correct_token_predictions_at_k(flat_logits, flat_labels, k=5, reduction='mean', ignore_index=self.metrics_ignore_index),
            "train_correct_token_predictions_at_10": correct_token_predictions_at_k(flat_logits, flat_labels, k=10, reduction='mean', ignore_index=self.metrics_ignore_index),
            "n_support": x_tensor.shape[1],
            "cumulative_training_tokens": self.cumulative_training_tokens,
            "cumulative_training_pflops": self.cumulative_training_pflops
        }, step=step)

        self.lr_scheduler.step()

        return loss.item()

    def _validate(self, val_dataset: FlashANSRDataset, numeric_prediction_loss_weight: float, contrastive_loss_weight: float, contrastive_n_per_class: int, step: int, size: int | None = None, batch_size: int = 128, verbose: bool = False) -> float:
        self.model.eval()
        with torch.no_grad():
            val_ce_loss = 0.0
            val_contrastive_loss = 0.0
            val_num_loss = 0.0
            val_loss = 0.0

            if size is None:
                size = len(val_dataset) * batch_size

            pbar = tqdm(total=size // batch_size, leave=False, position=1, disable=not verbose, desc="Validating")

            for batch in val_dataset.iterate(size=size, batch_size=batch_size, n_per_equation=contrastive_n_per_class):  # TODO: support both compiled dataset and non-compiled dataset that may use a custom batch size
                # Forward
                input_ids, x_tensor, y_tensor, labels, constants, skeleton_hashes = FlashANSRDataset.collate_batch(batch, device=self.model.device)

                # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
                pad_length = self.model.encoder_max_n_variables - x_tensor.shape[2] - y_tensor.shape[2]
                if pad_length > 0:
                    x_tensor = nn.functional.pad(x_tensor, (0, pad_length, 0, 0, 0, 0), value=0)

                # Concatenate x and y tensors as input to the set transformer
                data_tensor = torch.cat([x_tensor, y_tensor], dim=-1)

                # Forward pass
                logits, num_output = self.model.forward(input_ids, data_tensor, numeric_head=(numeric_prediction_loss_weight > 0))

                flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                flat_labels = labels.reshape(-1)

                # Calculate the loss
                ce_loss: torch.Tensor = self.cross_entropy_loss(flat_logits, flat_labels)

                if contrastive_loss_weight > 0:
                    # Use memory (embeddings of encoder) that the model cached during the forward pass
                    contrastive_loss = self.contrastive_loss_fn(self.model.memory.reshape(self.model.memory.shape[0], -1), skeleton_hashes)

                # Compare the num_output at the positions where the labels are `<num>`
                num_loss = torch.tensor(0, device=self.model.device, dtype=torch.float32)

                if numeric_prediction_loss_weight > 0:
                    n_tokens = 0.0
                    for i in range(len(constants)):
                        if len(constants[i]) == 0:
                            continue
                        num_loss += self.mse_loss(num_output[i, 1:, 0][labels[i] == self.numeric_token_index], constants[i])
                        n_tokens += torch.sum(labels[i] != self.metrics_ignore_index).item()

                    if n_tokens > 0:
                        num_loss /= n_tokens

                loss = (ce_loss + numeric_prediction_loss_weight * num_loss + contrastive_loss_weight * contrastive_loss) / (1 + numeric_prediction_loss_weight + contrastive_loss_weight)

                val_ce_loss += ce_loss.item()
                val_contrastive_loss += contrastive_loss.item()
                val_num_loss += num_loss.item()
                val_loss += loss.item()

                pbar.update(1)

            # Calculate the average loss
            val_ce_loss /= (size // batch_size)
            val_contrastive_loss /= (size // batch_size)
            val_num_loss /= (size // batch_size)
            val_loss /= (size // batch_size)

            pbar.close()

        wandb.log({  # type: ignore
            "val_ce_loss": val_ce_loss,
            "val_contrastive_loss": val_contrastive_loss,
            "val_num_loss": val_num_loss,
            "val_loss": val_loss,
            "val_mean_reciprocal_rank": reciprocal_rank(flat_logits, flat_labels, reduction='mean', ignore_index=self.metrics_ignore_index),
            "val_correct_token_predictions_at_1": correct_token_predictions_at_k(flat_logits, flat_labels, k=1, reduction='mean', ignore_index=self.metrics_ignore_index),
            "val_correct_token_predictions_at_5": correct_token_predictions_at_k(flat_logits, flat_labels, k=5, reduction='mean', ignore_index=self.metrics_ignore_index),
            "val_correct_token_predictions_at_10": correct_token_predictions_at_k(flat_logits, flat_labels, k=10, reduction='mean', ignore_index=self.metrics_ignore_index)
        }, step=step)

        return val_loss
