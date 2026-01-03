"""Batch collation utilities for preparing model inputs."""

from typing import Any

import torch

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.numeric import build_numeric_sequences, merge_numeric_sequence


class BatchFormatter:
    """Utility that normalizes jagged dataloader batches."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _pad_sequence(
        sequence: list[int] | torch.Tensor,
        max_length: int,
        pad_value: Any,
        device: str | torch.device | int = "cpu",
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        if not isinstance(sequence, torch.Tensor):
            seq_tensor = torch.tensor(sequence, device=device, dtype=dtype)
        else:
            seq_tensor = sequence.to(device=device, dtype=dtype)

        return torch.nn.functional.pad(seq_tensor, (0, max_length - len(seq_tensor)), value=pad_value)

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()

    def ensure_numeric_channel(self, batch: dict[str, Any]) -> None:
        """Ensure numeric channels exist by merging precomputed and fresh sequences."""
        input_ids = batch.get("input_ids")
        constants = batch.get("constants")

        if input_ids is None or constants is None:
            return

        computed_numeric = build_numeric_sequences(self.tokenizer, input_ids, constants)
        existing_numeric = batch.get("input_num")
        if existing_numeric is None:
            batch["input_num"] = computed_numeric
            return

        if isinstance(existing_numeric, torch.Tensor):
            existing_sequences = existing_numeric
        else:
            existing_sequences = existing_numeric

        merged = [
            merge_numeric_sequence(
                existing_sequences[idx] if idx < len(existing_sequences) else None,
                computed_seq,
            )
            for idx, computed_seq in enumerate(computed_numeric)
        ]

        batch["input_num"] = merged

    def collate(self, batch: dict[str, Any], device: str | torch.device | int = "cpu") -> dict[str, Any]:
        """Pad and bucket batch fields to consistent shapes for model consumption."""
        pad_token_id = self.tokenizer["<pad>"]

        def _adjust_length(tensor: torch.Tensor, target_length: int, pad_value: Any) -> torch.Tensor:
            if tensor.size(1) == target_length:
                return tensor
            if tensor.size(1) > target_length:
                return tensor[:, :target_length, ...]
            pad_shape = (tensor.size(0), target_length - tensor.size(1), *tensor.shape[2:])
            pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=1)

        if isinstance(batch["input_ids"][0], list):
            token_lengths = [len(seq) for seq in batch["input_ids"]]
        else:
            token_mask = batch["input_ids"] != pad_token_id
            if token_mask.ndim == 1:
                token_lengths = [int(token_mask.sum().item())]
            else:
                token_lengths = [int(length) for length in token_mask.sum(dim=1).tolist()]

        numeric_lengths: list[int] = []
        if "input_num" in batch:
            if isinstance(batch["input_num"][0], list):
                numeric_lengths = [len(seq) for seq in batch["input_num"]]
            else:
                numeric_tensor = batch["input_num"]
                if numeric_tensor.dim() == 3:
                    numeric_tensor = numeric_tensor.squeeze(-1)
                numeric_mask = torch.isfinite(numeric_tensor)
                numeric_lengths = [int(length) for length in numeric_mask.sum(dim=1).tolist()]

        prompt_lengths: list[int] = []
        if "prompt_mask" in batch:
            prompt_field = batch["prompt_mask"]
            if isinstance(prompt_field, list) and prompt_field:
                prompt_lengths = [len(seq) for seq in prompt_field]
            elif isinstance(prompt_field, torch.Tensor):
                prompt_lengths = [prompt_field.shape[1]] * prompt_field.shape[0]

        combined_lengths = token_lengths.copy() if token_lengths else []
        combined_lengths.extend(numeric_lengths)
        combined_lengths.extend(prompt_lengths)
        max_sequence_length = max(combined_lengths) if combined_lengths else 1
        token_bucket_length = self._next_power_of_two(max_sequence_length)

        if isinstance(batch["input_ids"][0], list):
            padded_input_ids = [
                self._pad_sequence(seq, token_bucket_length, pad_token_id, device=device, dtype=torch.long)
                for seq in batch["input_ids"]
            ]
            batch["input_ids"] = torch.stack(padded_input_ids)
        else:
            current_tensor = batch["input_ids"].to(device=device, dtype=torch.long)
            token_bucket_length = min(token_bucket_length, current_tensor.size(1))
            batch["input_ids"] = _adjust_length(current_tensor, token_bucket_length, pad_token_id)

        for key, dtype in [("x_tensors", torch.float32), ("y_tensors", torch.float32)]:
            if isinstance(batch[key], list):
                batch[key] = torch.stack(batch[key])
            batch[key] = batch[key].to(device=device, dtype=dtype)

        if "data_attn_mask" in batch:
            batch["data_attn_mask"] = batch["data_attn_mask"].to(device=device, dtype=torch.bool)
        else:
            attn_shape = batch["x_tensors"].shape[:2]
            batch["data_attn_mask"] = torch.ones(attn_shape, device=device, dtype=torch.bool)

        support_lengths = batch["data_attn_mask"].sum(dim=1)
        max_support_length = int(support_lengths.max().item()) if support_lengths.numel() > 0 else 1
        support_bucket_length = self._next_power_of_two(max_support_length)
        support_bucket_length = min(support_bucket_length, batch["x_tensors"].shape[1])
        if support_bucket_length < batch["x_tensors"].shape[1]:
            batch["x_tensors"] = batch["x_tensors"][:, :support_bucket_length, :]
            batch["y_tensors"] = batch["y_tensors"][:, :support_bucket_length, :]
            batch["data_attn_mask"] = batch["data_attn_mask"][:, :support_bucket_length]

        constants_list = []
        for const_item in batch["constants"]:
            if not isinstance(const_item, torch.Tensor):
                const_item = torch.tensor(const_item, dtype=torch.float32)
            constants_list.append(const_item.to(device))
        batch["constants"] = constants_list

        if "input_num" in batch:
            target_length = token_bucket_length
            if isinstance(batch["input_num"][0], list):
                padded_input_num = [
                    self._pad_sequence(seq, target_length, torch.nan, device=device, dtype=torch.float32)
                    for seq in batch["input_num"]
                ]
                batch["input_num"] = torch.stack(padded_input_num).unsqueeze(-1)
            else:
                input_num_tensor = batch["input_num"]
                if input_num_tensor.dim() == 2:
                    input_num_tensor = input_num_tensor.unsqueeze(-1)
                input_num_tensor = input_num_tensor.to(device=device, dtype=torch.float32)
                batch["input_num"] = _adjust_length(input_num_tensor, target_length, float("nan"))

        if "prompt_mask" in batch:
            target_length = token_bucket_length
            if isinstance(batch["prompt_mask"][0], list):
                padded_prompt_masks = [
                    self._pad_sequence(seq, target_length, False, device=device, dtype=torch.bool)
                    for seq in batch["prompt_mask"]
                ]
                batch["prompt_mask"] = torch.stack(padded_prompt_masks)
            else:
                prompt_mask_tensor = batch["prompt_mask"].to(device=device, dtype=torch.bool)
                batch["prompt_mask"] = _adjust_length(prompt_mask_tensor, target_length, False)

        if "complexity" in batch:
            batch["complexity"] = [
                torch.tensor(c, device=device, dtype=torch.float32) if c is not None else None
                for c in batch["complexity"]
            ]

        batch["labels"] = batch["input_ids"].clone()[..., 1:]

        batch["expression_ids"] = []
        expression_to_id: dict[tuple, int] = {}

        for expr in batch["input_ids"]:
            expr_key = tuple(expr.flatten().tolist())
            if expr_key not in expression_to_id:
                expression_to_id[expr_key] = len(expression_to_id)
            batch["expression_ids"].append(expression_to_id[expr_key])
        batch["expression_ids"] = torch.tensor(batch["expression_ids"], device=device, dtype=torch.long)

        return batch
