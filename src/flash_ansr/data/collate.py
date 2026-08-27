"""Batch collation utilities for preparing model inputs."""

from typing import Any

import torch

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.numeric import build_numeric_sequences


def mask_float_targets(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    float_token_id: int,
    ignore_index: int,
) -> torch.Tensor:
    """Mask the CE terms whose TARGET token is ``<float>``, wherever they occur.

    The model must never be trained to emit ``<float>``: in prompt spans it carries a prompt
    value (already prompt-masked), and under the v24 ``ieee754_mixed`` representation it is the
    compact-constant form that only ever enters a sequence by pipeline compaction, never by
    model emission. The mask is keyed on the SHIFTED inputs (``input_ids[..., 1:]``), the same
    shifted-label discipline as the prompt mask: ``labels[p]`` is the target of position ``p``
    (= input token ``p + 1``), so exactly the terms whose target is ``<float>`` are masked --
    not the terms at the ``<float>`` position itself (the classic off-by-one).

    Parameters
    ----------
    labels : torch.Tensor
        The (possibly already prompt-masked) label tensor, modified in place.
    input_ids : torch.Tensor
        The unshifted input token ids the labels were derived from.
    float_token_id : int
        The id of the ``<float>`` token.
    ignore_index : int
        The CE ignore index masked positions are set to.

    Returns
    -------
    torch.Tensor
        The ``labels`` tensor (masked in place).
    """
    target_mask = input_ids[..., 1:] == float_token_id
    if target_mask.shape[-1] > labels.shape[-1]:
        target_mask = target_mask[..., :labels.shape[-1]]
    elif target_mask.shape[-1] < labels.shape[-1]:
        padding = torch.zeros(
            (*target_mask.shape[:-1], labels.shape[-1] - target_mask.shape[-1]),
            dtype=torch.bool,
            device=target_mask.device,
        )
        target_mask = torch.cat([target_mask, padding], dim=-1)
    labels[target_mask] = ignore_index
    return labels


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

        if batch.get("input_num") is not None:
            # The worker's channel is AUTHORITATIVE (audit 2026-08-24). Recomputing
            # writes each constant's value at its <constant> position and the merge
            # preferred the computed value over the worker's NaN -- harmless while
            # mixed bodies contained no <constant> tokens, but a MASKED body's
            # placeholders would receive their own ground-truth values on the
            # numeric channel (model input): the infilling task becomes
            # copy-from-input and flagged emission trains with the answer visible.
            # For unmasked mixed batches the old merge was a no-op, so returning the
            # worker channel verbatim is byte-identical there.
            return

        batch["input_num"] = build_numeric_sequences(self.tokenizer, input_ids, constants)

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
        if "outlier_mask" in batch:
            batch["outlier_mask"] = batch["outlier_mask"].to(device=device, dtype=torch.bool)
        if "residual" in batch:
            batch["residual"] = batch["residual"].to(device=device, dtype=torch.float32)

        support_lengths = batch["data_attn_mask"].sum(dim=1)
        max_support_length = int(support_lengths.max().item()) if support_lengths.numel() > 0 else 1
        support_bucket_length = self._next_power_of_two(max_support_length)
        support_bucket_length = min(support_bucket_length, batch["x_tensors"].shape[1])
        if support_bucket_length < batch["x_tensors"].shape[1]:
            batch["x_tensors"] = batch["x_tensors"][:, :support_bucket_length, :]
            batch["y_tensors"] = batch["y_tensors"][:, :support_bucket_length, :]
            batch["data_attn_mask"] = batch["data_attn_mask"][:, :support_bucket_length]
            if "outlier_mask" in batch:
                batch["outlier_mask"] = batch["outlier_mask"][:, :support_bucket_length]
            if "residual" in batch:
                batch["residual"] = batch["residual"][:, :support_bucket_length]

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

        # v24 task-block loss mask: True = masked from the loss (same polarity and
        # shifted-label discipline as prompt_mask; padding False is safe -- pads are
        # already ignore_index in the labels).
        if "task_mask" in batch:
            target_length = token_bucket_length
            if isinstance(batch["task_mask"][0], list):
                padded_task_masks = [
                    self._pad_sequence(seq, target_length, False, device=device, dtype=torch.bool)
                    for seq in batch["task_mask"]
                ]
                batch["task_mask"] = torch.stack(padded_task_masks)
            else:
                task_mask_tensor = batch["task_mask"].to(device=device, dtype=torch.bool)
                batch["task_mask"] = _adjust_length(task_mask_tensor, target_length, False)

        # Per-position task-segment ids (0 expression / 1 complexity / 2 predict_y); pad 0.
        if "task_segments" in batch:
            target_length = token_bucket_length
            if isinstance(batch["task_segments"][0], list):
                padded_segments = [
                    self._pad_sequence(seq, target_length, 0, device=device, dtype=torch.long)
                    for seq in batch["task_segments"]
                ]
                batch["task_segments"] = torch.stack(padded_segments)
            else:
                segments_tensor = batch["task_segments"].to(device=device, dtype=torch.long)
                batch["task_segments"] = _adjust_length(segments_tensor, target_length, 0)

        if "complexity" in batch:
            batch["complexity"] = [
                torch.tensor(c, device=device, dtype=torch.float32) if c is not None else None
                for c in batch["complexity"]
            ]

        for key in ("fisher_metric", "curvature_metric"):
            if key not in batch:
                continue
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device=device, dtype=torch.float32)
            else:
                batch[key] = torch.tensor(batch[key], device=device, dtype=torch.float32)

        # Optional condition (CFG): per-example boolean (True = conditioned). Arrives as a list of bools
        # from the worker metadata expansion; tensorize to shape (B,) WITHOUT padding (it is per-example,
        # not per-token/per-support). Absent -> the model defaults to all-conditioned (condition_mask=None).
        if "condition_mask" in batch:
            condition_mask = batch["condition_mask"]
            if not isinstance(condition_mask, torch.Tensor):
                condition_mask = torch.tensor(condition_mask, dtype=torch.bool)
            batch["condition_mask"] = condition_mask.to(device=device, dtype=torch.bool)

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
