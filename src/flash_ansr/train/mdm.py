"""The masked-diffusion training objective (experiment exp/mdm-decoder).

A ``decoder_objective: mdm`` model is trained as an absorbing-state masked diffusion model in the MDLM /
LLaDA form (Sahoo et al. 2024; Nie et al. 2025), on exactly the sequences the autoregressive models train on:

* **The canvas** is the whole training sequence, padded to the decoder's ``max_seq_len`` for every batch, so
  its width never hints at the batch's longest sequence.
* **What is given and what is predicted** follows the autoregressive label masks, so the two objectives
  supervise the same tokens. A token the autoregressive loss supervises is a denoising target; a token it
  masks (``<bos>``, the emission flag, stated values, the task openers, every ``<float>``) is always
  visible. ``<expression>`` is always visible too: inference gives it as the last prompt token. The pad
  tail after the first ``<eos>`` is a target, so the model places the end of the sequence itself.
* **Noise**: one mask ratio ``r ~ U(eps, 1)`` per sequence, stratified across the batch; each target is
  replaced by ``<mask>`` with probability ``r``.
* **Loss**: cross-entropy at the masked positions only, each weighted ``1 / r`` -- the continuous-time
  bound on ``-log p`` under the linear schedule -- summed and divided by the number of supervised
  non-pad tokens, the same denominator as the autoregressive loss. No time input: the optimal denoiser of
  an absorbing process does not depend on the time once it sees which positions are masked (Ou et al. 2024).
"""
from dataclasses import dataclass

import torch

RATIO_FLOOR = 1e-3


@dataclass
class MaskedBatch:
    """One corrupted batch: the decoder's input and what the loss needs."""
    noisy_ids: torch.Tensor      # (B, L) the canvas with masked targets replaced by <mask>
    targets: torch.Tensor        # (B, L) the clean canvas
    masked: torch.Tensor         # (B, L) True where a target was masked (the loss positions)
    ratio: torch.Tensor          # (B,) the mask ratio drawn per sequence
    n_supervised: torch.Tensor   # () supervised non-pad tokens in the batch (the loss denominator)
    input_num: torch.Tensor | None


def pad_to_canvas(tensor: torch.Tensor, length: int, value: float | int | bool) -> torch.Tensor:
    """Right-pad (or refuse to cut) dimension 1 of ``tensor`` to ``length``."""
    current = tensor.shape[1]
    if current > length:
        raise ValueError(f"sequence of {current} positions does not fit the {length}-position canvas")
    if current == length:
        return tensor
    shape = list(tensor.shape)
    shape[1] = length - current
    return torch.cat([tensor, torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)], dim=1)


def targets_and_supervision(input_ids: torch.Tensor, labels: torch.Tensor, ignore_index: int,
                            eos_id: int, expression_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(maskable, supervised)`` over the unshifted canvas.

    ``labels[:, p]`` is the autoregressive target of position ``p``, i.e. the token at ``p + 1``, already
    masked to ``ignore_index`` wherever the autoregressive loss does not supervise it (pads included, since
    the ignore index is the pad id). ``supervised`` is that set moved onto the token positions, minus
    ``<expression>``; ``maskable`` adds the pad tail after the first ``<eos>``."""
    batch, length = input_ids.shape
    supervised = torch.zeros(batch, length, dtype=torch.bool, device=input_ids.device)
    n_labels = min(labels.shape[1], length - 1)
    supervised[:, 1:1 + n_labels] = labels[:, :n_labels] != ignore_index
    supervised &= input_ids != expression_id
    is_eos = input_ids == eos_id
    after_eos = (torch.cumsum(is_eos.int(), dim=1) - is_eos.int()) > 0
    return supervised | after_eos, supervised


def corrupt(input_ids: torch.Tensor, labels: torch.Tensor, *, ignore_index: int, eos_id: int, expression_id: int,
            mask_id: int, canvas: int, input_num: torch.Tensor | None = None,
            generator: torch.Generator | None = None) -> MaskedBatch:
    """Pad a collated batch to the canvas and mask its targets (see the module docstring)."""
    input_ids = pad_to_canvas(input_ids, canvas, ignore_index)
    if input_num is not None:
        input_num = pad_to_canvas(input_num, canvas, float("nan"))
    maskable, supervised = targets_and_supervision(input_ids, labels, ignore_index, eos_id, expression_id)
    batch = input_ids.shape[0]
    device = input_ids.device
    # Stratified ratios: one uniform offset, the batch spread evenly over [0, 1), in random row order.
    offset = torch.rand((), generator=generator, device=device)
    order = torch.randperm(batch, generator=generator, device=device)
    ratio = ((offset + order.to(torch.float32) / batch) % 1.0).clamp_min(RATIO_FLOOR)
    draw = torch.rand(input_ids.shape, generator=generator, device=device)
    masked = maskable & (draw < ratio.unsqueeze(1))
    noisy_ids = torch.where(masked, torch.full_like(input_ids, mask_id), input_ids)
    if input_num is not None:
        input_num = input_num.masked_fill(masked.unsqueeze(-1) if input_num.dim() == 3 else masked, float("nan"))
    return MaskedBatch(noisy_ids=noisy_ids, targets=input_ids, masked=masked, ratio=ratio,
                       n_supervised=supervised.sum(), input_num=input_num)


def masked_diffusion_loss(logits: torch.Tensor, batch: MaskedBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """``(loss, masked_logits)``: the 1/r-weighted cross-entropy over the masked positions divided by the
    supervised token count, and the logits at those positions (for the z-loss and the logit-scale alarms)."""
    masked_logits = logits[batch.masked]
    if masked_logits.shape[0] == 0:
        return logits.sum() * 0.0, masked_logits
    ce = torch.nn.functional.cross_entropy(masked_logits.float(), batch.targets[batch.masked], reduction="none")
    weight = (1.0 / batch.ratio).unsqueeze(1).expand_as(batch.masked)[batch.masked]
    return (ce * weight).sum() / batch.n_supervised.clamp_min(1), masked_logits
