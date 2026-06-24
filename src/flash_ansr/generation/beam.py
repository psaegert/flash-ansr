"""Beam-style generation helpers."""
from typing import Any, Iterable

from flash_ansr.model import FlashANSRModel
from flash_ansr.preprocessing import PromptPrefix


def _nan_rewards(count: int) -> list[float]:
    return [float("nan")] * count


def run_beam_search(
    transformer: FlashANSRModel,
    *,
    data: Any,
    verbose: bool,
    prompt_prefix: PromptPrefix | None,
    generation_kwargs: dict[str, Any] | Iterable[tuple[str, Any]] | None,
) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
    """Execute beam search and return beams with placeholder rewards."""
    kwargs = dict(generation_kwargs or {})
    beams, log_probs, completed = transformer.beam_search(
        data=data,
        verbose=verbose,
        prompt_prefix=prompt_prefix,
        **kwargs,
    )
    return beams, log_probs, completed, _nan_rewards(len(beams))


def run_softmax_sampling(
    transformer: FlashANSRModel,
    *,
    data: Any,
    verbose: bool,
    prompt_prefix: PromptPrefix | None,
    generation_kwargs: dict[str, Any] | Iterable[tuple[str, Any]] | None,
    memory: Any | None = None,
) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
    """Execute top-k / nucleus sampling and return beams with placeholder rewards.

    ``memory`` may be supplied by the caller to skip the encoder forward inside
    ``sample_top_kp``. The outer batching loop in ``FlashANSR.generate`` uses
    this to reuse a single encoder pass across chunks.
    """
    kwargs = dict(generation_kwargs or {})
    beams, log_probs, completed = transformer.sample_top_kp(
        data=data,
        verbose=verbose,
        prompt_prefix=prompt_prefix,
        memory=memory,
        **kwargs,
    )
    return beams, log_probs, completed, _nan_rewards(len(beams))
