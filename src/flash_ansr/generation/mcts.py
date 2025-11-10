"""Monte Carlo Tree Search generation helper."""
import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from simplipy import SimpliPyEngine

from flash_ansr.decoding.mcts import MCTSConfig
from flash_ansr.model import FlashANSRModel, Tokenizer
from flash_ansr.preprocessing import PromptPrefix
from flash_ansr.refine import ConvergenceError, Refiner


def run_mcts_generation(
    *,
    transformer: FlashANSRModel,
    tokenizer: Tokenizer,
    simplipy_engine: SimpliPyEngine,
    data: torch.Tensor,
    config: MCTSConfig,
    beam_width: int,
    completion_sort: str,
    n_variables: int,
    n_restarts: int,
    refiner_method: str,
    refiner_p0_noise: str | None,
    refiner_p0_noise_kwargs: dict | None,
    parsimony: float,
    compute_fvu: Callable[[float, int, float], float],
    score_from_fvu: Callable[[float, int, float], float],
    float64_eps: float,
    prompt_prefix: PromptPrefix | None,
    verbose: bool,
) -> tuple[list[list[int]], list[float], list[bool], list[float], Dict[Tuple[int, ...], Dict[str, Any]]]:
    """Decode beams with MCTS, returning refiner metadata for cached beams."""
    x_np = data[..., :n_variables].detach().cpu().numpy()
    y_np = data[..., n_variables:].detach().cpu().numpy()

    value_cache: Dict[Tuple[int, ...], tuple[float, dict[str, Any]]] = {}
    refiner_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    y_var = float(np.var(y_np)) if y_np.size > 1 else float(np.nan)

    def value_fn(tokens: Tuple[int, ...]) -> tuple[float, dict[str, Any]]:
        if tokens in value_cache:
            return value_cache[tokens]

        def cache_and_return(reward_value: float, metadata: Optional[dict[str, Any]] = None) -> tuple[float, dict[str, Any]]:
            info = dict(metadata) if metadata is not None else {}
            value_cache[tokens] = (reward_value, info)
            return value_cache[tokens]

        try:
            expression_tokens, _, _ = transformer.tokenizer.extract_expression_from_beam(list(tokens))
        except ValueError:
            return cache_and_return(-config.invalid_penalty, {"length": len(tokens), "log_fvu": float("nan")})

        expression_decoded = tokenizer.decode(expression_tokens, special_tokens="<constant>")
        expression_length = len(expression_decoded)

        if not simplipy_engine.is_valid(expression_decoded):
            return cache_and_return(-config.invalid_penalty, {"length": expression_length, "log_fvu": float("nan")})

        try:
            refiner = Refiner(simplipy_engine=simplipy_engine, n_variables=n_variables).fit(
                expression=expression_decoded,
                X=x_np,
                y=y_np,
                n_restarts=n_restarts,
                method=refiner_method,
                p0_noise=refiner_p0_noise,
                p0_noise_kwargs=refiner_p0_noise_kwargs,
                converge_error="ignore",
            )
        except ConvergenceError:
            return cache_and_return(-config.invalid_penalty, {"length": expression_length, "log_fvu": float("nan")})
        except Exception:
            return cache_and_return(-config.invalid_penalty, {"length": expression_length, "log_fvu": float("nan")})

        if not refiner.valid_fit or len(refiner._all_constants_values) == 0:
            return cache_and_return(-config.invalid_penalty, {"length": expression_length, "log_fvu": float("nan")})

        loss = refiner._all_constants_values[0][-1]
        if np.isnan(loss):
            return cache_and_return(-config.invalid_penalty, {"length": expression_length, "log_fvu": float("nan")})

        fvu = compute_fvu(float(loss), y_np.shape[0], y_var)
        if not np.isfinite(fvu):
            return cache_and_return(-config.invalid_penalty, {"length": expression_length, "log_fvu": float("nan")})

        score = score_from_fvu(fvu, len(expression_decoded), parsimony)
        reward = -score
        log_fvu = float(np.log10(max(float(fvu), float64_eps)))

        metadata = {
            "fvu": float(fvu),
            "log_fvu": log_fvu,
            "length": expression_length,
        }

        constantified_tokens = tuple(tokenizer.constantify_expression(list(tokens)))

        refiner_cache[constantified_tokens] = {
            "refiner": refiner,
            "score": score,
            "fvu": fvu,
            "log_fvu": log_fvu,
            "loss": float(loss),
            "reward": reward,
            "fits": copy.deepcopy(refiner._all_constants_values),
        }

        return cache_and_return(reward, metadata)

    beams, log_probs, completed, rewards = transformer.mcts_decode(
        data=data,
        config=config,
        beam_width=beam_width,
        value_fn=value_fn,
        completion_sort=completion_sort,
        verbose=verbose,
        prompt_prefix=prompt_prefix,
    )

    return beams, log_probs, completed, rewards, refiner_cache
