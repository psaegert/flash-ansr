"""Monte Carlo Tree Search generation helper."""
import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from simplipy import SimpliPyEngine

from flash_ansr.decoding.mcts import AsyncRefineHooks, MCTSConfig
from flash_ansr.model import FlashANSRModel, Tokenizer
from flash_ansr.model.flash_ansr_model import canonicalize_beam
from flash_ansr.preprocessing import PromptPrefix
from flash_ansr.refine import ConvergenceError, Refiner
from flash_ansr.scoring import count_constants, is_constant_token


def _is_constant_token(token: str) -> bool:
    return is_constant_token(token)


def _count_constants(tokens: list[str]) -> int:
    return count_constants(tokens)


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
    length_penalty: float,
    constants_penalty: float,
    likelihood_penalty: float,
    compute_fvu: Callable[[float, int, float], float],
    score_from_fvu: Callable[[float, int, int, float | None, float, float, float], float],
    float64_eps: float,
    prompt_prefix: PromptPrefix | None,
    verbose: bool,
    parallel_refine_fn: Optional[Callable[..., list]] = None,
    parallel_submit_fn: Optional[Callable[..., Any]] = None,
    serial_refine_fn: Optional[Callable[..., Any]] = None,
) -> tuple[list[list[int]], list[float], list[bool], list[float], Dict[Tuple[int, ...], Dict[str, Any]]]:
    """Decode beams with MCTS, returning refiner metadata for cached beams.

    The search refines the SAME (simplify+constantify) expression deployment fits and caches the fit keyed by
    the canonical beam (``canonicalize_beam``), so ``_fit_refine`` reuses it by ``tuple(raw_beam)`` instead of
    refining a second time. The refine is unseeded (fresh entropy), matching the default deploy refiner.
    """
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

        # Canonicalize to the deploy-identical (simplify+constantify) form. The refiner cache is keyed by this
        # canonical key so ONE refiner call serves every raw completion that reduces to it (making the count of
        # refiner calls == distinct valid canonical completions == the budget), and so _fit_refine can reuse the
        # fit by tuple(raw_beam) == canonical_key without a second refine.
        canonical_key, canonical_expression = canonicalize_beam(tokenizer, simplipy_engine, tokens)
        if canonical_expression is None:
            return cache_and_return(-config.invalid_penalty, {"length": len(tokens), "log_fvu": float("nan")})

        expression_length = len(canonical_expression)

        cached = refiner_cache.get(canonical_key)
        if cached is not None:
            # already refined this canonical form via a different raw completion -> reuse (no re-fit)
            return cache_and_return(cached["reward"], {"fvu": cached["fvu"], "log_fvu": cached["log_fvu"], "length": expression_length})

        # Refine the canonical expression with fresh entropy -- the SAME procedure (n_restarts, p0_noise) the
        # deploy refiner uses, whose default is likewise unseeded. The fit is cached and reused at deploy time
        # so each candidate is refined ONCE; a random draw here is quality-equivalent to a fresh deploy refine.
        try:
            refiner = Refiner(simplipy_engine=simplipy_engine, n_variables=n_variables).fit(
                expression=canonical_expression,
                X=x_np,
                y=y_np,
                n_restarts=n_restarts,
                method=refiner_method,
                p0=None,
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

        constant_count = _count_constants(canonical_expression)
        score = score_from_fvu(
            fvu,
            expression_length,
            constant_count,
            None,
            length_penalty,
            constants_penalty,
            likelihood_penalty,
        )
        reward = -score
        log_fvu = float(np.log10(max(float(fvu), float64_eps)))

        refiner_cache[canonical_key] = {
            "refiner": refiner,
            "score": score,
            "fvu": fvu,
            "log_fvu": log_fvu,
            "loss": float(loss),
            "reward": reward,
            "fits": copy.deepcopy(refiner._all_constants_values),
        }

        return cache_and_return(reward, {"fvu": float(fvu), "log_fvu": log_fvu, "length": expression_length})

    def batched_value_fn(token_lists: list[Tuple[int, ...]]) -> list[tuple[float, dict[str, Any]]]:
        """Value a whole batch's completions at once: refine the distinct NEW canonical ones in PARALLEL
        (fork pool via parallel_refine_fn), populate refiner_cache, return (reward, info) per completion.
        Same canonical key / scoring / cache semantics as the serial value_fn; the refine is unseeded (fresh
        entropy per key) matching the deploy default. Cache hits and invalids skip the pool."""
        canon = [canonicalize_beam(tokenizer, simplipy_engine, t) for t in token_lists]
        need: Dict[Tuple[int, ...], list] = {}
        for key, expr in canon:
            if expr is not None and key not in refiner_cache and key not in need:
                need[key] = expr
        if need:
            keys = list(need)
            exprs = [need[k] for k in keys]
            seeds = [int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0]) for _ in keys]
            results = parallel_refine_fn(exprs, seeds, x_np, y_np, y_var) if parallel_refine_fn is not None else [None] * len(keys)
            input_dim = int(x_np.shape[-1])
            for k, expr, res in zip(keys, exprs, results):
                if res is None or not res.get("valid_fit") or not res.get("fits"):
                    continue
                fvu = float(res["fvu"])
                if not np.isfinite(fvu):
                    continue
                fits = res["fits"]
                score = float(res["score"])
                try:
                    refiner_obj = Refiner.from_serialized(simplipy_engine=simplipy_engine, n_variables=n_variables,
                                                          expression=expr, n_inputs=input_dim, fits=fits)
                except Exception:
                    refiner_obj = None
                refiner_cache[k] = {
                    "refiner": refiner_obj, "score": score, "fvu": fvu,
                    "log_fvu": float(np.log10(max(fvu, float64_eps))),
                    "loss": float(fits[0][-1]), "reward": -score, "fits": fits,
                }
        out: list[tuple[float, dict[str, Any]]] = []
        for i, (key, expr) in enumerate(canon):
            if expr is None:
                out.append((-config.invalid_penalty, {"length": len(token_lists[i]), "log_fvu": float("nan")}))
                continue
            cached = refiner_cache.get(key)
            if cached is not None:
                out.append((cached["reward"], {"fvu": cached["fvu"], "log_fvu": cached["log_fvu"], "length": len(expr)}))
            else:
                out.append((-config.invalid_penalty, {"length": len(expr), "log_fvu": float("nan")}))
        return out

    # ---- async (overlapped) refine hooks: SPLIT batched_value_fn into submit/commit halves so the async loop
    # can overlap the GPU produce phase with pool refinement, preserving the identical canonical key + cache
    # schema + (reward, info) contract. Only built when the async pool submit closure is available. ----
    input_dim = int(x_np.shape[-1])

    def _commit_result(canonical_key: Tuple[int, ...], expr: Any, res: Optional[dict]) -> tuple[float, dict[str, Any]]:
        """Populate refiner_cache[canonical_key] IFF the fit is valid (schema byte-identical to batched_value_fn),
        then return (reward, info). Called ONCE per distinct canonical key on the main thread."""
        if res is not None and res.get("valid_fit") and res.get("fits"):
            fvu = float(res["fvu"])
            if np.isfinite(fvu):
                fits = res["fits"]
                score = float(res["score"])
                try:
                    refiner_obj = Refiner.from_serialized(simplipy_engine=simplipy_engine, n_variables=n_variables,
                                                          expression=expr, n_inputs=input_dim, fits=fits)
                except Exception:
                    refiner_obj = None
                refiner_cache[canonical_key] = {
                    "refiner": refiner_obj, "score": score, "fvu": fvu,
                    "log_fvu": float(np.log10(max(fvu, float64_eps))),
                    "loss": float(fits[0][-1]), "reward": -score, "fits": fits,
                }
        cached = refiner_cache.get(canonical_key)
        if cached is not None:
            return cached["reward"], {"fvu": cached["fvu"], "log_fvu": cached["log_fvu"], "length": len(expr)}
        return -config.invalid_penalty, {"length": len(expr), "log_fvu": float("nan")}

    def _canon(tokens: Tuple[int, ...]) -> tuple[Any, Any]:
        try:
            return canonicalize_beam(tokenizer, simplipy_engine, tokens)
        except Exception:
            return (tokens, None)

    def _cached(canonical_key: Any, expr: Any) -> Optional[tuple[float, dict[str, Any]]]:
        cached = refiner_cache.get(canonical_key)
        if cached is None:
            return None
        return cached["reward"], {"fvu": cached["fvu"], "log_fvu": cached["log_fvu"], "length": len(expr)}

    def _fresh_seed() -> int:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

    def _submit(expr: Any) -> Any:
        return parallel_submit_fn(expr, _fresh_seed(), x_np, y_np, y_var)   # type: ignore[misc]

    def _serial(expr: Any) -> Any:
        return serial_refine_fn(expr, _fresh_seed(), x_np, y_np, y_var)     # type: ignore[misc]

    async_hooks: Optional[AsyncRefineHooks] = None
    if config.async_search and parallel_submit_fn is not None and serial_refine_fn is not None:
        async_hooks = AsyncRefineHooks(canon=_canon, cached=_cached, submit=_submit, serial=_serial, commit=_commit_result)

    beams, log_probs, completed, rewards = transformer.mcts_decode(
        data=data,
        config=config,
        beam_width=beam_width,
        value_fn=value_fn,
        batched_value_fn=batched_value_fn if parallel_refine_fn is not None else None,
        async_hooks=async_hooks,
        completion_sort=completion_sort,
        verbose=verbose,
        prompt_prefix=prompt_prefix,
    )

    return beams, log_probs, completed, rewards, refiner_cache
