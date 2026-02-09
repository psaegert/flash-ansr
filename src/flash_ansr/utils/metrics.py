"""Torch-based metric helpers for expression analysis."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch.func import hessian, jacrev, vmap

from flash_ansr.expressions.compilation import codify


def build_expression_callable(
    simplipy_engine: Any,
    expression_tokens: Sequence[str],
    variables: Sequence[str],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile a prefix expression with embedded constants into a torch-callable function."""
    executable_prefix = simplipy_engine.operators_to_realizations(list(expression_tokens))
    code_string = simplipy_engine.prefix_to_infix(executable_prefix, realization=True)
    code = codify(code_string, list(variables))
    compiled_fn = simplipy_engine.code_to_lambda(code)

    def _fn(x: torch.Tensor) -> torch.Tensor:
        return compiled_fn(*x)

    return _fn


def estimate_curvature_metric(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    """Estimate average Hessian Frobenius norm over support points."""
    if X.numel() == 0:
        return torch.tensor(float("nan"), dtype=X.dtype, device=X.device)

    hess_fn = hessian(f, argnums=0)
    batch_hessian = vmap(hess_fn)(X)
    curvature_norms = torch.norm(batch_hessian, p="fro", dim=(1, 2))
    return curvature_norms.mean()


def estimate_fisher_metric(f: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    """Estimate trace of Fisher information under unit-variance noise."""
    if X.numel() == 0:
        return torch.tensor(float("nan"), dtype=X.dtype, device=X.device)

    grad_fn = jacrev(f, argnums=0)
    batch_grad = vmap(grad_fn)(X)
    fisher_trace = torch.sum(batch_grad ** 2, dim=1)
    return fisher_trace.mean()
