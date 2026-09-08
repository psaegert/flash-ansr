"""Optimizers for training: the factory, and AdaMuon.

AdaMuon (Si, Zhang, Shen 2025, arXiv:2507.11005) is Muon with an element-wise second moment on the
orthogonalized update and an RMS-aligned rescaling, so the AdamW learning-rate schedule is reused
unchanged. As in the reference implementation (github.com/Chongjie-Si/AdaMuon) it optimizes the
hidden weight matrices only; embeddings, the output projections and every one-dimensional
parameter stay on AdamW. Both live in ONE ``torch.optim.Optimizer`` here so the trainer's
scheduler, checkpoints and resume see a single object.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch_optimizer
from torch import Tensor, nn

# Quintic Newton-Schulz coefficients (Jordan et al. 2024, the Muon iteration): fast convergence of
# the singular values into a band around 1 rather than exact orthogonality.
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

# The RMS every AdaMuon update is rescaled to, the paper's Adam-matching constant.
_RMS_TARGET = 0.2

ROLE_HIDDEN_MATRIX = "hidden_matrix"
ROLE_EMBEDDING = "embedding"
ROLE_OUTPUT_PROJECTION = "output_projection"
ROLE_VECTOR = "vector"
PARAMETER_ROLES = (ROLE_HIDDEN_MATRIX, ROLE_EMBEDDING, ROLE_OUTPUT_PROJECTION, ROLE_VECTOR)


def newton_schulz_orthogonalize_batched(matrices: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximately orthogonalize a stack of same-shaped 2-D matrices ``[B, m, n]`` with the quintic
    Newton-Schulz iteration, each one exactly as :func:`newton_schulz_orthogonalize` would: in
    bfloat16, on the smaller side, normalized by its own Frobenius norm. One batched matmul per
    iteration instead of one launch per matrix -- the optimizer step's cost is launches, not FLOPs.
    """
    if matrices.ndim != 3:
        raise ValueError(f"newton_schulz_orthogonalize_batched expects [B, m, n], got shape {tuple(matrices.shape)}")
    x = matrices.to(torch.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = _NS_COEFFICIENTS
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transposed else x


def newton_schulz_orthogonalize(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximately orthogonalize a 2-D matrix with the quintic Newton-Schulz iteration.

    Runs in bfloat16 on the smaller side of the matrix, exactly as the Muon/AdaMuon reference
    code does; returns a bfloat16 tensor of the input's shape whose singular values sit in a
    band around 1 (the iteration trades exactness for speed).
    """
    if matrix.ndim != 2:
        raise ValueError(f"newton_schulz_orthogonalize expects a 2-D matrix, got shape {tuple(matrix.shape)}")
    return newton_schulz_orthogonalize_batched(matrix.unsqueeze(0), steps=steps, eps=eps).squeeze(0)


def _module_parameter_names(model: nn.Module, module: nn.Module) -> set[str]:
    ids = {id(p) for p in module.parameters(recurse=False)}
    return {name for name, p in model.named_parameters() if id(p) in ids}


def parameter_roles(model: nn.Module) -> dict[str, str]:
    """Role of every trainable parameter of ``model`` (see ``PARAMETER_ROLES``).

    A model can define ``parameter_roles()`` itself (FlashANSRModel does: it knows which of its
    Linear layers are input embeddings and which are the heads' final projections). The generic
    rule for any other module: 2-D weights of ``nn.Linear`` are hidden matrices, ``nn.Embedding``
    weights are embeddings, everything else is a vector.
    """
    own = getattr(model, "parameter_roles", None)
    if callable(own):
        roles = dict(own())
    else:
        roles = {}
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                roles[f"{module_name}.weight" if module_name else "weight"] = ROLE_HIDDEN_MATRIX
            elif isinstance(module, nn.Embedding):
                roles[f"{module_name}.weight" if module_name else "weight"] = ROLE_EMBEDDING
        for name, _ in model.named_parameters():
            roles.setdefault(name, ROLE_VECTOR)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    missing = sorted(set(names) - set(roles))
    if missing:
        raise ValueError(f"parameter_roles: no role for {missing}")
    bad = sorted((n, r) for n, r in roles.items() if r not in PARAMETER_ROLES)
    if bad:
        raise ValueError(f"parameter_roles: unknown roles {bad}")
    return {n: roles[n] for n in names}


def adamuon_param_groups(model: nn.Module, *, weight_decay: float, adam_weight_decay: float) -> list[dict[str, Any]]:
    """The three AdaMuon parameter groups, every trainable parameter in exactly one of them.

    hidden matrices -> AdaMuon with ``weight_decay``; embeddings and output projections -> AdamW
    with ``adam_weight_decay``; vectors (biases, norm gains) -> AdamW without decay, the
    reference implementation's split.
    """
    params = dict(model.named_parameters())
    roles = parameter_roles(model)
    matrices = [params[n] for n, r in roles.items() if r == ROLE_HIDDEN_MATRIX]
    decayed = [params[n] for n, r in roles.items() if r in (ROLE_EMBEDDING, ROLE_OUTPUT_PROJECTION)]
    vectors = [params[n] for n, r in roles.items() if r == ROLE_VECTOR]
    for p in matrices:
        if p.ndim != 2:
            raise ValueError(f"AdaMuon hidden matrices must be 2-D, got shape {tuple(p.shape)}")
    groups: list[dict[str, Any]] = []
    if matrices:
        groups.append({"params": matrices, "use_muon": True, "weight_decay": weight_decay, "role": "hidden_matrix"})
    if decayed:
        groups.append({"params": decayed, "use_muon": False, "weight_decay": adam_weight_decay, "role": "embedding_and_output"})
    if vectors:
        groups.append({"params": vectors, "use_muon": False, "weight_decay": 0.0, "role": "vector"})
    return groups


class AdaMuon(torch.optim.Optimizer):
    """AdaMuon for the hidden weight matrices, AdamW for everything else, in one optimizer.

    Parameter groups carry ``use_muon``. For a Muon group (2-D parameters only) each step is, per
    matrix, Algorithm 1 of the paper with the reference code's defaults:

        M <- momentum * M + G                        (Nesterov: the direction is G + momentum * M)
        O <- NewtonSchulz(sign(M))                     (bfloat16, ``ns_steps`` iterations)
        V <- momentum * V + (1 - momentum) * O * O     (no bias correction; the rescaling removes it)
        U <- O / (sqrt(V) + eps)
        W <- W - lr * (0.2 * sqrt(m * n) / ||U||_F * U + weight_decay * W)

    The second moment is kept in the parameter's dtype rather than the reference's bfloat16
    buffer. Adam groups run AdamW exactly as ``torch.optim.AdamW`` (decoupled decay,
    bias-corrected moments) with ``betas`` and ``adam_eps``.

    ``lr`` is shared by all groups: the RMS-aligned rescaling makes the AdamW schedule the right
    schedule for the Muon groups too (the paper's point), so the trainer's ``lr: 1`` times the
    piecewise-linear schedule applies unchanged.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be at least 1, got {ns_steps}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, eps=eps,
            betas=tuple(betas), adam_eps=adam_eps, use_muon=True,
        )
        # An Adam group that states no weight_decay of its own decays at the AdamW rate, not at
        # the Muon rate the group defaults would give it (set before the base class builds groups).
        self.adam_weight_decay = float(adam_weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            f"AdaMuon (use_muon=True) groups take 2-D weight matrices only, got shape "
                            f"{tuple(p.shape)}; put vectors, embeddings and output projections in a "
                            f"use_muon=False group")

    def add_param_group(self, param_group: dict[str, Any]) -> None:  # noqa: D102 (torch API)
        explicit_weight_decay = "weight_decay" in param_group
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if not group["use_muon"] and not explicit_weight_decay:
            group["weight_decay"] = self.adam_weight_decay

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: D102 (torch API)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        """The paper's update for every matrix of the group, in the same arithmetic as the
        per-matrix reference but without its launch storm: momentum, sign and second-moment
        updates as foreach kernels, Newton-Schulz once per distinct shape on the stacked
        matrices, and the RMS-aligning scale kept on the device (no host sync per parameter)."""
        lr, weight_decay, momentum = group["lr"], group["weight_decay"], group["momentum"]
        nesterov, ns_steps, eps = group["nesterov"], group["ns_steps"], group["eps"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        for p in params:
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
                state["second_moment"] = torch.zeros_like(p)
        grads = [p.grad for p in params]
        bufs = [self.state[p]["momentum_buffer"] for p in params]
        seconds = [self.state[p]["second_moment"] for p in params]
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        directions = torch._foreach_add(grads, bufs, alpha=momentum) if nesterov else bufs
        signs = torch._foreach_sign(directions)
        orthos: list[Tensor | None] = [None] * len(params)
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for i, sign in enumerate(signs):
            by_shape.setdefault(tuple(sign.shape), []).append(i)
        for indices in by_shape.values():
            stacked = torch.stack([signs[i] for i in indices])
            ortho = newton_schulz_orthogonalize_batched(stacked, steps=ns_steps)
            for j, i in enumerate(indices):
                orthos[i] = ortho[j].to(params[i].dtype)
        orthos_list = [o for o in orthos if o is not None]
        torch._foreach_mul_(seconds, momentum)
        torch._foreach_addcmul_(seconds, orthos_list, orthos_list, value=1.0 - momentum)
        denoms = torch._foreach_sqrt(seconds)
        torch._foreach_add_(denoms, eps)
        updates = torch._foreach_div(orthos_list, denoms)
        norms = torch._foreach_norm(updates)
        if weight_decay != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)
        for p, update, norm in zip(params, updates, norms):
            update.mul_(_RMS_TARGET * math.sqrt(p.shape[0] * p.shape[1]) / (norm + eps))
        torch._foreach_add_(params, updates, alpha=-lr)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        """AdamW for the group, torch's own arithmetic, as foreach kernels over the parameters
        that share a step count (all of them, in the usual run)."""
        lr, weight_decay, eps = group["lr"], group["weight_decay"], group["adam_eps"]
        beta1, beta2 = group["betas"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        by_step: dict[int, list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            by_step.setdefault(state["step"], []).append(p)
        for t, ps in by_step.items():
            grads = [p.grad for p in ps]
            exp_avgs = [self.state[p]["exp_avg"] for p in ps]
            exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in ps]
            if weight_decay != 0.0:
                torch._foreach_mul_(ps, 1.0 - lr * weight_decay)
            torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** t
            bias_correction2 = 1.0 - beta2 ** t
            denoms = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denoms, math.sqrt(bias_correction2))
            torch._foreach_add_(denoms, eps)
            torch._foreach_addcdiv_(ps, exp_avgs, denoms, value=-lr / bias_correction1)


def build_optimizer(spec: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer a train.yaml ``optimizer`` block describes, for ``model``.

    ``name: AdaMuon`` assigns the model's parameters by role (``adamuon_param_groups``); every
    other name is looked up in ``torch.optim`` and ``torch_optimizer`` and gets
    ``model.parameters()``.
    """
    name = spec["name"]
    kwargs = dict(spec.get("kwargs", {}))
    if name == "AdaMuon":
        weight_decay = float(kwargs.pop("weight_decay", 0.1))
        adam_weight_decay = float(kwargs.pop("adam_weight_decay", 0.01))
        groups = adamuon_param_groups(model, weight_decay=weight_decay, adam_weight_decay=adam_weight_decay)
        return AdaMuon(groups, weight_decay=weight_decay, adam_weight_decay=adam_weight_decay, **kwargs)
    return get_optimizer(name, params=model.parameters(), **kwargs)


def get_optimizer(name: str, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
    """Instantiate an optimiser by ``name`` from this module, ``torch.optim`` or ``torch_optimizer``."""
    if name == "AdaMuon":
        return AdaMuon(*args, **kwargs)
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(*args, **kwargs)
    if hasattr(torch_optimizer, name):
        return getattr(torch_optimizer, name)(*args, **kwargs)
    raise NotImplementedError(f"Optimizer {name} not found in torch.optim or torch_optimizer")
