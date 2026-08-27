"""Model weights on disk: safetensors, one filename, one code path.

Weights are the artifact people download, so they are stored in the format the ecosystem
reads: a flat tensor map with a JSON header, no pickle, memory-mappable, loadable from any
framework. Optimiser/scaler/scheduler state stays on ``torch.save`` -- it is not all
tensors (param groups, step counters, Python scalars), and it never leaves the machine
that wrote it.

Checkpoints written before this carry ``state_dict.pt`` and are NOT read here: the loader
serves one format so there is no second path to keep honest. :func:`convert_legacy_weights`
converts an existing directory in one shot, leaving the ``.pt`` file untouched beside it.
"""
import os
from typing import Any

import torch
from safetensors.torch import load_file, save_file

__all__ = ["WEIGHTS_FILENAME", "LEGACY_WEIGHTS_FILENAME",
           "save_weights", "load_weights", "convert_legacy_weights"]

WEIGHTS_FILENAME = "model.safetensors"
LEGACY_WEIGHTS_FILENAME = "state_dict.pt"


def save_weights(module: torch.nn.Module, directory: str) -> str:
    """Write ``module``'s state dict to ``directory`` as safetensors; return the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, WEIGHTS_FILENAME)
    # Contiguous, unshared, CPU. save_file refuses tied storages rather than silently
    # dropping one of the names, which is the behaviour we want if weights are ever tied.
    state = {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}
    save_file(state, path)
    return path


def load_weights(module: torch.nn.Module, directory: str, *,
                 device: Any = "cpu", strict: bool = True) -> None:
    """Load safetensors weights from ``directory`` into ``module``, in place."""
    path = os.path.join(directory, WEIGHTS_FILENAME)
    if not os.path.exists(path):
        legacy = os.path.join(directory, LEGACY_WEIGHTS_FILENAME)
        hint = (f" A legacy {LEGACY_WEIGHTS_FILENAME} is present: convert it with "
                f"`flash_ansr convert-weights {directory}`.") if os.path.exists(legacy) else ""
        raise FileNotFoundError(f"No {WEIGHTS_FILENAME} in {directory!r}.{hint}")
    module.load_state_dict(load_file(path, device=str(device)), strict=strict)


def convert_legacy_weights(directory: str, *, overwrite: bool = False) -> str:
    """One-shot: write ``model.safetensors`` beside an existing ``state_dict.pt``.

    The ``.pt`` file is left exactly where it is -- this adds, never replaces.
    """
    legacy = os.path.join(directory, LEGACY_WEIGHTS_FILENAME)
    if not os.path.exists(legacy):
        raise FileNotFoundError(f"No {LEGACY_WEIGHTS_FILENAME} in {directory!r} to convert.")
    target = os.path.join(directory, WEIGHTS_FILENAME)
    if os.path.exists(target) and not overwrite:
        raise FileExistsError(f"{target!r} already exists; pass overwrite=True to replace it.")
    state = torch.load(legacy, weights_only=True, map_location="cpu")
    non_tensor = sorted(k for k, v in state.items() if not torch.is_tensor(v))
    if non_tensor:
        raise ValueError(
            f"{legacy!r} carries non-tensor entries {non_tensor}, which safetensors cannot "
            f"hold. Convert those separately rather than losing them silently.")
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, target)
    return target
