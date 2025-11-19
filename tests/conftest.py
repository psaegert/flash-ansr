"""Global pytest fixtures and dependency shims for the flash-ansr test suite."""
from __future__ import annotations

import sys
import types


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_nesymres_stub() -> None:
    """Provide a lightweight stand-in for the optional nesymres dependency."""
    root = _ensure_module("nesymres")
    architectures = _ensure_module("nesymres.architectures")
    architectures.__dict__.setdefault("__path__", [])

    model_module = _ensure_module("nesymres.architectures.model")

    if not hasattr(model_module, "Model"):
        class _DummyModel:  # pragma: no cover - trivial shim
            pass

        model_module.Model = _DummyModel  # type: ignore[attr-defined]

    architectures.model = model_module  # type: ignore[attr-defined]
    root.architectures = architectures  # type: ignore[attr-defined]


_install_nesymres_stub()
