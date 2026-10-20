"""Fixtures of the hybrid's tests."""
import os

import pytest


@pytest.fixture(scope="session")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4-3", install=True)


@pytest.fixture(autouse=True)
def julia_operator_definitions(monkeypatch):
    """Without Julia (the default), defining the custom operators records the code instead of starting Julia; every
    test starts with nothing defined. Under FLASH_ANSR_HYBRID_JULIA=1 the real definitions run (once per process)."""
    import flash_ansr.hybrid.pysr_model as pm
    if os.environ.get("FLASH_ANSR_HYBRID_JULIA") == "1":
        yield None
        return
    evaluated = []
    monkeypatch.setattr(pm, "_seval", evaluated.append)
    monkeypatch.setattr(pm, "_OPERATORS_DEFINED", False)
    yield evaluated
