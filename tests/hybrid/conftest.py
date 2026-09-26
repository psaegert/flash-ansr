"""Fixtures of the hybrid's tests."""
import pytest


@pytest.fixture(scope="session")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4-3", install=True)
