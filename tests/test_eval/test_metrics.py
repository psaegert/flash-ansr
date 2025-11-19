"""Unit tests for evaluation metric helpers."""
from __future__ import annotations

import numpy as np
import pytest

from flash_ansr.eval.metrics.bootstrap import bootstrapped_metric_ci
from flash_ansr.eval.metrics.zss import build_tree, zss_tree_edit_distance


def test_bootstrap_returns_constant_interval_when_samples_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.array([1.0, 2.0, 3.0], dtype=float)
    repeats = 5
    indices = np.tile(np.arange(len(data)), (repeats, 1))

    def fake_randint(low: int, high: int, size: tuple[int, int]) -> np.ndarray:
        assert (low, high) == (0, len(data))
        assert size == (repeats, len(data))
        return indices

    monkeypatch.setattr("numpy.random.randint", fake_randint)

    median, lower, upper = bootstrapped_metric_ci(data, np.mean, n=repeats, interval=0.9)
    expected_mean = float(np.mean(data))

    assert median == pytest.approx(expected_mean)
    assert lower == pytest.approx(expected_mean)
    assert upper == pytest.approx(expected_mean)


def test_bootstrap_handles_percentage_interval_and_nd_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    n_samples = 4
    indices = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ])

    def fake_randint(low: int, high: int, size: tuple[int, int]) -> np.ndarray:
        assert size == (n_samples, len(data))
        return indices

    monkeypatch.setattr("numpy.random.randint", fake_randint)

    def test_metric(sample) -> np.ndarray:
        return float(sample[0, 0])

    median, lower, upper = bootstrapped_metric_ci(data, test_metric, n=n_samples, interval=80)

    assert median == pytest.approx(1.0)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(2.0)


def test_build_tree_converts_prefix_to_expected_structure() -> None:
    operators = {"+": 2, "*": 2}
    tree = build_tree(["+", "x1", "*", "x2", "x3"], operators)

    assert tree.label == "+"
    left, right = tree.children
    assert left.label == "x1"
    assert right.label == "*"

    mul_left, mul_right = right.children
    assert mul_left.label == "x2"
    assert mul_right.label == "x3"


def test_zss_tree_edit_distance_is_zero_for_identical_trees() -> None:
    operators = {"+": 2, "*": 2}
    expression = ["+", "x1", "*", "x2", "x3"]

    distance = zss_tree_edit_distance(expression, list(expression), operators)

    assert distance == pytest.approx(0.0)


def test_zss_tree_edit_distance_detects_label_changes() -> None:
    operators = {"+": 2, "*": 2}

    distance = zss_tree_edit_distance(["+", "x", "y"], ["*", "x", "y"], operators)

    assert distance == pytest.approx(1.0)


def test_zss_tree_edit_distance_detects_structural_changes() -> None:
    operators = {"+": 2, "*": 2}

    distance = zss_tree_edit_distance(["+", "x", "y"], ["+", "x", "*", "y", "z"], operators)

    assert distance > 1.0
