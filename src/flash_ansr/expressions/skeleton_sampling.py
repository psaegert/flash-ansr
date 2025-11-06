"""Utilities for sampling operator skeletons independent from SkeletonPool."""
from typing import Any, Sequence

import numpy as np
from simplipy import SimpliPyEngine

from flash_ansr.expressions.structure import generate_ubi_dist


class SkeletonSampler:
    """Sample prefix skeletons using the configured operator priors."""

    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        sample_strategy: dict[str, Any],
        variables: list[str],
        operator_weights: dict[str, float],
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.sample_strategy = sample_strategy
        self.variables = variables
        self.n_variables = len(variables)
        self.operator_weights = operator_weights

        self._n_leaves = 1
        self._n_unary_operators = 1
        self._n_binary_operators = 1

        self.unary_operators = [name for name, arity in simplipy_engine.operator_arity.items() if arity == 1]
        self.binary_operators = [name for name, arity in simplipy_engine.operator_arity.items() if arity == 2]

        self.unary_operator_probs = self._build_probability_vector(self.unary_operators)
        self.binary_operator_probs = self._build_probability_vector(self.binary_operators)

        max_operators = self.sample_strategy.get("max_operators", 10)
        self.unary_binary_distribution = generate_ubi_dist(
            max_operators,
            self._n_leaves,
            self._n_unary_operators,
            self._n_binary_operators,
        )

    def _build_probability_vector(self, operators: Sequence[str]) -> np.ndarray:
        probs = np.array([self.operator_weights.get(op, 0) for op in operators], dtype=np.float64)
        return probs / probs.sum()

    def _sample_next_pos_ubi(self, n_empty_nodes: int, n_operators: int) -> tuple[int, int]:
        if n_empty_nodes >= len(self.unary_binary_distribution):
            self.unary_binary_distribution = generate_ubi_dist(
                n_empty_nodes + 1,
                self._n_leaves,
                self._n_unary_operators,
                self._n_binary_operators,
            )

        probs: list[float] = []
        for index in range(n_empty_nodes):
            probs.append(
                (self._n_leaves ** index)
                * self._n_unary_operators
                * self.unary_binary_distribution[n_empty_nodes - index][n_operators - 1]
            )
        for index in range(n_empty_nodes):
            probs.append(
                (self._n_leaves ** index)
                * self._n_binary_operators
                * self.unary_binary_distribution[n_empty_nodes - index + 1][n_operators - 1]
            )

        probabilities_list = [value / self.unary_binary_distribution[n_empty_nodes][n_operators] for value in probs]
        probabilities = np.array(probabilities_list, dtype=np.float64)

        event = np.random.choice(2 * n_empty_nodes, p=probabilities)

        arity = 1 if event < n_empty_nodes else 2
        position = event % n_empty_nodes

        return position, arity

    def _get_leaves(self, t_leaves: int) -> list[str]:
        n_unique_variables = np.random.randint(1, min(t_leaves, self.n_variables) + 1)
        unique_variables = np.random.choice(self.variables + ["<constant>"], n_unique_variables, replace=False)

        guaranteed_part = unique_variables.copy()
        remaining_part = np.random.choice(unique_variables, t_leaves - n_unique_variables, replace=True)
        all_allowed_variables = np.concatenate([guaranteed_part, remaining_part])
        np.random.shuffle(all_allowed_variables)

        return all_allowed_variables.tolist()

    def sample(self, n_operators: int) -> list[str]:
        stack: list[str | None] = [None]
        n_empty_nodes = 1
        left_leaves = 0
        total_leaves = 1

        for remaining in range(n_operators, 0, -1):
            position, arity = self._sample_next_pos_ubi(n_empty_nodes, remaining)
            if arity == 1:
                operator = np.random.choice(self.unary_operators, p=self.unary_operator_probs)
            else:
                operator = np.random.choice(self.binary_operators, p=self.binary_operator_probs)

            arity_value = self.simplipy_engine.operator_arity[operator]
            n_empty_nodes += arity_value - 1 - position
            total_leaves += arity_value - 1
            left_leaves += position

            insert_index = [index for index, value in enumerate(stack) if value is None][left_leaves]
            stack = (
                stack[:insert_index]
                + [str(operator)]
                + [None for _ in range(arity_value)]
                + stack[insert_index + 1:]
            )

        assert len([1 for value in stack if value in self.simplipy_engine.operator_arity]) == n_operators
        assert len([1 for value in stack if value is None]) == total_leaves

        leaves = self._get_leaves(t_leaves=total_leaves)
        assert len(leaves) == total_leaves, f"Expected {total_leaves} leaves, got {len(leaves)}"

        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is None:
                stack = stack[:index] + [leaves.pop()] + stack[index + 1:]
        assert len(leaves) == 0

        return stack  # type: ignore[return-value]
