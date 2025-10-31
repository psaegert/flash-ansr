from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from simplipy import SimpliPyEngine

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.expressions.skeleton_pool import SkeletonPool, NoValidSampleFoundError


@dataclass(frozen=True)
class PromptFeatures:
    """Container holding the symbolic constraints extracted for a prompt."""

    expression_tokens: list[str]
    complexity: int
    allowed_terms: list[list[str]]
    include_terms: list[list[str]]
    exclude_terms: list[list[str]]


@dataclass
class PromptFeatureExtractorConfig:
    """Configuration controlling randomness and term sampling."""

    max_cover_terms: int = 48
    extra_allowed_range: tuple[int, int] = (0, 3)
    include_terms_range: tuple[int, int] = (0, 3)
    exclude_terms_range: tuple[int, int] = (0, 4)
    max_random_term_attempts: int = 64


@dataclass
class _ExpressionNode:
    token: str
    children: list["_ExpressionNode"] = field(default_factory=list)

    def to_prefix(self) -> list[str]:
        tokens = [self.token]
        for child in self.children:
            tokens.extend(child.to_prefix())
        return tokens

    def walk(self) -> Iterable["_ExpressionNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


class PromptFeatureExtractor:
    """Derive prompt control attributes from prefix expressions."""

    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,
        config: PromptFeatureExtractorConfig | None = None,
        variables: list[str] | None = None,
        skeleton_pool: SkeletonPool | None = None,
    ) -> None:
        self.engine = simplipy_engine
        self.tokenizer = tokenizer
        self.config = config or PromptFeatureExtractorConfig()
        self.variables = variables or self._infer_variables(tokenizer)
        if skeleton_pool is None:
            raise ValueError("PromptFeatureExtractor now requires a SkeletonPool for random term generation.")
        self.skeleton_pool = skeleton_pool

        operator_arity = getattr(self.engine, "operator_arity_compat", None)
        if operator_arity is None:
            operator_arity = getattr(self.engine, "operator_arity", {})
        self.operator_arity: dict[str, int] = dict(operator_arity)

        self.operator_aliases = getattr(self.engine, "operator_aliases", {})
        self.operator_tokens = list(self.operator_arity.keys())

    def extract(self, expression_tokens: Sequence[str]) -> PromptFeatures:
        expression_list = list(expression_tokens)
        if not expression_list:
            raise ValueError("Expression tokens cannot be empty.")

        root, next_index = self._parse_prefix_expression(expression_list, 0)
        if next_index != len(expression_list):
            raise ValueError("Expression tokens contain trailing data beyond a valid prefix expression.")

        all_subtrees = [node.to_prefix() for node in root.walk()]
        coverage_terms = self._build_cover_terms(all_subtrees)
        additional_allowed = self._sample_additional_terms(all_subtrees)
        allowed_terms = self._deduplicate_terms(coverage_terms + additional_allowed)

        include_terms = self._deduplicate_terms(self._sample_include_terms(coverage_terms))
        exclude_terms = self._deduplicate_terms(self._sample_exclude_terms(all_subtrees))

        allowed_set = {tuple(term) for term in allowed_terms}
        include_terms = [term for term in include_terms if tuple(term) in allowed_set]
        include_set = {tuple(term) for term in include_terms}

        exclude_terms = [term for term in exclude_terms if tuple(term) not in allowed_set and tuple(term) not in include_set]
        exclude_terms = self._deduplicate_terms(exclude_terms)

        return PromptFeatures(
            expression_tokens=expression_list,
            complexity=len(expression_list),
            allowed_terms=allowed_terms,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

    # ------------------------------------------------------------------
    # Prefix expression parsing utilities
    # ------------------------------------------------------------------
    def _parse_prefix_expression(self, tokens: Sequence[str], idx: int) -> tuple[_ExpressionNode, int]:
        if idx >= len(tokens):
            raise ValueError("Unexpected end of tokens while parsing prefix expression.")

        token = tokens[idx]
        idx += 1

        normalized_token = self.operator_aliases.get(token, token)
        arity = self.operator_arity.get(normalized_token, 0)

        children: list[_ExpressionNode] = []
        for _ in range(arity):
            child, idx = self._parse_prefix_expression(tokens, idx)
            children.append(child)

        return _ExpressionNode(token, children), idx

    # ------------------------------------------------------------------
    # Term collection and sampling
    # ------------------------------------------------------------------
    def _build_cover_terms(self, subtrees: list[list[str]]) -> list[list[str]]:
        unique = self._deduplicate_terms(subtrees)
        if len(unique) <= self.config.max_cover_terms:
            return unique

        root = unique[0] if unique else []
        sorted_terms = sorted(unique, key=len)

        cover_terms: list[list[str]] = []
        seen_tokens: set[str] = set()
        for term in sorted_terms:
            cover_terms.append(term)
            seen_tokens.update(term)
            if len(cover_terms) >= self.config.max_cover_terms:
                break

        if root and root not in cover_terms:
            cover_terms[-1] = root

        return cover_terms

    def _sample_additional_terms(self, existing_terms: list[list[str]]) -> list[list[str]]:
        min_extra, max_extra = self.config.extra_allowed_range
        min_extra = max(0, min_extra)
        max_extra = max(min_extra, max_extra)
        if max_extra == 0:
            return []

        target = random.randint(min_extra, max_extra)
        if target == 0:
            return []

        existing = {tuple(term) for term in existing_terms}
        new_terms: list[list[str]] = []
        new_keys: set[tuple[str, ...]] = set()
        attempts = 0
        while len(new_terms) < target and attempts < self.config.max_random_term_attempts:
            attempts += 1
            term = self._generate_random_term()
            if term is None:
                continue
            key = tuple(term)
            if key in existing or key in new_keys:
                continue
            new_terms.append(term)
            new_keys.add(key)
        return new_terms

    def _sample_include_terms(self, coverage_terms: list[list[str]]) -> list[list[str]]:
        min_inc, max_inc = self.config.include_terms_range
        min_inc = max(0, min_inc)
        max_inc = max(min_inc, max_inc)
        if not coverage_terms or max_inc == 0:
            return []

        candidates = [term for term in coverage_terms if len(term) > 1]
        if not candidates:
            return []

        upper = min(len(candidates), max_inc)
        lower = min(min_inc, upper)
        if upper == 0:
            return []

        count = random.randint(lower, upper)
        if count == 0:
            return []

        return [list(term) for term in random.sample(candidates, count)]

    def _sample_exclude_terms(self, existing_terms: list[list[str]]) -> list[list[str]]:
        min_exc, max_exc = self.config.exclude_terms_range
        min_exc = max(0, min_exc)
        max_exc = max(min_exc, max_exc)
        if max_exc == 0:
            return []

        target = random.randint(min_exc, max_exc)
        if target == 0:
            return []

        existing = {tuple(term) for term in existing_terms}
        exclusions: list[list[str]] = []
        exclusion_keys: set[tuple[str, ...]] = set()
        attempts = 0
        while len(exclusions) < target and attempts < self.config.max_random_term_attempts:
            attempts += 1
            term = self._generate_random_term()
            if term is None:
                continue
            key = tuple(term)
            if key in existing or key in exclusion_keys:
                continue
            exclusions.append(term)
            exclusion_keys.add(key)
        return exclusions

    def _generate_random_term(self) -> list[str] | None:
        term = self._generate_term_via_skeleton_pool()
        if term is None:
            raise RuntimeError("Failed to generate a random term from the provided SkeletonPool.")
        return term

    def _generate_term_via_skeleton_pool(self) -> list[str] | None:
        attempts = 0
        max_attempts = max(1, self.config.max_random_term_attempts)

        while attempts < max_attempts:
            attempts += 1
            try:
                skeleton, _, _ = self.skeleton_pool.sample_skeleton(new=True, decontaminate=False)
            except NoValidSampleFoundError:
                continue

            tokens = list(skeleton)
            if not tokens:
                continue

            try:
                root, _ = self._parse_prefix_expression(tokens, 0)
            except ValueError:
                continue

            nodes = list(root.walk())
            if not nodes:
                continue

            selected = random.choice(nodes)
            term = selected.to_prefix()
            if term:
                return term

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate_terms(terms: Iterable[Sequence[str]]) -> list[list[str]]:
        seen: dict[tuple[str, ...], list[str]] = {}
        for term in terms:
            key = tuple(term)
            if key not in seen:
                seen[key] = list(term)
        return list(seen.values())

    @staticmethod
    def _infer_variables(tokenizer: Tokenizer) -> list[str]:
        variables: set[str] = set()
        for token in tokenizer:
            if token.startswith("x") and token[1:].isdigit():
                variables.add(token)
        return sorted(variables)