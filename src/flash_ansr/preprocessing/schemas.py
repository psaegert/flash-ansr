"""Shared dataclasses for preprocessing components."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPrefix:
    """Tokens and metadata that form the prompt prefix."""

    tokens: list[int]
    numeric: list[float]
    mask: list[bool]
    metadata: dict[str, list[list[str]]]


@dataclass(frozen=True)
class PromptFeatures:
    """Symbolic constraints extracted from an expression skeleton."""

    expression_tokens: list[str]
    complexity: int
    allowed_terms: list[list[str]]
    include_terms: list[list[str]]
    exclude_terms: list[list[str]]
