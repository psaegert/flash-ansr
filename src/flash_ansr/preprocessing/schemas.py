"""Shared dataclasses for preprocessing components."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPrefix:
    """Tokens and metadata that form the prompt prefix."""

    tokens: list[int]
    numeric: list[float]
    mask: list[bool]
    metadata: dict[str, list[list[str]]]
