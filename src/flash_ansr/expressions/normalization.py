"""Shared normalization helpers for skeleton and expression tokens.

These functions normalize variable tokens (v1/v2/x1 style) to a stable
``x{i}`` naming and convert numeric literals to ``<constant>`` tokens where
appropriate. Keep the helpers small and well-tested so different evaluation and
adapter modules can reuse identical behavior.
"""
from __future__ import annotations

import re
from typing import Any, Sequence


_VAR_TOKEN_PATTERN = re.compile(r"^[vx](\d+)$", re.IGNORECASE)


def normalize_variable_token(token: str) -> tuple[str, bool]:
    """Return (normalized_token, is_variable).

    Recognizes tokens like ``v1`` or ``x2`` and returns them as ``x{n}``.
    """
    match = _VAR_TOKEN_PATTERN.match(token)
    if match:
        return f"x{int(match.group(1))}", True
    return token, False


def normalize_skeleton(tokens: Sequence[str | Any] | None) -> list[str] | None:
    """Normalize a skeleton/prefix into a list of tokens.

    - Variable tokens (``v1``/``x1``) are normalized to ``x{n}``.
    - Numeric literals are converted to the ``"<constant>"`` placeholder.
    - Existing ``"<constant>"`` tokens are preserved.
    """
    if tokens is None:
        return None
    normalized: list[str] = []
    for token in tokens:
        token_str = str(token)
        normalized_token, is_var = normalize_variable_token(token_str)
        if is_var:
            normalized.append(normalized_token)
            continue
        if token_str in {"<constant>", "<c>"}:
            normalized.append("<constant>")
            continue
        # numeric literal -> constant placeholder
        try:
            float(token_str)
        except ValueError:
            normalized.append(token_str)
        else:
            normalized.append("<constant>")
    return normalized


def normalize_expression(tokens: Sequence[str | Any] | None) -> list[str] | None:
    """Normalize an expression/prefix while keeping numeric literals intact.

    This converts variable tokens to canonical ``x{i}`` names but leaves numeric
    literals (so the 'expression' fields can still contain float strings).
    """
    if tokens is None:
        return None
    normalized: list[str] = []
    for token in tokens:
        token_str = str(token)
        normalized_token, is_var = normalize_variable_token(token_str)
        normalized.append(normalized_token if is_var else token_str)
    return normalized


__all__ = ["normalize_variable_token", "normalize_skeleton", "normalize_expression"]
