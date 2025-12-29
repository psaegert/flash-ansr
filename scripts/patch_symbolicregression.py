#!/usr/bin/env python3
"""Patch the local E2E symbolicregression checkout for modern deps.

The upstream E2E repo assumes older numpy/scaler handling. This script makes the
copy under ``e2e/symbolicregression`` compatible with current numpy and avoids an
infinite loop in ``rescale_function`` when scaler params are missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


class PatchError(RuntimeError):
    """Raised when a patch cannot be applied."""


def _remove_numpy_compat(repo_root: Path) -> bool:
    path = repo_root / "symbolicregression" / "envs" / "generators.py"
    if not path.exists():
        raise PatchError(f"Missing file: {path}")

    text = path.read_text()
    target = "from numpy.compat.py3k import npy_load_module"
    if target not in text:
        return False

    updated = text.replace(target + "\n", "")
    if updated == text:
        return False

    path.write_text(updated)
    return True


def _fix_rescale_loop(repo_root: Path) -> bool:
    path = repo_root / "symbolicregression" / "model" / "utils_wrapper.py"
    if not path.exists():
        raise PatchError(f"Missing file: {path}")

    text = path.read_text()

    # Only patch when the guard is missing.
    if "idx += 1  # guard against missing scaler params" in text:
        return False

    snippet = """
                if k>=len(a):
                    continue
    """
    replacement = """
                if k >= len(a):
                    idx += 1  # guard against missing scaler params
                    continue
    """
    if snippet.strip() not in text:
        raise PatchError("Expected rescale_function guard not found; file layout changed?")

    updated = text.replace(snippet, replacement)
    if updated == text:
        return False

    path.write_text(updated)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        type=Path,
        help="Path to the symbolicregression repo root (directory containing symbolicregression/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo.expanduser().resolve()

    if not repo_root.exists():
        print(f"[error] symbolicregression repo not found at {repo_root}")
        return 1

    patches: list[tuple[str, Callable[[Path], bool]]] = [
        ("remove numpy.compat import", _remove_numpy_compat),
        ("fix rescale_function guard", _fix_rescale_loop),
    ]

    failures = 0
    for label, func in patches:
        try:
            changed = func(repo_root)
        except PatchError as exc:
            print(f"[skip] {label}: {exc}")
            failures += 1
            continue
        status = "patched" if changed else "ok"
        print(f"[{status}] {label}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
