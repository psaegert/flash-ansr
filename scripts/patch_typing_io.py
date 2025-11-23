#!/usr/bin/env python3
"""Patch legacy dependencies that still expect Python <3.13 behavior.

Hydra 1.3.x and OmegaConf 2.3.0 ship ANTLR-generated files that import
`TextIO` from the removed `typing.io` namespace. Python 3.13 also enforces
``default_factory`` for dataclass fields with mutable defaults, which Hydra’s
configuration classes violate. This script fixes both issues in-place so the
packages import cleanly on Python 3.13.

Re-run this script whenever you recreate the environment or reinstall
Hydra/OmegaConf/ANTLR.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

TARGET_FILES: list[tuple[str, str]] = [
    ("antlr4", "Lexer.py"),
    ("antlr4", "Parser.py"),
    ("omegaconf", "grammar/gen/OmegaConfGrammarLexer.py"),
    ("omegaconf", "grammar/gen/OmegaConfGrammarParser.py"),
]

HYDRA_CONFIG: tuple[str, str] = ("hydra", "conf/__init__.py")

REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("from typing.io import TextIO", "from typing import TextIO"),
    ("typing.io", "typing"),
)

HYDRA_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        "        override_dirname: OverrideDirname = OverrideDirname()",
        "        override_dirname: OverrideDirname = field(default_factory=OverrideDirname)",
    ),
    (
        "    config: JobConfig = JobConfig()",
        "    config: JobConfig = field(default_factory=JobConfig)",
    ),
    (
        "    run: RunDir = RunDir()",
        "    run: RunDir = field(default_factory=RunDir)",
    ),
    (
        "    sweep: SweepDir = SweepDir()",
        "    sweep: SweepDir = field(default_factory=SweepDir)",
    ),
    (
        "    help: HelpConf = HelpConf()",
        "    help: HelpConf = field(default_factory=HelpConf)",
    ),
    (
        "    hydra_help: HydraHelpConf = HydraHelpConf()",
        "    hydra_help: HydraHelpConf = field(default_factory=HydraHelpConf)",
    ),
    (
        "    overrides: OverridesConf = OverridesConf()",
        "    overrides: OverridesConf = field(default_factory=OverridesConf)",
    ),
    (
        "    job: JobConf = JobConf()",
        "    job: JobConf = field(default_factory=JobConf)",
    ),
    (
        "    runtime: RuntimeConf = RuntimeConf()",
        "    runtime: RuntimeConf = field(default_factory=RuntimeConf)",
    ),
)


def patch_file(path: Path) -> bool:
    text = path.read_text()
    updated = text
    for old, new in REPLACEMENTS:
        updated = updated.replace(old, new)
    if updated == text:
        return False
    path.write_text(updated)
    return True


def locate_package_file(package: str, relative_path: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"Could not locate package {package}")
    package_root = Path(spec.submodule_search_locations[0])
    return package_root / relative_path


def patch_hydra_conf(path: Path) -> bool:
    text = path.read_text()
    updated = text
    for old, new in HYDRA_REPLACEMENTS:
        updated = updated.replace(old, new)
    if updated == text:
        return False
    path.write_text(updated)
    return True


def main(args: Iterable[str] | None = None) -> int:  # noqa: D401 - simple CLI
    patched = 0
    skipped = 0
    for package, rel_path in TARGET_FILES:
        try:
            file_path = locate_package_file(package, rel_path)
        except ImportError as exc:
            print(f"[skip] Could not import {package}: {exc}")
            skipped += 1
            continue
        except RuntimeError as exc:
            print(f"[skip] {exc}")
            skipped += 1
            continue
        if patch_file(file_path):
            print(f"[patch] Updated {file_path}")
            patched += 1
        else:
            print(f"[ok]     No changes needed in {file_path}")
    try:
        hydra_path = locate_package_file(*HYDRA_CONFIG)
    except ImportError as exc:
        print(f"[skip] Could not import hydra: {exc}")
        skipped += 1
    except RuntimeError as exc:
        print(f"[skip] {exc}")
        skipped += 1
    else:
        if patch_hydra_conf(hydra_path):
            print(f"[patch] Updated Hydra dataclass defaults in {hydra_path}")
            patched += 1
        else:
            print(f"[ok]     Hydra dataclass defaults already patched ({hydra_path})")
    if patched == 0:
        print(
            "No files required changes. If you're still seeing compatibility errors,\n"
            "make sure the target modules match the versions in your environment.",
        )
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
