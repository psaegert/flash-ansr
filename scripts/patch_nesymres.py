#!/usr/bin/env python3
"""Apply the local Flash-ANSR NeSymReS patches automatically.

The upstream NeSymReS repository still targets Python 3.8–3.10.  This helper keeps
our copy reproducible by rewriting the handful of places that need tweaks for
Python 3.13 + modern Hydra/OmegaConf versions:

* Ensure ``field(default_factory=...)`` is used for the ``bfgs`` dataclass field.
* Expand the ``install_requires`` section to depend on Hydra 1.3.x/OmegaConf 2.3.x.

Run this script whenever you freshly clone the NeSymReS repo or reset it to an
upstream commit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable

DEPENDENCIES = [
    "numpy",
    "sympy",
    "pandas",
    "click",
    "tqdm",
    "numexpr",
    "jsons",
    "h5py",
    "scipy",
    "dataclass_dict_convert",
    "ordered_set",
    "wandb",
    "hydra-core>=1.3.2,<1.4",
    "omegaconf>=2.3.0,<2.4",
]


class PatchError(RuntimeError):
    """Raised when a patch cannot be applied."""


def patch_dclasses(repo_root: Path) -> bool:
    path = repo_root / "src/nesymres/dclasses.py"
    if not path.exists():
        raise PatchError(f"Missing file: {path}")

    text = path.read_text()
    updated = text

    if "from dataclasses import dataclass, field" not in updated:
        updated = updated.replace(
            "from dataclasses import dataclass",
            "from dataclasses import dataclass, field",
            1,
        )

    pattern = "    bfgs: BFGSParams = BFGSParams()"
    replacement = "    bfgs: BFGSParams = field(default_factory=BFGSParams)"
    if pattern in updated:
        updated = updated.replace(pattern, replacement, 1)

    changed = updated != text
    if changed:
        path.write_text(updated)
    return changed


def patch_setup(repo_root: Path) -> bool:
    path = repo_root / "src/setup.py"
    if not path.exists():
        raise PatchError(f"Missing file: {path}")

    text = path.read_text()
    pattern = re.compile(r"(?P<indent>\s*)install_requires\s*=\s*\[(?P<body>.*?)\]", re.S)
    match = pattern.search(text)
    if not match:
        raise PatchError("Could not locate install_requires block in setup.py")

    indent = match.group("indent")
    dep_indent = indent + "    "
    deps_block = ",\n".join(f"{dep_indent}'{dep}'" for dep in DEPENDENCIES)
    replacement = f"{indent}install_requires=[\n{deps_block}\n{indent}]"

    updated = text[: match.start()] + replacement + text[match.end():]
    if updated != text:
        path.write_text(updated)
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        type=Path,
        help="Path to the NeSymReS repository clone (directory containing src/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo.expanduser().resolve()

    if not repo_root.exists():
        print(f"[error] NeSymReS repo not found at {repo_root}")
        return 1

    patches: list[tuple[str, Callable[[Path], bool]]] = [
        ("dclasses dataclass defaults", patch_dclasses),
        ("setup.py dependency pins", patch_setup),
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
    sys.exit(main())
