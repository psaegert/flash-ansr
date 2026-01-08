#!/usr/bin/env python3
"""Patch the local E2E symbolicregression checkout for modern deps.

The upstream E2E repo assumes older numpy/scaler handling. This script makes the
copy under ``e2e/symbolicregression`` compatible with current numpy, avoids an
infinite loop in ``rescale_function`` when scaler params are missing, drops the
deprecated ``functorch`` dependency that conflicts with modern torch, rebuilds an
unpinned ``requirements.txt`` from ``environment.yml`` (sans functorch), emits a
``pyproject.toml`` for editable installs, and adds a `tree_idx` alias expected by
newer call sites.
"""
from __future__ import annotations

import argparse
import re
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

    pattern = r"(?m)^(?P<indent>\s*)if\s*k\s*>=\s*len\(a\):\s*\n(?P=indent)\s*continue\s*$"
    replacement = (
        "\\g<indent>if k >= len(a):\n"
        "\\g<indent>    idx += 1  # guard against missing scaler params\n"
        "\\g<indent>    continue"
    )

    if not re.search(pattern, text):
        raise PatchError("Expected rescale_function guard not found; file layout changed?")

    updated = re.sub(pattern, replacement, text, count=1)
    if updated == text:
        return False

    path.write_text(updated)
    return True


def _add_tree_idx_alias(repo_root: Path) -> bool:
    path = repo_root / "symbolicregression" / "model" / "sklearn_wrapper.py"
    if not path.exists():
        raise PatchError(f"Missing file: {path}")

    text = path.read_text()

    if "tree_idx=None" in text:
        return False

    signature = "def retrieve_tree(self, refinement_type=None, dataset_idx=0, all_trees=False, with_infos=False):"
    replacement = """def retrieve_tree(\n        self,\n        refinement_type=None,\n        dataset_idx=0,\n        all_trees=False,\n        with_infos=False,\n        tree_idx=None,\n    ):\n        # `tree_idx` aliases `dataset_idx` for newer call sites.\n        if tree_idx is not None:\n            dataset_idx = tree_idx\n\n    """

    if signature not in text:
        raise PatchError("Could not find retrieve_tree signature to patch; file layout changed?")

    updated = text.replace(signature + "\n        self.exchange_tree_features()", replacement + "        self.exchange_tree_features()", 1)
    if updated == text:
        return False

    path.write_text(updated)
    return True


def _replace_np_infty(repo_root: Path) -> bool:
    targets = [
        "symbolicregression/model/sklearn_wrapper.py",
        "symbolicregression/model/utils_wrapper.py",
        "symbolicregression/metrics.py",
        "symbolicregression/regressors.py",
        "symbolicregression/trainer.py",
    ]

    changed_any = False
    for rel_path in targets:
        path = repo_root / rel_path
        if not path.exists():
            raise PatchError(f"Missing file: {path}")
        text = path.read_text()
        if "np.infty" not in text:
            continue
        updated = text.replace("np.infty", "np.inf")
        if updated != text:
            path.write_text(updated)
            changed_any = True
    return changed_any


def _switch_to_torch_func_grad(repo_root: Path) -> bool:
    path = repo_root / "symbolicregression" / "model" / "utils_wrapper.py"
    if not path.exists():
        raise PatchError(f"Missing file: {path}")

    text = path.read_text()

    if "torch.func import grad" in text or "torch_grad" in text:
        return False

    if "from functorch import grad" not in text:
        raise PatchError("Expected functorch import not found; file layout changed?")

    updated = text.replace(
        "from functorch import grad\n",
        "try:\n    from torch.func import grad as torch_grad\nexcept Exception:\n    from functorch import grad as torch_grad\n",
        1,
    )
    updated = updated.replace("grad(objective_torch)", "torch_grad(objective_torch)")

    if updated == text:
        return False

    path.write_text(updated)
    return True


def _drop_functorch_dependency(repo_root: Path) -> bool:
    """Remove functorch pins that conflict with modern torch builds."""

    changed_any = False

    req_path = repo_root / "requirements.txt"
    if req_path.exists():
        req_text = req_path.read_text()
        req_lines = req_text.splitlines()
        filtered_req = [line for line in req_lines if not line.strip().startswith("functorch")]
        updated_req = "\n".join(filtered_req) + ("\n" if req_text.endswith("\n") else "")
        if updated_req != req_text:
            req_path.write_text(updated_req)
            changed_any = True

    env_path = repo_root / "environment.yml"
    if env_path.exists():
        env_text = env_path.read_text()
        env_lines = env_text.splitlines()
        filtered_env = [line for line in env_lines if "functorch" not in line.strip()]
        updated_env = "\n".join(filtered_env) + ("\n" if env_text.endswith("\n") else "")
        if updated_env != env_text:
            env_path.write_text(updated_env)
            changed_any = True

    return changed_any


def _rewrite_requirements_from_env(repo_root: Path) -> bool:
    """Regenerate requirements.txt from environment.yml pip section, dropping functorch and pins.

    Also appends sympytorch from its git source so installs are self-contained.
    """

    env_path = repo_root / "environment.yml"
    req_path = repo_root / "requirements.txt"

    if not env_path.exists():
        raise PatchError(f"Missing file: {env_path}")

    env_lines = env_path.read_text().splitlines()

    in_pip = False
    pip_indent = None
    pip_packages: list[str] = []
    seen: set[str] = set()

    for line in env_lines:
        stripped = line.strip()
        if not in_pip:
            if stripped == "- pip:":
                in_pip = True
                pip_indent = len(line) - len(line.lstrip())
            continue

        current_indent = len(line) - len(line.lstrip())
        if pip_indent is not None and current_indent <= pip_indent:
            # Left the pip section.
            in_pip = False
            continue

        if stripped.startswith("- "):
            pkg = stripped[2:].strip()
            if not pkg or pkg.startswith("#"):
                continue
            if pkg.startswith("functorch"):
                continue

            # Strip version specifiers (==, >=, <=, etc.) to keep requirements unpinned.
            base = re.split(r"[<>=]", pkg, maxsplit=1)[0].strip()
            if not base or base in seen:
                continue
            seen.add(base)
            pip_packages.append(base)

    if not pip_packages:
        raise PatchError("No pip dependencies found in environment.yml")

    if "sympytorch" not in seen:
        pip_packages.append("sympytorch @ git+https://github.com/pakamienny/sympytorch.git")

    rebuilt = "\n".join(pip_packages) + "\n"

    if req_path.exists():
        current = req_path.read_text()
        if current == rebuilt:
            return False

    req_path.write_text(rebuilt)
    return True


def _write_pyproject(repo_root: Path) -> bool:
    """Create a minimal pyproject.toml for editable installs if missing."""

    path = repo_root / "pyproject.toml"
    if path.exists():
        return False

    req_path = repo_root / "requirements.txt"
    if not req_path.exists():
        raise PatchError("requirements.txt not found; run rewrite step first")

    deps: list[str] = []
    for line in req_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        deps.append(stripped)

    if not deps:
        raise PatchError("No dependencies found in requirements.txt for pyproject generation")

    deps_str = "\n    \"" + "\",\n    \"".join(deps) + "\",\n"

    content = (
        "[build-system]\n"
        "requires = [\"setuptools>=61\", \"wheel\"]\n"
        "build-backend = \"setuptools.build_meta\"\n\n"
        "[project]\n"
        "name = \"symbolicregression\"\n"
        "version = \"0.0.0\"\n"
        "description = \"Meta symbolic regression baseline (packaged for flash-ansr compatibility).\"\n"
        "readme = \"README.md\"\n"
        "license = { file = \"LICENSE\" }\n"
        "authors = [{ name = \"Meta Platforms, Inc.\" }]\n"
        "requires-python = \">=3.8\"\n"
        "dependencies = [\n"
        f"{deps_str}"
        "]\n\n"
        "[tool.setuptools]\n"
        "packages = { find = { where = [\".\"], include = [\"symbolicregression*\"], exclude = [\"**/tests\", \"**/.ipynb_checkpoints\"] } }\n"
        "include-package-data = true\n\n"
        "[project.urls]\n"
        "Homepage = \"https://github.com/facebookresearch/symbolicregression\"\n"
        "Repository = \"https://github.com/facebookresearch/symbolicregression\"\n"
    )

    path.write_text(content)
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
        ("add retrieve_tree tree_idx alias", _add_tree_idx_alias),
        ("replace np.infty with np.inf", _replace_np_infty),
        ("switch to torch.func.grad where available", _switch_to_torch_func_grad),
        ("drop functorch dependency", _drop_functorch_dependency),
        ("rewrite requirements.txt from environment.yml", _rewrite_requirements_from_env),
        ("write pyproject.toml for editable install", _write_pyproject),
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
