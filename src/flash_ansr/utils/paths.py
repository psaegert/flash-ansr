"""Filesystem helpers for locating project assets."""
import os


#: Environment variable that overrides the project root resolved by :func:`get_root` (and therefore
#: by :func:`get_path` / :func:`substitute_root_path`). When set to a non-empty value, asset lookups
#: resolve against it instead of the source-checkout root. This lets a separate repository that
#: imports ``flash_ansr`` (e.g. the planned ``srbf`` / research repos) -- or any deployment whose
#: assets live outside the installed package tree -- anchor lookups at its own tree. See
#: REPO_SPLIT_PLAN.md section 10.
ROOT_ENV_VAR = "FLASH_ANSR_ROOT"

#: Default project root for a source checkout: ``<repo>/``. This module lives at
#: ``<repo>/src/flash_ansr/utils/paths.py``, i.e. three directories below the repo root.
_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def get_root() -> str:
    """Return the project root that asset paths resolve against.

    Honours the ``FLASH_ANSR_ROOT`` environment variable (:data:`ROOT_ENV_VAR`) when it is set to a
    non-empty value, otherwise returns the source-checkout root (:data:`_DEFAULT_ROOT`). Resolving the
    root through this single function is what lets the repository split point each repo's lookups at
    its own tree without touching the many :func:`get_path` call sites.
    """
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return os.path.abspath(override)
    return _DEFAULT_ROOT


def normalize_path_preserve_leading_dot(path: str) -> str:
    """Normalise ``path`` while preserving a leading ``./`` when present."""
    starts_with_dot_sep = path.startswith(f'.{os.sep}')
    normalized_path = os.path.normpath(path)
    if (
        starts_with_dot_sep
        and not os.path.isabs(normalized_path)
        and not normalized_path.startswith('..')
        and normalized_path != '.'
    ):
        return f'.{os.sep}{normalized_path}'
    return normalized_path


def get_path(*args: str, filename: str | None = None, create: bool = False) -> str:
    """Resolve a path relative to the project root (see :func:`get_root`).

    Optionally creates the directories leading to the resolved path when ``create`` is set.
    """
    if any(not isinstance(arg, str) for arg in args):
        raise TypeError("All arguments must be strings.")

    path = normalize_path_preserve_leading_dot(
        os.path.join(get_root(), *args, filename or '')
    )

    if create:
        if filename is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    return os.path.abspath(path)


def substitute_root_path(path: str) -> str:
    """Replace ``{{ROOT}}`` placeholders with the project root (see :func:`get_root`)."""
    return path.replace(r"{{ROOT}}", get_root())
