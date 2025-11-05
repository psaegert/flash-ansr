"""Filesystem helpers for locating project assets."""
import os


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
    """Resolve a path relative to the repository root (optionally creating directories)."""
    if any(not isinstance(arg, str) for arg in args):
        raise TypeError("All arguments must be strings.")

    path = normalize_path_preserve_leading_dot(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', *args, filename or '')
    )

    if create:
        if filename is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    return os.path.abspath(path)


def substitute_root_path(path: str) -> str:
    """Replace ``{{ROOT}}`` placeholders with the project root."""
    return path.replace(r"{{ROOT}}", get_path())
