"""Deprecated shim. Expression-token normalization moved to `simplipy` (>=0.3.1) in flash-ansr 0.7."""
import warnings as _warnings

_warnings.warn(
    "flash_ansr.expressions.normalization moved to simplipy (>=0.3.1) in flash-ansr 0.7; "
    "import normalize_skeleton / normalize_expression from simplipy.",
    DeprecationWarning,
    stacklevel=2,
)
from simplipy import normalize_skeleton, normalize_expression  # noqa: E402,F401
