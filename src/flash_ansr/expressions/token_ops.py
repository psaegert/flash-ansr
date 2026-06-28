"""Deprecated shim. Moved to `symbolic_data.token_ops` in flash-ansr 0.7."""
import sys as _sys
import warnings as _warnings

import symbolic_data.token_ops as _m

_warnings.warn(
    "flash_ansr.expressions.token_ops moved to symbolic_data.token_ops in flash-ansr 0.7; "
    "install flash-ansr[train] (or symbolic-data) and import from symbolic_data.token_ops.",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _m
