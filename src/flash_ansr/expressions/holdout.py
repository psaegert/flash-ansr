"""Deprecated shim. Moved to `symbolic_data.holdout` in flash-ansr 0.7."""
import sys as _sys
import warnings as _warnings

import symbolic_data.holdout as _m

_warnings.warn(
    "flash_ansr.expressions.holdout moved to symbolic_data.holdout in flash-ansr 0.7; "
    "install flash-ansr[train] (or symbolic-data) and import from symbolic_data.holdout.",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _m
