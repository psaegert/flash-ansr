"""Deprecated shim. Moved to `symbolic_data.skeleton_sampling` in flash-ansr 0.7."""
import sys as _sys
import warnings as _warnings

import symbolic_data.skeleton_sampling as _m

_warnings.warn(
    "flash_ansr.expressions.skeleton_sampling moved to symbolic_data.skeleton_sampling in flash-ansr 0.7; "
    "install flash-ansr[train] (or symbolic-data) and import from symbolic_data.skeleton_sampling.",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _m
