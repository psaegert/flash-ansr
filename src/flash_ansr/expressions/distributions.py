"""Deprecated shim. Moved to `symbolic_data.distributions` in flash-ansr 0.7."""
import sys as _sys
import warnings as _warnings

import symbolic_data.distributions as _m

_warnings.warn(
    "flash_ansr.expressions.distributions moved to symbolic_data.distributions in flash-ansr 0.7; "
    "install flash-ansr[train] (or symbolic-data) and import from symbolic_data.distributions.",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _m
