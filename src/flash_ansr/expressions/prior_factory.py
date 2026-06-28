"""Deprecated shim. Moved to `symbolic_data.prior_factory` in flash-ansr 0.7."""
import sys as _sys
import warnings as _warnings

import symbolic_data.prior_factory as _m

_warnings.warn(
    "flash_ansr.expressions.prior_factory moved to symbolic_data.prior_factory in flash-ansr 0.7; "
    "install flash-ansr[train] (or symbolic-data) and import from symbolic_data.prior_factory.",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _m
