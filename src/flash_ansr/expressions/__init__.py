"""Deprecated shim package. The expression/data layer moved to `symbolic_data` in flash-ansr 0.7."""
import warnings as _warnings

_warnings.warn(
    "flash_ansr.expressions moved to the symbolic_data package in flash-ansr 0.7; "
    "install flash-ansr[train] (or symbolic-data) and import from symbolic_data.",
    DeprecationWarning,
    stacklevel=2,
)
from symbolic_data import SkeletonPool, NoValidSampleFoundError  # noqa: E402,F401
