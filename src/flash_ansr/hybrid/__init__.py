"""The hybrid: Flash-ANSR seeds PySR; PySR's hall of fame joins Flash-ANSR's candidate pool; Flash-ANSR's ranking picks.

    from flash_ansr import FlashANSR
    from flash_ansr.hybrid import HybridRegressor

    hybrid = HybridRegressor(FlashANSR.load(...))      # the r* = 0.5 default: 1024 draws, 64 PySR iterations
    result = hybrid.fit(X, y)                           # a HybridFitResult: the extended pool, best first
    result.get_expression(), result.best.source         # 'flash-ansr' or 'pysr'

PySR is an optional dependency (``pip install flash-ansr[pysr]``); Julia is fetched on its first use.
"""
from flash_ansr.hybrid.bridge import data_fingerprint, evaluate_prefix, prefix_to_julia
from flash_ansr.hybrid.candidates import REFINE_DEFAULTS, pick_prediction, pysr_candidates, rank_candidates, refine_settings
from flash_ansr.hybrid.clock import ClockState, RunningMean
from flash_ansr.hybrid.pysr_model import PySRSettings
from flash_ansr.hybrid.regressor import (DEFAULT_DRAWS, R_STAR_LADDER, HybridCandidate, HybridConfig, HybridFitResult,
                                         HybridRegressor, r_star_iterations)

__all__ = [
    "HybridRegressor", "HybridConfig", "HybridFitResult", "HybridCandidate", "PySRSettings", "R_STAR_LADDER", "DEFAULT_DRAWS",
    "r_star_iterations", "ClockState", "RunningMean", "REFINE_DEFAULTS", "data_fingerprint", "evaluate_prefix",
    "prefix_to_julia", "pick_prediction", "pysr_candidates", "rank_candidates", "refine_settings",
]
