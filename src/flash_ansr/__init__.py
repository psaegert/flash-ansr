from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

try:
    __version__ = _pkg_version("flash-ansr")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0+unknown"

from .model import (
    ModelFactory,
    FlashANSRModel,
    SetTransformer,
    Tokenizer,
    RotaryEmbedding,
    IEEE75432PreEncoder,
    IEEE75416PreEncoder,
    install_model,
    remove_model,
)
from symbolic_data import LampleChartonCatalog, NoValidSampleFoundError
from .utils import (
    GenerationConfig,
    GenerationConfigBase,
    BeamSearchConfig,
    SoftmaxSamplingConfig,
    MCTSGenerationConfig,
    create_generation_config,
    get_path,
    get_root,
    load_config,
    save_config,
    substitute_root_path,
)
from .refine import Refiner, ConvergenceError
from .flash_ansr import FlashANSR
from .data.data import FlashANSRDataset
from .preprocessing import FlashANSRPreprocessor


# v0.6: the evaluation framework (``Evaluation``) and comparison baselines
# (``SkeletonPoolModel``, ``BruteForceModel``) moved to the ``srbf`` package
# (https://github.com/psaegert/srbf). A PEP 562 module ``__getattr__`` keeps old
# imports discoverable by raising a helpful, actionable error instead of a bare
# ``ImportError`` / ``AttributeError``.
_MOVED_TO_SRBF = {
    "Evaluation": ("srbf", "Benchmark"),
    "SkeletonPoolModel": ("srbf.baselines", "LampleChartonModel"),
    "BruteForceModel": ("srbf.baselines", "BruteForceModel"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _MOVED_TO_SRBF:
        module, new_name = _MOVED_TO_SRBF[name]
        raise AttributeError(
            f"`flash_ansr.{name}` moved to the `srbf` package (now `{new_name}`). "
            f"Install it with `pip install srbf`, then use `from {module} import {new_name}`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
