from typing import Any

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
    "Evaluation": "srbf.eval",
    "SkeletonPoolModel": "srbf.baselines",
    "BruteForceModel": "srbf.baselines",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _MOVED_TO_SRBF:
        module = _MOVED_TO_SRBF[name]
        raise AttributeError(
            f"`flash_ansr.{name}` moved to the `srbf` package in flash-ansr 0.6. "
            f"Install it with `pip install srbf`, then use `from {module} import {name}`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
