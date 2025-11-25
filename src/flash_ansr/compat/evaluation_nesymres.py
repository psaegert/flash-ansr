from typing import Any, Callable

from simplipy import SimpliPyEngine

from flash_ansr import FlashANSRDataset
from flash_ansr.eval.data_sources import SkeletonDatasetSource
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.model_adapters import NeSymReSAdapter
from flash_ansr.eval.result_store import ResultStore
from flash_ansr.utils.config_io import load_config

from nesymres.architectures.model import Model  # type: ignore[import]


class NeSymReSEvaluation():
    def __init__(
            self,
            n_support: int | None = None,
            noise_level: float = 0.0,
            beam_width: int | None = None,
            device: str = 'cpu',
            remove_padding: bool = True) -> None:

        self.n_support = n_support
        self.noise_level = noise_level
        self.beam_width = beam_width
        self.remove_padding = remove_padding

        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "NeSymReSEvaluation":
        config_ = load_config(config)

        if "evaluation" in config_.keys():
            config_ = config_["evaluation"]

        return cls(
            n_support=config_["n_support"],
            noise_level=config_.get("noise_level", 0.0),
            beam_width=config_.get("beam_width"),
            device=config_["device"],
            remove_padding=config_.get("remove_padding", True),
        )

    def evaluate(
            self,
            model: Model,
            fitfunc: Callable,
            simplipy_engine: SimpliPyEngine,
            dataset: FlashANSRDataset,
            size: int | None = None,
            verbose: bool = True) -> dict[str, Any]:

        if size is None:
            size = len(dataset.skeleton_pool)

        store = ResultStore()

        data_source = SkeletonDatasetSource(
            dataset,
            target_size=size,
            n_support=self.n_support,
            noise_level=self.noise_level,
            device=self.device,
        )

        adapter = NeSymReSAdapter(
            model=model,
            fitfunc=fitfunc,
            simplipy_engine=simplipy_engine,
            device=self.device,
            beam_width=self.beam_width,
            remove_padding=self.remove_padding,
        )

        engine = EvaluationEngine(
            data_source=data_source,
            model_adapter=adapter,
            result_store=store,
        )

        results = engine.run(limit=size, verbose=verbose, progress=verbose)

        return dict(sorted(results.items()))
