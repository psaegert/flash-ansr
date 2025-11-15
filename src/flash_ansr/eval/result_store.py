"""Utility helpers for accumulating and persisting evaluation outputs."""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Mapping

from flash_ansr.utils.paths import substitute_root_path


class ResultStore:
    """Dictionary-of-lists accumulator with persistence helpers."""

    def __init__(self, initial: Mapping[str, Iterable[Any]] | None = None) -> None:
        self._store: DefaultDict[str, list[Any]] = defaultdict(list)
        if initial is not None:
            self.extend(initial)

    @property
    def size(self) -> int:
        lengths = {len(values) for values in self._store.values()}
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise ValueError("ResultStore is in an inconsistent state")
        return lengths.pop()

    def extend(self, records: Mapping[str, Iterable[Any]]) -> None:
        snapshots = {key: list(values) for key, values in records.items()}
        lengths = {len(values) for values in snapshots.values()}
        if lengths and len(lengths) != 1:
            raise ValueError("Existing results have inconsistent lengths")
        for key, values in snapshots.items():
            self._store[key].extend(values)

    def append(self, record: Mapping[str, Any]) -> None:
        for key, value in record.items():
            self._store[key].append(value)
        self._validate_lengths()

    def snapshot(self) -> Dict[str, list[Any]]:
        return {key: list(values) for key, values in self._store.items()}

    def save(self, path: str | Path) -> None:
        resolved = Path(substitute_root_path(str(path)))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("wb") as handle:
            pickle.dump(self.snapshot(), handle)

    def _validate_lengths(self) -> None:
        lengths = [len(values) for values in self._store.values()]
        if lengths and len(set(lengths)) != 1:
            raise ValueError("ResultStore lists must maintain identical lengths")


__all__ = ["ResultStore"]
