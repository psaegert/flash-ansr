"""Utility helpers for accumulating and persisting evaluation outputs."""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Mapping

from flash_ansr.utils.paths import substitute_root_path


_REQUIRED_RESULT_FIELDS: dict[str, Any] = {
    "placeholder": False,
    "placeholder_reason": None,
}


class ResultStore:
    """Dictionary-of-lists accumulator with persistence helpers."""

    def __init__(self, initial: Mapping[str, Iterable[Any]] | None = None) -> None:
        self._store: DefaultDict[str, list[Any]] = defaultdict(list)
        self._required_fields: Mapping[str, Any] = dict(_REQUIRED_RESULT_FIELDS)
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
        target_len = lengths.pop() if lengths else 0
        current_size = self.size
        snapshots = self._ensure_snapshot_defaults(snapshots, target_len)
        for existing_key in self._store.keys():
            snapshots.setdefault(existing_key, [None for _ in range(target_len)])
        for key, values in snapshots.items():
            if key not in self._store:
                self._store[key] = [None for _ in range(current_size)]
        for key, values in snapshots.items():
            self._store[key].extend(values)
        self._validate_lengths()

    def append(self, record: Mapping[str, Any]) -> None:
        normalized = self._ensure_record_defaults(dict(record))
        current_size = self.size
        for existing_key in self._store.keys():
            normalized.setdefault(existing_key, None)
        for key in normalized.keys():
            if key not in self._store:
                self._store[key] = [None for _ in range(current_size)]
        for key, value in normalized.items():
            self._store[key].append(value)
        self._validate_lengths()

    def snapshot(self) -> Dict[str, list[Any]]:
        return {key: list(values) for key, values in self._store.items()}

    def save(self, path: str | Path) -> None:
        resolved = Path(substitute_root_path(str(path)))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("wb") as handle:
            pickle.dump(self.snapshot(), handle)

    def statistics(self) -> Dict[str, Any]:
        """Return aggregate counts for valid results and placeholders."""

        total = self.size
        placeholder_flags = self._store.get("placeholder", [])
        placeholder_count = sum(1 for flag in placeholder_flags if flag)
        placeholder_reasons: Counter[str] = Counter()
        if placeholder_count:
            reason_values = self._store.get("placeholder_reason", [])
            for flag, reason in zip(placeholder_flags, reason_values):
                if flag:
                    placeholder_reasons[str(reason or "unspecified")] += 1

        return {
            "total": total,
            "valid": total - placeholder_count,
            "placeholders": placeholder_count,
            "placeholder_reasons": dict(placeholder_reasons),
        }

    def _validate_lengths(self) -> None:
        lengths = [len(values) for values in self._store.values()]
        if lengths and len(set(lengths)) != 1:
            raise ValueError("ResultStore lists must maintain identical lengths")

    def _ensure_snapshot_defaults(
        self,
        snapshots: Dict[str, list[Any]],
        target_len: int,
    ) -> Dict[str, list[Any]]:
        if target_len <= 0:
            return snapshots
        for field, default in self._required_fields.items():
            if field not in snapshots:
                snapshots[field] = [default for _ in range(target_len)]
        return snapshots

    def _ensure_record_defaults(self, record: Dict[str, Any]) -> Dict[str, Any]:
        for field, default in self._required_fields.items():
            record.setdefault(field, default)
        return record


__all__ = ["ResultStore"]
