"""The clock's running costs: what PySR's share pays besides the search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["RunningMean", "ClockState"]


class RunningMean:
    """A running mean seeded with a prior value (the configured estimate counts as one observation);
    finite observations only."""

    def __init__(self, initial: float, count: int = 1):
        self.value = float(initial)
        self.count = max(1, int(count))

    def observe(self, x: float | None) -> None:
        if x is None or not np.isfinite(x):
            return
        self.count += 1
        self.value += (float(x) - self.value) / self.count


class ClockState:
    """The two costs PySR's share pays besides the search, as running means over EVERY problem run so
    far -- they are properties of the machine, not of a cell -- persisted in a JSON file (one cell runs
    at a time): PySR's fixed cost above its timeout (the Julia dispatch before the search clock starts,
    the return after it) and the pricing of the candidates PySR adds (refit, ladder, MDL, score)."""

    def __init__(self, path: Path | str | None, pysr_overhead_s: float, pricing_reserve_s: float):
        self.path = Path(path) if path is not None else None
        self.pysr_overhead = RunningMean(pysr_overhead_s)
        self.pricing_reserve = RunningMean(pricing_reserve_s)
        if self.path is not None and self.path.exists():
            try:
                with self.path.open() as fh:
                    stored = json.load(fh)
                self.pysr_overhead = RunningMean(stored["pysr_overhead"]["value"], stored["pysr_overhead"]["count"])
                self.pricing_reserve = RunningMean(stored["pricing_reserve"]["value"], stored["pricing_reserve"]["count"])
            except Exception:  # noqa: BLE001 - an unreadable state file: start from the configured seeds
                pass

    def save(self) -> None:
        if self.path is None:
            return
        payload = {"pysr_overhead": {"value": self.pysr_overhead.value, "count": self.pysr_overhead.count},
                   "pricing_reserve": {"value": self.pricing_reserve.value, "count": self.pricing_reserve.count}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(payload, fh)
        tmp.replace(self.path)
