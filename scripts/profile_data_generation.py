#!/usr/bin/env python3
"""Utility to profile Flash-ANSR data generation with and without preprocessing."""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import time
from typing import Iterable, Sequence

import pstats

from flash_ansr.data import FlashANSRDataset


def _run_iteration(
    config: str,
    *,
    steps: int,
    batch_size: int,
    preprocess: bool,
    preprocess_in_worker: bool,
    verbose: bool,
) -> tuple[float, pstats.Stats, io.StringIO]:
    dataset = FlashANSRDataset.from_config(config)
    profile = cProfile.Profile()
    dataset_iter = dataset.iterate(
        steps=steps,
        batch_size=batch_size,
        preprocess=preprocess,
        preprocess_in_worker=preprocess_in_worker,
        verbose=verbose,
    )

    start = time.perf_counter()
    profile.enable()
    for _ in dataset_iter:
        pass
    profile.disable()
    elapsed = time.perf_counter() - start

    # Always shut down to release subprocesses between runs.
    with contextlib.suppress(Exception):
        dataset.shutdown()

    stats_stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stats_stream)
    return elapsed, stats, stats_stream


def _print_stats(
    header: str,
    elapsed: float,
    stats: pstats.Stats,
    stats_stream: io.StringIO,
    *,
    sort: str,
    limit: int,
) -> None:
    print(f"\n=== {header} ===")
    print(f"Total wall time: {elapsed:.2f}s")
    stats.strip_dirs().sort_stats(sort).print_stats(limit)
    print(stats_stream.getvalue())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to dataset/train configuration file")
    parser.add_argument("--steps", type=int, default=10, help="Number of batches to generate per run")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to use during iteration")
    parser.add_argument(
        "--modes",
        choices=["off", "on", "both"],
        default="both",
        help="Which preprocessing modes to profile",
    )
    parser.add_argument("--sort", default="cumtime", help="pstats sorting key")
    parser.add_argument("--top", type=int, default=25, help="Number of rows to display from profiling output")
    parser.add_argument(
        "--worker-preprocess",
        action="store_true",
        help="Run preprocessing inside producer workers (requires --modes on/both)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable iterator progress output")

    parsed_args: Sequence[str] | None
    if argv is None:
        parsed_args = None
    else:
        parsed_args = list(argv)

    args = parser.parse_args(parsed_args)

    modes: list[tuple[str, bool]]
    match args.modes:
        case "off":
            modes = [("preprocess=False", False)]
        case "on":
            modes = [("preprocess=True", True)]
        case "both":  # default
            modes = [("preprocess=False", False), ("preprocess=True", True)]
        case _:
            raise AssertionError("Unhandled profiling mode")

    for label, preprocess in modes:
        elapsed, stats, stats_stream = _run_iteration(
            args.config,
            steps=args.steps,
            batch_size=args.batch_size,
            preprocess=preprocess,
            preprocess_in_worker=args.worker_preprocess and preprocess,
            verbose=args.verbose,
        )
    header = f"{label} (worker_preprocess={'True' if args.worker_preprocess and preprocess else 'False'})"
    _print_stats(header, elapsed, stats, stats_stream, sort=args.sort, limit=args.top)

    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
