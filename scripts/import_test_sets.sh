#!/bin/bash
# Import benchmark equations into a holdout skeleton pool (training-time decontamination).
#
# STATUS: pending symbolic-data 0.2.
#
# This step parsed a raw benchmark spec (the FastSRB `expressions.yaml`, or CSV sets such as
# Feynman/Nguyen) into a skeleton pool, which training then referenced under `holdout_pools:` so
# the data-generating process never samples a test skeleton. The raw-spec ingest (ParserFactory)
# was part of flash-ansr's data CLI, which moved to the symbolic-data data layer in flash-ansr 0.7.
# symbolic-data 0.1.0 ships the data API (SkeletonPool / sampling) but NOT yet the ingest CLI;
# that lands in symbolic-data 0.2.
#
# In the meantime there is NO interim command that rebuilds a benchmark holdout pool:
#   * If you already have a benchmark holdout pool on disk, point holdout_pools: at it, e.g.
#       holdout_pools: ["./data/ansr-data/test_set/fastsrb/skeleton_pool"]
#     (this is what the shipped configs/v23.* bundles do). generate_test_set.sh samples RANDOM
#     skeletons, not the benchmark equations, so it is NOT a substitute.
#   * Otherwise train with holdout_pools: [] -- fine for non-benchmark use, but benchmark eval
#     numbers (e.g. FastSRB/Feynman) will be contaminated until you can decontaminate.
#   * symbolic_data.load_benchmark('fastsrb') provides the FastSRB benchmark for EVALUATION
#     (used by srbf); it is an (X, y) sampler and does NOT build a training-holdout pool.
#
# Track: https://github.com/psaegert/symbolic-data  (ingest CLI -> symbolic-data 0.2)

echo "import_test_sets: raw-spec ingest (ParserFactory) is pending symbolic-data 0.2." >&2
echo "Interim: point holdout_pools: at a pre-built pool (see the comments in this script)." >&2
exit 1
