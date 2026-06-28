#!/bin/bash
# Generate a held-out validation skeleton pool for a config bundle.
#
# Pool generation lives in the symbolic-data data layer (`symbolic_data.SkeletonPool`):
# flash-ansr 0.7 consumes skeleton pools and no longer ships a data CLI. A first-class
# `symbolic-data` CLI is planned for symbolic-data 0.2; until then this wraps the Python API.
# Run from the project root so the relative config/output paths resolve.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: generate_validation_set.sh <CONFIG>"
    exit 1
fi
CONFIG="$1"

python - "$CONFIG" <<'PY'
import sys
from symbolic_data import SkeletonPool

config = f"./configs/{sys.argv[1]}/skeleton_pool_val.yaml"
output = f"./data/ansr-data/{sys.argv[1]}/skeleton_pool_val"

pool = SkeletonPool.from_config(config)
pool.create(size=1000, verbose=True)  # 1000 skeletons for validation
pool.save(output, config=config)
print(f"Saved validation skeleton pool to {output}")
PY
