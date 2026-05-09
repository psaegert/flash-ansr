#!/bin/bash
# Mirror the ablation-{1,2} → A-U / C1-uniform rename on solomon, after the
# rename has been pushed and pulled on this machine.
#
# Tracked file renames (configs, eval YAMLs, scripts) come down via git pull;
# this script handles the untracked artifacts that aren't in git: the model
# directory and the per-sweep result directories. Idempotent and safe to re-run
# (mv silently skips if the source dir is already gone).
#
# Usage: bash scripts/rename_ablations_solomon.sh
set -e

ROOT=$(pwd)

rename_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
        if [ -e "$dst" ]; then
            echo "[skip] $dst already exists; not overwriting (delete it first if you want to re-rename)"
        else
            echo "[mv]   $src -> $dst"
            mv "$src" "$dst"
        fi
    else
        echo "[skip] $src not present"
    fi
}

echo "=== model dirs ==="
rename_if_exists "${ROOT}/models/ansr-models/v23.0-120M-ablation-1" \
                 "${ROOT}/models/ansr-models/v23.0-120M-A-U"
rename_if_exists "${ROOT}/models/ansr-models/v23.0-120M-ablation-2" \
                 "${ROOT}/models/ansr-models/v23.0-120M-C1-uniform"

echo ""
echo "=== result dirs ==="
for sweep in scaling noise_sweep support_sweep; do
    rename_if_exists "${ROOT}/results/evaluation/${sweep}/v23.0-120M-ablation-1" \
                     "${ROOT}/results/evaluation/${sweep}/v23.0-120M-A-U"
    rename_if_exists "${ROOT}/results/evaluation/${sweep}/v23.0-120M-ablation-2" \
                     "${ROOT}/results/evaluation/${sweep}/v23.0-120M-C1-uniform"
done

echo ""
echo "Done. Tracked-file renames (configs, eval YAMLs, scripts) come via 'git pull'."
