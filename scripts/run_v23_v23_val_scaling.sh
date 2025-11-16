#!/usr/bin/env bash
set -euo pipefail

# Runs every experiment defined in configs/evaluation/scaling/v23.0-20M_v23_val.yaml sequentially.
# Extra CLI arguments are forwarded to `flash_ansr evaluate-run` so you can override limits,
# disable resume, etc. Example:
#   ./scripts/run_v23_v23_val_scaling.sh --limit 512 --no-resume

command -v flash_ansr >/dev/null 2>&1 || {
  echo "flash_ansr CLI not found in PATH. Activate the repo's environment first." >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/configs/evaluation/scaling/v23.0-20M_v23_val.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found at ${CONFIG_FILE}." >&2
  exit 1
fi

EXPERIMENTS=(
  flash_ansr_v23_choices_00001
  flash_ansr_v23_choices_00002
  flash_ansr_v23_choices_00004
  flash_ansr_v23_choices_00008
  flash_ansr_v23_choices_00016
  flash_ansr_v23_choices_00032
  flash_ansr_v23_choices_00064
  flash_ansr_v23_choices_00128
  flash_ansr_v23_choices_00256
  flash_ansr_v23_choices_00512
  flash_ansr_v23_choices_01024
  flash_ansr_v23_choices_02048
  flash_ansr_v23_choices_04096
  flash_ansr_v23_choices_08192
  flash_ansr_v23_choices_16384
)

EXTRA_ARGS=("$@")

echo "Running ${#EXPERIMENTS[@]} experiments defined in ${CONFIG_FILE}" >&2

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  echo
  echo "==== $(date -Is) :: ${EXPERIMENT} ====" >&2
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    flash_ansr evaluate-run \
      -c "${CONFIG_FILE}" \
      --experiment "${EXPERIMENT}" \
      -v \
      "${EXTRA_ARGS[@]}"
  else
    flash_ansr evaluate-run \
      -c "${CONFIG_FILE}" \
      --experiment "${EXPERIMENT}" \
      -v
  fi
done
