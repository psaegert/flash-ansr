#!/bin/bash
set -e

# Re-run C1-uniform (formerly ablation-2) evaluations that were invalid due to simplify: false (should be true)

for dataset in fastsrb val; do
    echo "=== scaling / C1-uniform / ${dataset} ==="
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-C1-uniform_${dataset}.yaml -v

    echo "=== noise_sweep / C1-uniform / ${dataset} ==="
    flash_ansr evaluate-run -c configs/evaluation/noise_sweep/v23.0-120M-C1-uniform_${dataset}.yaml -v

    echo "=== support_sweep / C1-uniform / ${dataset} ==="
    flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-C1-uniform_${dataset}.yaml -v
done
