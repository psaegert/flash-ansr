for dataset in fastsrb val; do
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-1B_${dataset}.yaml -v
    python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_${dataset}.yaml  -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/e2e_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/skeleton_pool_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-A-U_${dataset}.yaml -v
    flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-C1-uniform_${dataset}.yaml -v
done