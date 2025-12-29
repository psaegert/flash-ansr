# FastSRB
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M_fastsrb.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_fastsrb.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/skeleton_pool_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/brute_force_fastsrb.yaml -v
PYTHONPATH="$(pwd)/e2e/symbolicregression:${PYTHONPATH}" python -m flash_ansr evaluate-run -c configs/evaluation/scaling/e2e_fastsrb.yaml -v

# v23_val
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_v23_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_v23_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M_v23_val.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_v23_val.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_v23_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/skeleton_pool_v23_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/brute_force_v23_val.yaml -v