# FastSRB
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-3M_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-20M_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/nesymres_fastsrb_support.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/support_sweep/pysr_fastsrb_support.yaml  -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/skeleton_pool_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/brute_force_fastsrb_support.yaml -v