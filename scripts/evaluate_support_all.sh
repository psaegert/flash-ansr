# FastSRB
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-3M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-20M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/nesymres_fastsrb.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/support_sweep/pysr_fastsrb.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/skeleton_pool_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/brute_force_fastsrb.yaml -v