# flash_ansr evaluate-run -c configs/evaluation/noise_sweep/v23.0-3M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/noise_sweep/v23.0-20M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/noise_sweep/v23.0-120M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/noise_sweep/v23.0-1B_fastsrb.yaml -v
# python scripts/evaluate_PySR.py  -c configs/evaluation/noise_sweep/pysr_fastsrb.yaml  -v
# flash_ansr evaluate-run -c configs/evaluation/noise_sweep/nesymres_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/noise_sweep/e2e_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/noise_sweep/skeleton_pool_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/noise_sweep/brute_force_fastsrb.yaml -v