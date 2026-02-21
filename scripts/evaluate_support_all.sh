# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-3M_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-20M_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-1B_fastsrb.yaml -v
# python scripts/evaluate_PySR.py  -c configs/evaluation/support_sweep/pysr_fastsrb.yaml  -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/nesymres_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/e2e_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/skeleton_pool_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-ablation-1_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-ablation-2_fastsrb.yaml -v  # TODO

flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-3M_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-20M_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-1B_val.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/support_sweep/pysr_val.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/nesymres_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/e2e_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/skeleton_pool_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-ablation-1_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-ablation-2_val.yaml -v