# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-3M_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-20M_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-1B_fastsrb_support.yaml -v
# python scripts/evaluate_PySR.py  -c configs/evaluation/support_sweep/pysr_fastsrb_support.yaml  -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/nesymres_fastsrb_support.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/support_sweep/e2e_fastsrb_support.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/skeleton_pool_fastsrb_support.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-ablation-1_fastsrb_support.yaml -v
flash_ansr evaluate-run -c configs/evaluation/support_sweep/v23.0-120M-ablation-2_fastsrb_support.yaml -v