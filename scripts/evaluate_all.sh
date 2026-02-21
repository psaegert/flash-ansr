# flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-1B_fastsrb.yaml -v
# python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_fastsrb.yaml  -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_fastsrb.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/e2e_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/skeleton_pool_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-ablation-1_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-ablation-2_fastsrb.yaml -v

flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-1B_val.yaml -v
# python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_val.yaml  -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_val.yaml -v
# flash_ansr evaluate-run -c configs/evaluation/scaling/e2e_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/skeleton_pool_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-ablation-1_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-ablation-2_val.yaml -v