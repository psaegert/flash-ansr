flash_ansr evaluate-run -c configs/evaluation/v23.0-3M_fastsrb_long.yaml -v
flash_ansr evaluate-run -c configs/evaluation/v23.0-20M_fastsrb_long.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_fastsrb_long.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/nesymres_fastsrb_long.yaml -v