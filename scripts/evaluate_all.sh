# FastSRB
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_fastsrb.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_fastsrb.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_fastsrb.yaml -v

# v23_val
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_v23_val.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_v23_val.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_v23_val.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_v23_val.yaml -v

# FastSRB Long
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_fastsrb_long.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb_long.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_fastsrb_long.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_fastsrb_long.yaml -v

# v23_val Long
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-3M_v23_val_long.yaml -v
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_v23_val_long.yaml -v
python scripts/evaluate_PySR.py  -c configs/evaluation/scaling/pysr_v23_val_long.yaml  -v
flash_ansr evaluate-run -c configs/evaluation/scaling/nesymres_v23_val_long.yaml -v