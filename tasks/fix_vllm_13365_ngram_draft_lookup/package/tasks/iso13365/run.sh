#!/bin/bash
# run.sh <out> <timed|refout|correct> : ISO-Bench #13365 workload, driven by runner.py (see its docstring).
#   timed / refout : official load (refout = the same run; calibrate.sh uses it as a second parent reference).
#   correct        : runner.py correct mode (both modes run the engine core in-process,
#                    VLLM_ENABLE_V1_MULTIPROCESSING=0, so batches are reproducible and the live engine can be read back).
# Workload env: N (32) IN (2048) OL (128) KIND (repeat|random|late) SEG (40) REPS (6) SEED_BASE (0; warm-up jobs 500).
OUT=$1; MODE=${2:-timed}; TD=$(dirname "$(readlink -f "$0")")
export N=${N:-32} IN=${IN:-2048} OL=${OL:-128} KIND=${KIND:-repeat} SEG=${SEG:-40} REPS=${REPS:-6} SEED_BASE=${SEED_BASE:-0}
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0   # engine core in-process: reproducible batches (runner.py docstring)
M=Qwen/Qwen2.5-1.5B-Instruct
PP=${P2_OVERLAY:-}; [ -n "${P2_AUDIT_DIR:-}" ] && PP=${PP:+$PP:}$P2_AUDIT_DIR
env ${PP:+PYTHONPATH=$PP} python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" > "$OUT/import_check.txt" 2>&1
case "$MODE" in
  correct) env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/runner.py" $M correct ;;
  *)       env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/runner.py" $M timed ;;
esac
