#!/bin/bash
# run.sh <out> <timed|refout|correct> : ISO-Bench #14930 workload (V1, n-gram spec decode), driven by runner.py / correct.py
# (see their docstrings). The candidate overlay is on PYTHONPATH (P2_OVERLAY) together with the P2 audit hook.
#   timed  : official load (runner.py defaults) or a transfer load (workload env N IN OL K TEMP REPS MIX PRIME SEED_BASE)
#   refout : same as timed (greedy outputs carry no logprobs here; kept for calibrate.sh's OUTREF step)
#   correct: correct.py (greedy ids + sampled statistics on a fixed prompt set)
OUT=$1; MODE=${2:-timed}; TD=$(dirname "$(readlink -f "$0")")
M=Qwen/Qwen2.5-1.5B-Instruct
export VLLM_USE_V1=1
PP=${P2_OVERLAY:-}; [ -n "${P2_AUDIT_DIR:-}" ] && PP=${PP:+$PP:}$P2_AUDIT_DIR
env ${PP:+PYTHONPATH=$PP} python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" > "$OUT/import_check.txt" 2>&1
case "$MODE" in
  correct) env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/correct.py" $M ;;
  *)       env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/runner.py" $M ;;
esac
