#!/bin/bash
# run.sh <out> <timed|refout|correct> : ISO-Bench #12380 workload (runner.py) or the correctness check (correct.py).
# The candidate overlay is on PYTHONPATH (P2_OVERLAY), together with the P2 audit hook (sitecustomize).
#   timed  : official load = batch BS of IN random token ids, OL greedy tokens (ignore_eos), REPS timed reps with a
#            different prompt set each; metric = median rep latency.
#   refout : same load with top-5 logprobs (parent only; reference for the output check's tie test).
#   correct: correct.py (fixed text prompts, several batch sizes, 48 tokens, top-5 logprobs).
# Workload env: BS (8) IN (32) OL (128) REPS (10) WARM (3) SEED_BASE (0; harness warm-up jobs use 1000).
OUT=$1; MODE=${2:-timed}; TD=$(dirname "$(readlink -f "$0")")
export BS=${BS:-8} IN=${IN:-32} OL=${OL:-128} REPS=${REPS:-10} WARM=${WARM:-3} SEED_BASE=${SEED_BASE:-0}
M=RedHatAI/Meta-Llama-3-8B-Instruct-FP8
PP=${P2_OVERLAY:-}; [ -n "${P2_AUDIT_DIR:-}" ] && PP=${PP:+$PP:}$P2_AUDIT_DIR
env ${PP:+PYTHONPATH=$PP} python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" > "$OUT/import_check.txt" 2>&1
case "$MODE" in
  correct) env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/correct.py" $M ;;
  refout)  env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" LOGPROBS=5 python3 "$TD/runner.py" $M ;;
  *)       env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/runner.py" $M ;;
esac
