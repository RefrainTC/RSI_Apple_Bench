#!/bin/bash
# run.sh <out> <timed|refout|correct> : ISO-Bench #7822 workload, driven by runner.py (see its docstring). The engine is
# built as the upstream benchmark_prefix_caching.py builds it (+ --use-v2-block-manager, the PR's perf_command), memory
# capped at 0.85. The candidate overlay is on PYTHONPATH (P2_OVERLAY), together with the P2 audit hook (sitecustomize).
#   timed  : official load = shared 645-token prefix + REPEAT distinct questions, greedy, OL tokens each (ignore_eos);
#            metric = pass 0 wall (empty prefix cache).
#   refout : same load with top-5 logprobs (parent only; reference for the output check's tie test).
#   correct: correct.py (fixed prompt groups, 32 tokens, top-5 logprobs, cold + warm cache).
# Workload env: OL (200) REPEAT (100) WORKLOAD (shared | identical) SUFFIX_BASE (0; warm-ups use 1000).
OUT=$1; MODE=${2:-timed}; TD=$(dirname "$(readlink -f "$0")")
export OL=${OL:-200} REPEAT=${REPEAT:-100} WORKLOAD=${WORKLOAD:-shared} SUFFIX_BASE=${SUFFIX_BASE:-0}
export P2_BENCH=$T5/bench/7822_parent_benchmark_prefix_caching.py
M=RedHatAI/Meta-Llama-3-8B-Instruct-FP8
PP=${P2_OVERLAY:-}; [ -n "${P2_AUDIT_DIR:-}" ] && PP=${PP:+$PP:}$P2_AUDIT_DIR
env ${PP:+PYTHONPATH=$PP} python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" > "$OUT/import_check.txt" 2>&1
case "$MODE" in
  correct) env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/correct.py" "$P2_BENCH" $M ;;
  refout)  env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" LOGPROBS=5 python3 "$TD/runner.py" $M ;;
  *)       env ${PP:+PYTHONPATH=$PP} R4_OUT="$OUT" python3 "$TD/runner.py" $M ;;
esac
