#!/usr/bin/env bash
# One verifier pass = the task's tests/test.sh minus its GPU reap (same _bench_rounds.py arguments), on one 4090.
# usage: RSI_GPU=<gpu> RSI_EXTRA_SERVE_ARGS="..." run_phase.sh <ws name> <result name> [rounds]
# exit 3 = GPU busy before start, 4 = aborted by the foreign-process guard
[ -z "${RSI_CLEAN:-}" ] && exec env -i RSI_CLEAN=1 RSI_GPU="${RSI_GPU:?}" RSI_EXTRA_SERVE_ARGS="${RSI_EXTRA_SERVE_ARGS-}" RSI_HARNESS_DIR="${RSI_HARNESS_DIR-}" RSI_ROOT="${RSI_ROOT-}" RSI_OUT="${RSI_OUT-}" RSI_TASK_DIR="${RSI_TASK_DIR-}" RSI_CUDA_HOME="${RSI_CUDA_HOME-}" RSI_ST_MODEL="${RSI_ST_MODEL-}" RSI_CPU_MAP="${RSI_CPU_MAP-}" RSI_ARCH="${RSI_ARCH-}" PATH=/usr/bin:/bin bash "$0" "$@"
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source $S/env.sh
ws=$RSI_WORK/$1; art=$RSI_RES/$2; rounds=${3:-2}; tag=st_$2
[ -d "$ws/repo" ] || { echo "no workspace $ws"; exit 2; }
[ ! -e "$art" ] || { echo "result dir $art exists; refusing to overwrite"; exit 2; }
bash $S/gpu_free.sh "$RSI_GPU" || exit 3
rm -f $RSI_LOGS/ABORT_$tag; mkdir -p "$art/raw" "$art/verifier"
export INFB_WORKSPACE=$ws INFB_SOURCE_TASK_DIR=$RSI_TASK
{ echo "[run_phase] $(date -Is) host=$(hostname) gpu=$RSI_GPU cpus=$RSI_CPUS ws=$ws art=$art rounds=$rounds"
  echo "[run_phase] RSI_EXTRA_SERVE_ARGS='$RSI_EXTRA_SERVE_ARGS' venv=$VIRTUAL_ENV HOME=$HOME TMPDIR=$TMPDIR harness=$RSI_HARNESS"
  git -C "$ws/repo" diff --stat inference_base; } | tee "$art/run_meta.txt"
git -C "$ws/repo" diff inference_base > "$art/candidate.diff" || true
RSI_LOGDIR=$RSI_LOGS bash $S/mem_guard.sh "$tag" "$RSI_GPU" "$VIRTUAL_ENV" &
guard=$!
trap 'touch "$RSI_LOGS/mem_$tag.stop"; wait $guard 2>/dev/null' EXIT
cd "$TMPDIR"; t0=$(date +%s)
taskset -c "$RSI_CPUS" "$VPY" -E -s "$RSI_HARNESS/_bench_rounds.py" \
  --mode throughput --pristine "$ws/baseline_repo" --candidate "$ws/repo" --artifacts "$art" \
  --raw-name sglang_bench_throughput.json --manifest-name sglang_bench_manifest.json \
  --metric-source sglang --metric-key throughput.output_tokens_per_sec \
  --rounds "$rounds" --min-warm 1 --warmup-cap 4 --budget-sec 1200 --bench-timeout-sec 1600 \
  > "$art/raw/bench_rounds.log" 2>&1 || echo "[run_phase] bench rounds returned non-zero"
grep '^\[bench_rounds\]' "$art/raw/bench_rounds.log" || true
[ -f $RSI_LOGS/ABORT_$tag ] && { echo "[run_phase] aborted by foreign-process guard"; exit 4; }
taskset -c "$RSI_CPUS" "$VPY" -E -s "$RSI_HARNESS/verify_regression_real.py" --workspace "$ws" --tests-dir "$RSI_HARNESS" \
  --reward "$art/verifier/reward.json" --metrics "$art/metrics.json" --raw-dir "$art/raw"
echo "[run_phase] $(date -Is) done wall_s=$(( $(date +%s) - t0 ))" | tee -a "$art/run_meta.txt"
cat "$art/verifier/reward.json"
