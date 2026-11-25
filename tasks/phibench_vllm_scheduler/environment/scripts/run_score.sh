#!/usr/bin/env bash
# run_score.sh <label> <arm> : one full scoring = a fresh container that places one candidate in /app/submission and
# runs the task's own tests/test.sh, unmodified. Our scripts gpu_run.sh + score.sh (and the sweep's copies), merged.
#   arm = starter   the task's starting submission/launch_server.sh (the 0.85 copy from $PHI_TASK)
#         negative  solution/solve.sh --negative (fake server that returns fixed text at once)
#         oracle    solution/solve.sh --oracle (= solution/oracle_v2_launch_server.sh)
#         identity  identity_launch_server.sh (= tests/launch_baseline.sh without its comment lines: the A/A control)
#         variant:<file>  a sweep candidate from make_variants.py, copied to /app/submission/launch_server.sh
# Variables: PHI_TASK (task copy with the 0.85 edit, required), PHI_MODEL (model dir from download_model.py, required),
#   PHI_RUNS (result root, required), IMG (image), LOCK (a GPU lock file, optional), LIMIT_MIB (whole-card watchdog,
#   default 30720), CAP_S (wall-clock cap, default 3000 s; a normal scoring takes about 1300 s), MEM_LIMIT_GB (32).
# The container has its own PID namespace (no --pid=host), so test.sh's `pkill -f vllm` only reaches the container.
# Never run test.sh directly on a host. The watchdog only kills this run's own container.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
T=${PHI_TASK:?}; MODEL=${PHI_MODEL:?}; IMG=${IMG:-phibench-vllm-sched:oss-sm120}
LIMIT_MIB=${LIMIT_MIB:-30720}; CAP_S=${CAP_S:-3000}; MEM_LIMIT_GB=${MEM_LIMIT_GB:-32}
label="${1:?label}"; arm="${2:?arm}"; OUT=${PHI_RUNS:?}/$label; name="phi-$label"
mkdir -p "$OUT"; VDIR=$S
case "$arm" in
  starter)  step='cmp -s /task/submission/launch_server.sh /app/submission/launch_server.sh || { diff /app/submission/launch_server.sh /task/submission/launch_server.sh; cp /task/submission/launch_server.sh /app/submission/launch_server.sh; chmod +x /app/submission/launch_server.sh; }' ;;
  negative) step='bash /task/solution/solve.sh --negative' ;;
  oracle)   step='bash /task/solution/solve.sh --oracle' ;;
  identity) step='cp /phi/identity_launch_server.sh /app/submission/launch_server.sh && chmod +x /app/submission/launch_server.sh' ;;
  variant:*) v=$(realpath "${arm#variant:}"); VDIR=$(dirname "$v")
            step="cp /phi/$(basename "$v") /app/submission/launch_server.sh && chmod +x /app/submission/launch_server.sh" ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac
cat > "$OUT/inner.sh" <<IN
set -uo pipefail
cd /app
echo "[inner] \$(date -Is) label=$label arm=$arm"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$step
echo "[inner] submission:"; sha256sum /app/submission/*
cp -a /app/submission /logs/submission_snapshot
t0=\$(date +%s)
bash /tests/test.sh > /logs/test_stdout.log 2>&1
echo "[inner] test.sh rc=\$? wall_s=\$(( \$(date +%s) - t0 ))"
IN
RUN="docker run --rm --name $name --gpus all --cpus 8 --memory ${MEM_LIMIT_GB}g --memory-swap ${MEM_LIMIT_GB}g --shm-size 8g --network none \
 -v $MODEL:/app/model:ro -v $T:/task:ro -v $T/tests:/tests:ro -v $VDIR:/phi:ro -v $OUT:/logs $IMG bash /logs/inner.sh"
echo "$RUN" > "$OUT/docker_cmd.txt"
body() {
  # wait until the GPU is idle (a desktop may hold about 0.8 GiB)
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "$u" -lt 2000 ] && break; sleep 10; done
  [ "$u" -lt 2000 ] || { echo "[phi] ABORT: $u MiB in use before start"; return 3; }
  echo "[phi] pre: load $(cut -d' ' -f1-3 /proc/loadavg)"   # record host load: it moves REF (see README)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/mem.log" & smi=$!
  t0=$(date +%s); $RUN > "$OUT/docker_client.log" 2>&1 & job=$!; killed=0; timed_out=0
  while kill -0 $job 2>/dev/null; do
    m=$(tail -1 "$OUT/mem.log" 2>/dev/null | tr -d ' ')
    if [[ "$m" =~ ^[0-9]+$ ]] && (( m > LIMIT_MIB )); then echo "[phi-watchdog] VRAM $m MiB > $LIMIT_MIB"; docker kill $name >/dev/null 2>&1; killed=1; break; fi
    if (( $(date +%s) - t0 > CAP_S )); then echo "[phi-timeout] wall > $CAP_S s"; docker kill $name >/dev/null 2>&1; timed_out=1; break; fi
    sleep 0.5
  done
  wait $job; rc=$?; t1=$(date +%s); sleep 3; kill $smi 2>/dev/null
  peak=$(sort -n "$OUT/mem.log" | tail -1)
  printf '{"label":"%s","rc":%s,"wall_s":%s,"peak_mib":%s,"watchdog_killed":%s,"timed_out":%s,"load_end":"%s"}\n' \
    "$label" "$rc" "$((t1-t0))" "${peak:-null}" "$killed" "$timed_out" "$(cut -d' ' -f1-3 /proc/loadavg)" > "$OUT/run_meta.json"
  # the container writes /logs as root: hand the files back
  docker run --rm --network none -v $OUT:/logs $IMG chown -R $(id -u):$(id -g) /logs >/dev/null 2>&1
}
if [ -n "${LOCK:-}" ]; then exec 9>"$LOCK"; flock 9; fi
body 2>&1 | tee -a "$OUT/driver.log"
