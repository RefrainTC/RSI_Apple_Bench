#!/usr/bin/env bash
# mem_guard.sh <tag> <gpu> <venv> : 500 ms sampler of the one GPU we use -> $RSI_LOGS/mem_<tag>.log (memory MiB, util %),
# plus a foreign-process check that only ever touches OUR processes: if the GPU has more compute apps than processes
# that hold its /dev/nvidiaN and map our venv, 3 checks in a row (5 s apart) -> touch ABORT_<tag> and stop our own
# harness + servers (the phase is then marked contaminated and rerun). The venv path may contain symlinks: /proc/<pid>/maps
# shows resolved paths, so both the given and the resolved path are matched there. No memory kill: the GPU is ours alone, an OOM
# only ends our own server. Stops when mem_<tag>.stop appears; writes peak to mem_<tag>.events.
set -u
L=${RSI_LOGDIR:-${RSI_ROOT:?set RSI_ROOT}/mm/logs}; tag=$1; GPU=$2; VENV=$3
VREAL=$(realpath -m "$VENV")
UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
MINOR=$(nvidia-smi -q -i $GPU | awk -F': ' '/Minor Number/{gsub(/ /,"",$2);print $2}')
log=$L/mem_$tag.log; stop=$L/mem_$tag.stop; events=$L/mem_$tag.events
rm -f "$stop"; echo "$(date -Is) gpu=$GPU uuid=$UUID minor=$MINOR venv=$VENV resolved=$VREAL" >> "$events"
nvidia-smi -i $GPU --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 >> "$log" &
smi=$!
ours_on_gpu() { local n=0 p; for p in /proc/[0-9]*; do
    ls -l $p/fd 2>/dev/null | grep -q "/dev/nvidia$MINOR\$" || continue
    grep -q -F -e "$VENV/" -e "$VREAL/" $p/maps 2>/dev/null && n=$((n+1)); done; echo $n; }
kill_ours() {
  pkill -f "^$VENV/bin/python -E -s .*/_bench_rounds.py" ; pkill -f "^$VENV/bin/python .*/_candidate_runner.py"
  for pid in $(pgrep -f "^$VENV/bin/python3? -m sglang.launch_server"); do
    pg=$(ps -o pgid= -p "$pid" | tr -d ' '); [[ -n "$pg" ]] && kill -9 -- "-$pg" 2>/dev/null && echo "  killed server pgid $pg" >> "$events"
  done; }
strikes=0
while [[ ! -f "$stop" ]]; do
  sleep 5
  ncomp=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID")
  if (( ncomp > 0 )); then nours=$(ours_on_gpu); else nours=0; fi
  if (( ncomp > nours )); then strikes=$((strikes+1)); echo "$(date -Is) compute_apps=$ncomp > ours=$nours (strike $strikes)" >> "$events"; else strikes=0; fi
  if (( strikes >= 3 )); then
    echo "$(date -Is) FOREIGN PROCESS on GPU $GPU: aborting this phase" >> "$events"; touch "$L/ABORT_$tag"; kill_ours; strikes=0
  fi
done
kill $smi 2>/dev/null
echo "peak_mib=$(awk -F', ' '{print $2}' "$log" | sort -n | tail -1)" >> "$events"
