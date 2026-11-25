#!/bin/bash
# job.sh <tag> <venv> <outdir> -- <cmd...> : one guarded GPU run on $GPU (from env.sh). Copied from batch 4's b4_job.sh
# (itself from the R4090 T5 harness) with machine paths turned into env.sh variables.
# - refuses to start if any compute process is on the GPU (rc 3); aborts if a foreign process appears (rc 4);
# - VRAM watchdog at WATCH_FRAC % of the card (default 97; rc 5); timeout JOB_TIMEOUT s (default 1200; rc 7);
# - kills only its own process tree / processes tagged R4_TAG=<tag> / processes whose cwd is its scratch dir;
# - clean environment (env -i), CPU pinning with taskset $CPUS, HF offline, scratch and TMPDIR under $STAGE.
# Env from the caller: EXTRA_ENV ("K=V K=V"), JOB_TIMEOUT, WATCH_FRAC.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"
TAG=$1; VENV=$2; OUT=$3; shift 4
SCR=$STAGE/tmp/s_$TAG
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
MINOR=$(nvidia-smi -q -i "$GPU" | awk -F': ' '/Minor Number/{gsub(/ /,"",$2);print $2}')
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU"); WATCH_MIB=$(( TOTAL * ${WATCH_FRAC:-97} / 100 ))
gpu_pids() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2==u{print $1}'; }
ours_pids() {  # <root pid or empty>
  local base c d
  base=$( { [ -n "${1:-}" ] && echo "$1"
    grep -l -a -z -x "R4_TAG=$TAG" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3
    for d in /proc/[0-9]*; do c=$(readlink $d/cwd 2>/dev/null) || continue
      case "$c" in "$SCR"|"$SCR"/*) echo ${d#/proc/};; esac; done; } | sort -u | tr '\n' ' ')
  [ -n "${base// /}" ] || return 0
  ps -eo pid=,ppid= | awk -v roots="$base" 'BEGIN{n=split(roots,r," "); for(i=1;i<=n;i++) if(r[i]!="") ins[r[i]]=1} {pp[$1]=$2} END{ch=1; while(ch){ch=0; for(p in pp) if(!(p in ins) && (pp[p] in ins)){ins[p]=1; ch=1}} for(p in ins) if(p in pp) print p}' | sort -u
}
ours_on_dev() { local n=0 p; for p in $(ours_pids "${1:-}"); do ls -l /proc/$p/fd 2>/dev/null | grep -q "/dev/nvidia${MINOR}\$" && n=$((n+1)); done; echo $n; }
kill_ours() { local p; for p in $(ours_pids "${2:-}"); do [ "$p" = "$$" ] && continue; [ "$p" = "$BASHPID" ] && continue; kill -s "$1" "$p" 2>/dev/null; done; }
if [ -n "$(gpu_pids)" ]; then echo "[job] $TAG: GPU $GPU busy"; exit 3; fi
rm -rf "$OUT" "$SCR"; mkdir -p "$OUT" "$SCR/home" "$SCR/tmp" "$SCR/cwd"; trap 'rm -rf "$SCR"' EXIT
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 100 > "$OUT/mem.log" 2>&1 &
SAMPLER=$!
START=$(date +%s.%N)
( cd "$SCR/cwd" && exec env -i HOME="$SCR/home" TMPDIR="$SCR/tmp" USER="${USER:-user}" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PATH="$VENV/bin:/usr/bin:/bin" VIRTUAL_ENV="$VENV" ISO_ROOT="$ISO_ROOT" STAGE="$STAGE" \
    HF_HOME="$STAGE/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID R4_TAG="$TAG" \
    ${EXTRA_ENV:-} taskset -c "$CPUS" "$@" ) > "$OUT/job_stdout.log" 2>&1 &
JPID=$!
ABORT=""; strikes=0; maxcomp=0; T0=$(date +%s)
while kill -0 $JPID 2>/dev/null; do
  for _k in 1 2 3 4 5 6 7 8 9 10; do sleep 1; kill -0 $JPID 2>/dev/null || break; done
  kill -0 $JPID 2>/dev/null || break
  ncomp=$(gpu_pids | wc -l); nours=$(ours_on_dev $JPID)
  [ "$ncomp" -gt "$maxcomp" ] && maxcomp=$ncomp
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  echo "$(date '+%F %T') compute_apps=$ncomp ours_on_dev=$nours used=${used}MiB" >> "$OUT/guard_counts.log"
  if [ "$ncomp" -gt "$nours" ]; then strikes=$((strikes+1)); else strikes=0; fi
  [ "$strikes" -ge 3 ] && ABORT="foreign_gpu_process(compute_apps=$ncomp>ours=$nours)"
  [ "${used:-0}" -gt "$WATCH_MIB" ] && ABORT="mem_watchdog:${used}MiB>${WATCH_MIB}"
  [ $(( $(date +%s) - T0 )) -gt ${JOB_TIMEOUT:-1200} ] && ABORT="timeout"
  if [ -n "$ABORT" ]; then
    echo "$(date '+%F %T') ABORT $ABORT" >> "$OUT/guard.log"
    kill_ours TERM $JPID; sleep 10; kill_ours KILL $JPID; break
  fi
done
wait $JPID 2>/dev/null; RC=$?
END=$(date +%s.%N)
if [ -n "$(ours_pids)" ]; then sleep 2; kill_ours TERM; sleep 3; kill_ours KILL; fi
kill $SAMPLER 2>/dev/null
PEAK=$(awk -F', ' '{if($2+0>m)m=$2+0}END{print m+0}' "$OUT/mem.log")
printf '{"tag":"%s","gpu":%s,"venv":"%s","cpus":"%s","host":"%s","start":%s,"end":%s,"wall_s":%s,"rc":%s,"aborted":"%s","max_compute_apps":%s,"peak_mem_mib":%s,"total_mem_mib":%s,"peak_frac":%s,"extra_env":"%s"}\n' \
  "$TAG" "$GPU" "$(basename $VENV)" "$CPUS" "$(hostname)" "$START" "$END" "$(awk -v a=$START -v b=$END 'BEGIN{printf "%.2f", b-a}')" "$RC" "$ABORT" "$maxcomp" "$PEAK" "$TOTAL" "$(awk -v p=$PEAK -v t=$TOTAL 'BEGIN{printf "%.4f", p/t}')" "${EXTRA_ENV:-}" > "$OUT/job_meta.json"
cat "$OUT/job_meta.json"
rm -rf "$SCR"
case "$ABORT" in "") ;; mem_watchdog*) exit 5;; timeout) exit 7;; *) exit 4;; esac
exit $RC
