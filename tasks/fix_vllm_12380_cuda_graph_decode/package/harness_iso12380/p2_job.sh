#!/bin/bash
# p2_job.sh <tag> <gpu> <venv> <outdir> -- <cmd...>
# Guarded GPU job (adapted from R4090 T5 t5_job.sh). Differences: process ownership is ONLY
#   (a) descendants of this job's root pid, (b) processes whose environ carries R4_TAG=<tag>,
#   (c) processes whose cwd is under this job's unique scratch dir (catches orphans re-parented to init),
#   plus descendants of those. No interpreter-path heuristics, so concurrent jobs on other GPUs
#   that use the same venv are never touched.
# Env: P2_STORAGE (label recorded in job_meta: nas | tmp-nvme), P2_SCRATCH_ROOT, HF_HOME_OVERRIDE, P2_CPUS_OVERRIDE (else
#      the per-GPU row of /home/coder/work/cpu_pinning_main.txt), JOB_TIMEOUT (s, default 1800), EXTRA_ENV ("K=V K=V"), P2_PYTHONPATH (prepended overlay root),
#      T5 (passed through for task scripts), WATCH_FRAC (default 92 = % of total VRAM).
# exit: cmd rc; 3 = GPU busy before start; 4 = foreign GPU process appeared; 5 = own VRAM watchdog; 7 = timeout
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TAG=$1; GPU=$2; VENV=$3; OUT=$4; shift 5
P2=$ISO_ROOT; T5=${T5:-$ISO_ROOT}; CPUS=${P2_CPUS_OVERRIDE:-$(bash $P2/harness_iso12380/p2_cpus.sh $GPU)} || exit 1   # always the pinning table (callers' CPUS ignored)
SCR=${P2_SCRATCH_ROOT:-$P2/scratch}/$TAG   # P2_SCRATCH_ROOT=$P2T/scratch when the task is staged on local NVMe
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
MINOR=$(nvidia-smi -q -i "$GPU" | awk -F': ' '/Minor Number/{gsub(/ /,"",$2);print $2}')
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU"); WATCH_MIB=$(( TOTAL * ${WATCH_FRAC:-92} / 100 ))
gpu_pids() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$UUID" '$2==u{print $1}'; }
ours_pids() {  # <root pid or empty>
  local base p c
  base=$( { [ -n "${1:-}" ] && echo "$1"
    grep -l -a -z -x "R4_TAG=$TAG" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3
    for d in /proc/[0-9]*; do c=$(readlink $d/cwd 2>/dev/null) || continue
      case "$c" in "$SCR"|"$SCR"/*) echo ${d#/proc/};; esac; done; } | sort -u | tr '\n' ' ')
  [ -n "${base// /}" ] || return 0
  ps -eo pid=,ppid= | awk -v roots="$base" 'BEGIN{n=split(roots,r," "); for(i=1;i<=n;i++) if(r[i]!="") ins[r[i]]=1} {pp[$1]=$2} END{ch=1; while(ch){ch=0; for(p in pp) if(!(p in ins) && (pp[p] in ins)){ins[p]=1; ch=1}} for(p in ins) if(p in pp) print p}' | sort -u
}
ours_on_dev() { local n=0 p; for p in $(ours_pids "${1:-}"); do ls -l /proc/$p/fd 2>/dev/null | grep -q "/dev/nvidia${MINOR}\$" && n=$((n+1)); done; echo $n; }
kill_ours() { local p; for p in $(ours_pids "${2:-}"); do [ "$p" = "$$" ] && continue; [ "$p" = "$BASHPID" ] && continue; kill -s "$1" "$p" 2>/dev/null; done; }
if [ -n "$(gpu_pids)" ]; then echo "[p2_job] $TAG: GPU $GPU busy"; exit 3; fi
rm -rf "$OUT" "$SCR"; mkdir -p "$OUT" "$SCR/home" "$SCR/tmp" "$SCR/cwd"
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 100 > "$OUT/mem.log" 2>&1 &
SAMPLER=$!
START=$(date +%s.%N)
( cd "$SCR/cwd" && exec env -i HOME="$SCR/home" TMPDIR="$SCR/tmp" USER=coder LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PATH="$VENV/bin:/usr/bin:/bin" VIRTUAL_ENV="$VENV" T5="$T5" P2="$P2" \
    HF_HOME="${HF_HOME_OVERRIDE:-$T5/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID R4_TAG="$TAG" \
    ${P2_PYTHONPATH:+P2_OVERLAY=$P2_PYTHONPATH} ${EXTRA_ENV:-} taskset -c "$CPUS" "$@" ) > "$OUT/job_stdout.log" 2>&1 &
JPID=$!
ABORT=""; strikes=0; maxcomp=0; T0=$(date +%s)
while kill -0 $JPID 2>/dev/null; do
  for _k in 1 2 3 4 5 6 7 8 9 10; do sleep 1; kill -0 $JPID 2>/dev/null || break; done   # notice job end within 1 s
  kill -0 $JPID 2>/dev/null || break
  ncomp=$(gpu_pids | wc -l); nours=$(ours_on_dev $JPID)
  [ "$ncomp" -gt "$maxcomp" ] && maxcomp=$ncomp
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  echo "$(date '+%F %T') compute_apps=$ncomp ours_on_dev=$nours used=${used}MiB" >> "$OUT/guard_counts.log"
  if [ "$ncomp" -gt "$nours" ]; then strikes=$((strikes+1)); else strikes=0; fi
  [ "$strikes" -ge 3 ] && ABORT="foreign_gpu_process(compute_apps=$ncomp>ours=$nours)"
  [ "${used:-0}" -gt "$WATCH_MIB" ] && ABORT="mem_watchdog:${used}MiB>${WATCH_MIB}"
  [ $(( $(date +%s) - T0 )) -gt ${JOB_TIMEOUT:-1800} ] && ABORT="timeout"
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
printf '{"host":"%s","tag":"%s","gpu":%s,"venv":"%s","cpus":"%s","start":%s,"end":%s,"wall_s":%s,"rc":%s,"aborted":"%s","max_compute_apps":%s,"peak_mem_mib":%s,"total_mem_mib":%s,"peak_frac":%s,"overlay":"%s","extra_env":"%s","storage":"%s","venv_path":"%s","hf_home":"%s","scratch":"%s","cache":"%s"}\n' \
  "$(bash $P2/harness_iso12380/hosttag.sh)" "$TAG" "$GPU" "$(basename $VENV)" "$CPUS" "$START" "$END" "$(awk -v a=$START -v b=$END 'BEGIN{printf "%.2f", b-a}')" "$RC" "$ABORT" "$maxcomp" "$PEAK" "$TOTAL" "$(awk -v p=$PEAK -v t=$TOTAL 'BEGIN{printf "%.4f", p/t}')" "${P2_PYTHONPATH:-}" "${EXTRA_ENV:-}" "${P2_STORAGE:-nas}" "$VENV" "${HF_HOME_OVERRIDE:-$T5/hf}" "$SCR" "${P2_CACHE_STATE:-none}" > "$OUT/job_meta.json"
cat "$OUT/job_meta.json"
rm -rf "$SCR"
case "$ABORT" in "") ;; mem_watchdog*) exit 5;; timeout) exit 7;; *) exit 4;; esac
exit $RC
