#!/bin/bash
# run_variant.sh <taskdir> <role> <stage|-> <tag> <out> <mode> [K=V ...]
# role: parent  = pristine venv + empty overlay (symmetric with cand)
#       cand    = pristine venv + overlay of <stage> (agent's gated changes)
#       fix     = pristine venv + overlay of the human fix ($TD/calib/oracle_stage)
#       parentvenv / fixvenv = the venvs directly, no overlay (overlay-equivalence check)
# mode: passed to the task run script (timed | correct | refout | counter ...). K=V: workload env (task-defined names).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TD=$(readlink -f "$1"); ROLE=$2; STAGE=$3; TAG=$4; OUT=$5; MODE=$6; shift 6
P2=$ISO_ROOT; H=$P2/harness_iso13365; T5=${T5:-$ISO_ROOT}
source "$TD/task.env"
# storage: task.env may stage the venvs / weights / caches on local NVMe (P2_STORAGE=tmp-nvme, see scripts/stage_tmp.sh).
# A staged task refuses to run when $P2T was wiped (pod restart) instead of silently falling back to the NAS.
OVR=${P2_OVERLAY_ROOT:-$P2/overlays}
# the human-fix venv is only needed (and only checked) for the fixvenv role: during trials it is sealed away (vault.sh)
VCHK=$VENV; [ "$ROLE" = fixvenv ] && VCHK="$VENV $FIX_VENV"
P2T=$P2T
# iso13365: operator stop file, checked before every GPU run (a stopped run exits 8; pairs.py moves on, so the job ends fast)
[ -e $ISO_ROOT/STOP_iso13365 ] && { echo "[run_variant] STOP_iso13365 present: not started"; exit 8; }
case "$VENV" in $P2T/*) for v in $VCHK; do [ -f "$P2T/.staged/venv_$(basename $v)" ] || {
  echo "[run_variant] $v is not staged (pod restarted?): run scripts/stage_tmp.sh"; exit 6; }; done;; esac
MVENV=$VENV; [ "$ROLE" = fixvenv ] && MVENV=$FIX_VENV
# per-run venv (review F4): the job runs from a per-GPU copy of the staged master, verified before and after the run
# (harness/venv_guard.sh); a run that changes it is flagged (venv_changed) and the copy is replaced.
RV=""; VG=$OUT.venv_guard.json; rm -f "$VG"
if [ "${P2_FRESH_VENV:-0}" = 1 ]; then
  RV=$(bash $H/venv_guard.sh pre "$MVENV" "$P2T/run/g$GPU/$(basename $MVENV)" "$VG" | tail -1) || { echo "[run_variant] venv_guard pre failed"; exit 6; }
  JVENV=$RV
else JVENV=$MVENV; fi
SITE=$("$JVENV/bin/python3" -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
# compile caches (vLLM torch.compile, Inductor, Triton), one directory per (code under test, GPU): parent and pristine
# share one, the human fix has its own, every distinct candidate stage gets its own (content hash). vLLM's cache key
# includes the traced source files' PATHS, so with caching on, the overlay path is stable per (task, variant, GPU).
# P2_CACHE_MODE:
#   dev-warm : persistent per-variant cache (dev runs; speed matters, never read by scoring)
#   job      : final scoring / calibration. $P2_CACHE_ROOT is fresh for the scoring job. Warm-up jobs (P2_CACHE_WARMUP=1,
#              untimed, different prompt set) write <variant>/g<gpu>/master from empty; every other run gets its own copy
#              of the master (or an empty dir if there is none = cold) that is deleted after the run, so nothing a run
#              writes reaches a later run. score.sh / calibrate.sh delete the whole root at the end.
OVKEY=$TAG; CACHE_ENV=""; CSTATE=none; DELCD=""; CD=""
if [ -n "${P2_CACHE_ROOT:-}" ]; then
  case "$ROLE" in
    cand) CK=cand-$(cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16) ;;
    fix) CK=fix ;; fixvenv) CK=fixvenv ;; *) CK=pristine ;;
  esac
  B=$P2_CACHE_ROOT/$CK/g$GPU; OVKEY=$TASK/$CK/g$GPU
  case "${P2_CACHE_MODE:-dev-warm}" in
    job) if [ "${P2_CACHE_WARMUP:-0}" = 1 ]; then rm -rf "$B/master"; CD=$B/master; mkdir -p "$CD"; CSTATE=job-warmup-from-empty
         else CD=$B/run-$TAG; rm -rf "$CD"
              if [ -d "$B/master" ]; then cp -a "$B/master" "$CD"; CSTATE=warm-in-job; else mkdir -p "$CD"; CSTATE=cold; fi
              DELCD=$CD; fi ;;
    *) CD=$B; mkdir -p "$CD"; CSTATE=dev-warm ;;
  esac
  CACHE_ENV="VLLM_CACHE_ROOT=$CD/vllm TORCHINDUCTOR_CACHE_DIR=$CD/inductor TRITON_CACHE_DIR=$CD/triton"
fi
OV=""
case "$ROLE" in
  parent) OV=$OVR/$OVKEY; bash $H/mk_overlay.sh "$SITE" "$PKG" - "$OV" >&2 ;;
  cand)   OV=$OVR/$OVKEY; bash $H/mk_overlay.sh "$SITE" "$PKG" "$STAGE" "$OV" >&2 ;;
  fix)    OV=$OVR/$OVKEY; bash $H/mk_overlay.sh "$SITE" "$PKG" "$TD/calib/oracle_stage" "$OV" >&2 ;;
  parentvenv|fixvenv) ;;
  *) echo "bad role $ROLE"; exit 2 ;;
esac
# audit hook (review F4.2): sitecustomize in every Python process of the measured job, parent and candidate alike
AUD=""
if [ "${AUDIT:-0}" = 1 ]; then
  SCR=${P2_SCRATCH_ROOT:-$P2/scratch}/$TAG
  AUD="P2_AUDIT_DIR=$H/audit P2_AUDIT_OUT=$OUT P2_AUDIT_ALLOW_W=$SCR${CD:+:$CD} P2_AUDIT_NEED_END=${P2_AUDIT_NEED_END:-0}"
  AUD="$AUD P2_AUDIT_PROTECT_W=$P2:$T5:$P2T/venvs:$P2T/hf:$P2T/uvpy:$P2T/run:$ISO_ROOT:$ISO_ROOT/cpu_pinning_main.txt"
  AUD="$AUD P2_AUDIT_CODE_ROOTS=$TD:$T5/bench:$SCR${CD:+:$CD} P2_AUDIT_TRUSTED=$TD/runner.py:$TD/correct.py:$TD/correct_client.py"
  # leakage (2026-11-23): reads / listings under everything we own on the pod are flagged unless they are this run's own
  # dirs (scratch, cache copy, OUT, overlay, venv copy: the last three are added by the hook itself) or the model files
  # job-mode caches: vLLM 0.8's torch.compile cache records ABSOLUTE paths of the Triton kernels the warm-up compiled,
  # so a run reads them from this job's master of the same variant, not from its own copy (seen 2026-11-23, #16436)
  MASTER=""; [ "${P2_CACHE_MODE:-}" = job ] && [ -n "$CD" ] && MASTER=$(dirname "$CD")/master
  AUD="$AUD P2_AUDIT_PROTECT_R=$P2:$T5:$P2T:$HOME P2_AUDIT_ALLOW_R=$SCR${CD:+:$CD}${MASTER:+:$MASTER}:${HF_HOME_TASK:-$T5/hf}:$P2T/uvpy P2_AUDIT_TASKDIR=$TD"
fi
EXTRA="PYTHONDONTWRITEBYTECODE=1 ${TASK_ENV:-} $CACHE_ENV $AUD $*"
P2_PYTHONPATH="$OV" EXTRA_ENV="$EXTRA" JOB_TIMEOUT=${JOB_TIMEOUT:-1800} \
  HF_HOME_OVERRIDE=${HF_HOME_TASK:-} P2_STORAGE=${P2_STORAGE:-nas} P2_CACHE_STATE=$CSTATE \
  bash $H/p2_job.sh "$TAG" "$GPU" "$JVENV" "$OUT" -- bash "$TD/run.sh" "$OUT" "$MODE"
RC=$?
[ -n "$OV" ] && rm -rf "$OV"
[ -n "$DELCD" ] && rm -rf "$DELCD"
if [ -n "$RV" ]; then bash $H/venv_guard.sh post "$MVENV" "$RV" "$VG" > /dev/null; mv "$VG" "$OUT/venv_guard.json" 2>/dev/null
  mv "${VG%.json}".*_diff.txt "$OUT/" 2>/dev/null; fi
printf '{"role":"%s","mode":"%s","workload":"%s","storage":"%s","cache":"%s","fresh_venv":"%s","audit":"%s"}\n' \
  "$ROLE" "$MODE" "$*" "${P2_STORAGE:-nas}" "$CSTATE" "${RV:-no}" "${AUDIT:-0}" > "$OUT/variant.json"
exit $RC
