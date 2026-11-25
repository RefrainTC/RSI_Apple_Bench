#!/bin/bash
# run.sh <label> <mode> [candidate] : calibration / scoring jobs on GPU $GPU (env.sh). Results: $ISO_ROOT/results/<label>/.
#   cal       3 correctness passes (parent, parent, fix), 2 untimed warm-ups (parent, fix), 5 AB pairs (AB / BA
#             alternating) + 5 parent/parent A/A pairs                                                      (25 runs)
#   score     2 warm-ups (parent, candidate), correctness passes (parent, candidate), 3 ABBA pairs          (10 runs)
#   quick     (broken variants) warm-up of the candidate, correctness passes (parent, candidate), 1 pair      (5 runs)
#   transfer  one parent/candidate pair per transfer load (TRANSFERS in loads/cal.env)
#   smoke     harness check: one correctness pass and one 1-rep timed run of the parent
# candidate = a venv name under $STAGE/venvs without the PR prefix: fix (default) or var_<name> (harness/mkvariant.sh).
# Every run goes through harness/job.sh. Before every run: stop if $ISO_ROOT/STOP or any of $STOP_FILES exists, or if the
# GPU shows a compute process. Holds $ISO_ROOT/locks/<host>_gpu<GPU>.lock. Finished runs are skipped (re-runnable).
# Afterwards: harness/check.py <label> writes results/<label>/verdict.json and prints the summary.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
LABEL=$1; MODE=$2; CAND=${3:-fix}
R=$ISO_ROOT/results/$LABEL; mkdir -p "$R" "$ISO_ROOT/locks"; LOG=$R/run.log
log() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
exec 9>"$ISO_ROOT/locks/$(hostname)_gpu$GPU.lock"; flock -w 300 9 || { echo "GPU $GPU lock busy"; exit 1; }
EC=$ISO_ROOT/loads/cal.env; EK=$ISO_ROOT/loads/cor.env; . "$EC"
for r in parent $CAND; do [ -f "$STAGE/.staged/venv_${PR}_$r" ] || { log "not staged: ${PR}_$r (setup/stage.sh, harness/mkvariant.sh)"; exit 2; }; done
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
log "start $MODE $LABEL cand=$CAND gpu=$GPU cpus=$CPUS port=$PORT host=$(hostname)"
echo "{\"label\":\"$LABEL\",\"mode\":\"$MODE\",\"cand\":\"$CAND\",\"pr\":\"$PR\",\"gpu\":$GPU,\"cpus\":\"$CPUS\",\"host\":\"$(hostname)\"}" > "$R/run_meta.json"
RUNS=()
job() {  # tag role envfile [extra env]
  local tag=$1 role=$2 ef=$3 extra=${4:-} out=$R/$1 rc f
  case $tag in wu_*|tf) ;; *) RUNS+=("$out");; esac
  [ -f "$out/job_meta.json" ] && grep -q '"rc":0,"aborted":""' "$out/job_meta.json" && return 0
  for f in "$ISO_ROOT/STOP" $STOP_FILES; do [ -e "$f" ] && { log "STOP ($f) before $tag"; exit 8; }; done
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID" && { log "GPU $GPU busy before $tag: stop"; exit 9; }
  [ -d "$out" ] && mv "$out" "${out}.failed.$(date +%H%M%S)"
  local C=$STAGE/cache/$PR/$role/$(hostname | cut -c1-12)-g$GPU
  mkdir -p "$C"
  EXTRA_ENV="SGLANG_PORT_BASE=$((PORT + 100)) MASTER_PORT=$((PORT + 50)) TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton FLASHINFER_WORKSPACE_BASE=$C/fi XDG_CACHE_HOME=$C/xdg CUDA_HOME=$STAGE/cuda $extra" \
    JOB_TIMEOUT=${JOB_TIMEOUT:-1200} WATCH_FRAC=97 \
    bash "$ISO_ROOT/harness/job.sh" c5y_${PR}_${LABEL}_$tag "$STAGE/venvs/${PR}_$role" "$out" -- \
    bash "$ISO_ROOT/bench/run_bench.sh" "$ef" "$out" > "$R/$tag.joblog" 2>&1 9>&-; rc=$?
  log "end $tag rc=$rc $(tail -1 "$out/job_meta.json" 2>/dev/null | grep -o '"wall_s":[0-9.]*,"rc":[0-9]*,"aborted":"[^"]*"')"
  [ $rc = 0 ] || { log "$tag failed rc=$rc: stop"; exit 10; }
}
tfjob() { :; }  # no teacher-forced check in this task: outputs must equal the parent's exactly
case $MODE in
  cal)
    job cor_1_parent parent $EK; job cor_2_parent parent $EK; job cor_3_fix fix $EK
    job wu_1_parent parent $EC; job wu_2_fix fix $EC
    for i in 1 2 3 4 5; do ii=$(printf %02d $i)
      if [ $((i % 2)) = 1 ]; then job ab${ii}_1_parent parent $EC; job ab${ii}_2_fix fix $EC; else job ab${ii}_1_fix fix $EC; job ab${ii}_2_parent parent $EC; fi
      job aa${ii}_1_parent parent $EC; job aa${ii}_2_parent parent $EC
    done
    tfjob ;;
  score)
    job wu_1_parent parent $EC; job wu_2_cand $CAND $EC
    job cor_1_parent parent $EK; job cor_2_cand $CAND $EK
    job ab01_1_cand $CAND $EC; job ab01_2_parent parent $EC; job ab02_1_parent parent $EC; job ab02_2_cand $CAND $EC
    job ab03_1_cand $CAND $EC; job ab03_2_parent parent $EC
    tfjob ;;
  quick)
    job wu_2_cand $CAND $EC
    job cor_1_parent parent $EK; job cor_2_cand $CAND $EK
    job ab01_1_cand $CAND $EC; job ab01_2_parent parent $EC
    tfjob ;;
  transfer)
    for t in $TRANSFERS; do job tr_${t}_1_parent parent $ISO_ROOT/loads/tr_$t.env; job tr_${t}_2_cand $CAND $ISO_ROOT/loads/tr_$t.env; done
    tfjob ;;
  smoke)
    job cor_1_parent parent $EK; job ab01_1_parent parent $ISO_ROOT/loads/smoke.env
    tfjob ;;
  *) echo "mode: cal | score | quick | transfer | smoke"; exit 1;;
esac
log "DONE $MODE $LABEL"
"$SYSPY" -I "$ISO_ROOT/harness/check.py" "$ISO_ROOT" "$LABEL" | tee -a "$LOG"
