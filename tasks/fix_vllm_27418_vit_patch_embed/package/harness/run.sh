#!/bin/bash
# run.sh <label> <mode> [candidate] : calibration / scoring jobs on GPU $GPU (env.sh). Results: $ISO_ROOT/results/<label>/.
#   cal       text passes (parent, parent, fix), image passes (parent, parent, fix), 2 untimed warm-ups (parent, fix),
#             5 AB pairs (AB / BA alternating) + 5 parent/parent A/A pairs, then the teacher-forced re-read of every
#             timed run and image pass by the parent                                                 (28 runs + TF)
#   score     2 warm-ups (parent, candidate), text passes (parent, candidate), image pass (candidate), 3 ABBA pairs, TF
#   quick     (broken variants) warm-ups (parent, candidate), text passes (parent, candidate), image pass (candidate),
#             1 pair, TF
#   transfer  one parent/candidate pair per transfer load (TRANSFERS in loads/cal.env), TF
#   smoke     harness check: text pass, image pass, one 1-rep timed run of the parent, TF
# candidate = a venv name under $STAGE/venvs without the PR prefix: fix (default) or var_<name> (harness/mkvariant.sh).
# Every run goes through harness/job.sh. Before every run: stop if $ISO_ROOT/STOP or any of $STOP_FILES exists, or if the
# GPU shows a compute process. Holds $ISO_ROOT/locks/<host>_gpu<GPU>.lock. Finished runs are skipped (re-runnable).
# Compile caches (Triton, Inductor, vLLM's torch.compile cache) are kept per role, host and GPU under $STAGE/cache/<pr>/:
# the score is the steady state after start-up, so warm caches only shorten start-up.
# Afterwards: harness/check.py <label> writes results/<label>/verdict.json and prints the summary.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
LABEL=$1; MODE=$2; CAND=${3:-fix}
R=$ISO_ROOT/results/$LABEL; mkdir -p "$R" "$ISO_ROOT/locks"; LOG=$R/run.log
log() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
exec 9>"$ISO_ROOT/locks/$(hostname)_gpu$GPU.lock"; flock -w 300 9 || { echo "GPU $GPU lock busy"; exit 1; }
EC=$ISO_ROOT/loads/cal.env; EK=$ISO_ROOT/loads/cor.env; EI=$ISO_ROOT/loads/img.env; . "$EC"
for r in parent $CAND; do [ -f "$STAGE/.staged/venv_${PR}_$r" ] || { log "not staged: ${PR}_$r (setup/stage.sh, harness/mkvariant.sh)"; exit 2; }; done
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
log "start $MODE $LABEL cand=$CAND gpu=$GPU cpus=$CPUS port=$PORT host=$(hostname)"
echo "{\"label\":\"$LABEL\",\"mode\":\"$MODE\",\"cand\":\"$CAND\",\"pr\":\"$PR\",\"gpu\":$GPU,\"cpus\":\"$CPUS\",\"host\":\"$(hostname)\"}" > "$R/run_meta.json"
RUNS=()
job() {  # tag role envfile [extra env]
  local tag=$1 role=$2 ef=$3 extra=${4:-} out=$R/$1 rc f
  case $tag in wu_*|tf|cor_*) ;; *) RUNS+=("$out");; esac
  [ -f "$out/job_meta.json" ] && grep -q '"rc":0,"aborted":""' "$out/job_meta.json" && return 0
  for f in "$ISO_ROOT/STOP" $STOP_FILES; do [ -e "$f" ] && { log "STOP ($f) before $tag"; exit 8; }; done
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID" && { log "GPU $GPU busy before $tag: stop"; exit 9; }
  [ -d "$out" ] && mv "$out" "${out}.failed.$(date +%H%M%S)"
  local C=$STAGE/cache/$PR/$role/$(hostname | cut -c1-12)-g$GPU
  if [ "$role" != parent ]; then
    # a candidate run starts from a fresh copy of the parent's compile caches on this host and GPU: Inductor stores the
    # Triton configurations it benchmarked (*.best_config), and a separately tuned cache can pick other ones for the same
    # graph, whose sums round differently (seen in calibration: same engine, other cache -> text-pass logprobs off by up
    # to 0.1 nats). With the copy both arms run the same kernels; the candidate's own writes never reach the parent's cache.
    local PCD=$STAGE/cache/$PR/parent/$(hostname | cut -c1-12)-g$GPU
    # (fixed 2026-11-25 after acc6: on a fresh stage $C's parent directory did not exist, the copy failed unnoticed and
    # the candidate started on an empty cache; now the directory is made and a failed copy stops the job)
    rm -rf "$C"; mkdir -p "$PCD" "$(dirname "$C")"; cp -a "$PCD" "$C" || { log "cache copy $PCD -> $C failed: stop"; exit 11; }
  fi
  mkdir -p "$C"/{triton,inductor,vllm,xdg,fi,torch}
  EXTRA_ENV="VLLM_PORT=$PORT MASTER_PORT=$((PORT + 50)) TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg FLASHINFER_WORKSPACE_BASE=$C/fi TORCH_HOME=$C/torch $extra" \
    JOB_TIMEOUT=${JOB_TIMEOUT:-1200} WATCH_FRAC=97 \
    bash "$ISO_ROOT/harness/job.sh" c5y_${PR}_${LABEL}_$tag "$STAGE/venvs/${PR}_$role" "$out" -- \
    bash "$ISO_ROOT/bench/run_bench.sh" "$ef" "$out" > "$R/$tag.joblog" 2>&1 9>&-; rc=$?
  log "end $tag rc=$rc $(tail -1 "$out/job_meta.json" 2>/dev/null | grep -o '"wall_s":[0-9.]*,"rc":[0-9]*,"aborted":"[^"]*"')"
  if [ $rc != 0 ] && [ $tag = tf ] && [ -s "$out/tf.json" ]; then log "tf rc=$rc after tf.json was written (teardown abort): accepted"; rc=0; fi
  [ $rc = 0 ] || { log "$tag failed rc=$rc: stop"; exit 10; }
}
tfjob() {  # teacher-forced re-read (trusted parent venv) of every timed run and image pass of this job
  local dirs; dirs=$(IFS=:; echo "${RUNS[*]}")
  job tf parent "$ISO_ROOT/loads/tf.env" "TF_DIRS=$dirs"
}
case $MODE in
  cal)
    # warm-ups first (as in score mode), so that every correctness pass, the parent's included, runs on warm caches
    job wu_1_parent parent $EC; job wu_2_fix fix $EC
    job cor_1_parent parent $EK; job cor_2_parent parent $EK; job cor_3_fix fix $EK
    job img_1_parent parent $EI; job img_2_parent parent $EI; job img_3_fix fix $EI
    for i in 1 2 3 4 5; do ii=$(printf %02d $i)
      if [ $((i % 2)) = 1 ]; then job ab${ii}_1_parent parent $EC; job ab${ii}_2_fix fix $EC; else job ab${ii}_1_fix fix $EC; job ab${ii}_2_parent parent $EC; fi
      job aa${ii}_1_parent parent $EC; job aa${ii}_2_parent parent $EC
    done
    tfjob ;;
  score)
    job wu_1_parent parent $EC; job wu_2_cand $CAND $EC
    job cor_1_parent parent $EK; job cor_2_cand $CAND $EK; job img_2_cand $CAND $EI
    job ab01_1_cand $CAND $EC; job ab01_2_parent parent $EC; job ab02_1_parent parent $EC; job ab02_2_cand $CAND $EC
    job ab03_1_cand $CAND $EC; job ab03_2_parent parent $EC
    tfjob ;;
  quick)
    job wu_1_parent parent $EC; job wu_2_cand $CAND $EC
    job cor_1_parent parent $EK; job cor_2_cand $CAND $EK; job img_2_cand $CAND $EI
    job ab01_1_cand $CAND $EC; job ab01_2_parent parent $EC
    tfjob ;;
  transfer)
    for t in $TRANSFERS; do job tr_${t}_1_parent parent $ISO_ROOT/loads/tr_$t.env; job tr_${t}_2_cand $CAND $ISO_ROOT/loads/tr_$t.env; done
    tfjob ;;
  smoke)
    job cor_1_parent parent $EK; job img_1_parent parent $EI; job ab01_1_parent parent $ISO_ROOT/loads/smoke.env
    tfjob ;;
  *) echo "mode: cal | score | quick | transfer | smoke"; exit 1;;
esac
log "DONE $MODE $LABEL"
"$SYSPY" -I "$ISO_ROOT/harness/check.py" "$ISO_ROOT" "$LABEL" | tee -a "$LOG"
