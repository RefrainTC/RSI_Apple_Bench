#!/bin/bash
# run.sh <label> <mode> [candidate] : calibration / scoring jobs on GPU $GPU (env.sh). Results: $ISO_ROOT/results/<label>/.
# Every timed run is a COLD start: each run gets new, empty compile caches (Triton, Inductor, vLLM / torch.compile, CUDA
# JIT, FlashInfer, HOME and XDG caches) under $STAGE/cache/runs/<label>_<tag>, which are counted (cold witness) and deleted
# after the run. Score = cold time to first result (process launch -> first generate() returned).
#   cal       2 untimed warm-ups (parent, fix: page cache only, caches still cold), 5 AB pairs (AB / BA alternating)
#             + 5 parent/parent A/A pairs, then the teacher-forced check of every run                        (22 runs + TF)
#   score     1 warm-up (parent), 3 ABBA pairs, teacher-forced check of the candidate's and the parent's runs  (7 runs + TF)
#   quick     (broken variants) 1 pair (candidate, parent), TF                                                  (2 runs + TF)
#   transfer  one parent/candidate pair per transfer load (TRANSFERS in loads/cal.env); "warm" = both arms run twice
#             with a cache kept between their two runs, the second run is measured                             (+ TF)
#   donor     one parent run whose caches are kept in $STAGE/donor (only for building the shipcache variant)
#   smoke     harness check: one parent run (1 steady rep), TF
# Before the first candidate run: harness/audit.py (candidate venv vs the parent venv: only Python source may differ).
# After every candidate run: the candidate venv must be unchanged (path, size, mtime of every file, bytecode included).
# Before every run: stop if $ISO_ROOT/STOP or any of $STOP_FILES exists, or if the GPU shows a compute process.
# Holds $ISO_ROOT/locks/<host>_gpu<GPU>.lock. Finished runs are skipped (re-runnable). Then harness/check.py.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
LABEL=$1; MODE=$2; CAND=${3:-fix}
R=$ISO_ROOT/results/$LABEL; mkdir -p "$R" "$ISO_ROOT/locks"; LOG=$R/run.log
log() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
exec 9>"$ISO_ROOT/locks/$(hostname)_gpu$GPU.lock"; flock -w 300 9 || { echo "GPU $GPU lock busy"; exit 1; }
EC=$ISO_ROOT/loads/cal.env; . "$EC"
for r in parent $CAND; do [ -f "$STAGE/.staged/venv_${PR}_$r" ] || { log "not staged: ${PR}_$r (setup/stage.sh, harness/mkvariant.sh)"; exit 2; }; done
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
CR=fix; [ $MODE = cal ] || CR=$CAND
log "start $MODE $LABEL cand=$CAND gpu=$GPU cpus=$CPUS port=$PORT host=$(hostname)"
echo "{\"label\":\"$LABEL\",\"mode\":\"$MODE\",\"cand\":\"$CAND\",\"pr\":\"$PR\",\"gpu\":$GPU,\"cpus\":\"$CPUS\",\"host\":\"$(hostname)\"}" > "$R/run_meta.json"
# static audit of the candidate (recorded; check.py turns a finding into fail:audit)
if [ $MODE != donor ] && [ $MODE != smoke ] && [ ! -f "$R/audit.json" ]; then
  "$SYSPY" -I "$ISO_ROOT/harness/audit.py" "$STAGE/venvs/${PR}_parent" "$STAGE/venvs/${PR}_$CR" "$R/audit.json" | tee -a "$LOG"
fi
vsnap() {  # quick manifest of a venv: path, size, mtime of every file (bytecode included: setup and mkvariant pre-compile it)
  (cd "$1" && find . \( -type f -o -type l \) -printf '%p %s %T@\n' | LC_ALL=C sort) > "$2"
}
RUNS=()
job() {  # tag role envfile [extra env]
  local tag=$1 role=$2 ef=$3 extra=${4:-} out=$R/$1 rc f
  case $tag in wu_*|tf|donor|*_w0_*) ;; *) RUNS+=("$out");; esac
  [ -f "$out/job_meta.json" ] && grep -q '"rc":0,"aborted":""' "$out/job_meta.json" && return 0
  for f in "$ISO_ROOT/STOP" $STOP_FILES; do [ -e "$f" ] && { log "STOP ($f) before $tag"; exit 8; }; done
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID" && { log "GPU $GPU busy before $tag: stop"; exit 9; }
  [ -d "$out" ] && mv "$out" "${out}.failed.$(date +%H%M%S)"
  local C=$STAGE/cache/runs/${LABEL}_$tag KEEP=0
  case $tag in tf) C=$STAGE/cache/tf/$(hostname | cut -c1-12)-g$GPU; KEEP=1;;
               tr_warm_*) C=$STAGE/cache/warm/${LABEL}_$role; KEEP=1;; donor) KEEP=1;; esac
  [ $KEEP = 1 ] || rm -rf "$C"
  mkdir -p "$C"/{triton,inductor,vllm,xdg,nv,fi,th,torch}
  local pre=$R/.vsnap_pre_$tag; [ "$role" != parent ] && vsnap "$STAGE/venvs/${PR}_$role" "$pre"
  EXTRA_ENV="VLLM_PORT=$PORT MASTER_PORT=$((PORT + 50)) TRITON_CACHE_DIR=$C/triton TRITON_HOME=$C/th TORCHINDUCTOR_CACHE_DIR=$C/inductor VLLM_CACHE_ROOT=$C/vllm VLLM_CONFIG_ROOT=$C/vllm/config XDG_CACHE_HOME=$C/xdg CUDA_CACHE_PATH=$C/nv FLASHINFER_WORKSPACE_BASE=$C/fi TORCH_HOME=$C/torch $extra" \
    JOB_TIMEOUT=${JOB_TIMEOUT:-1500} WATCH_FRAC=97 \
    bash "$ISO_ROOT/harness/job.sh" c5y_${PR}_${LABEL}_$tag "$STAGE/venvs/${PR}_$role" "$out" -- \
    bash "$ISO_ROOT/bench/run_bench.sh" "$ef" "$out" > "$R/$tag.joblog" 2>&1 9>&-; rc=$?
  # cold witness: what this run compiled into its own (initially empty) caches
  printf '{"triton_cubins":%s,"triton_files":%s,"inductor_files":%s,"vllm_files":%s,"cache_mb":%s,"cache_dir":"%s"}\n' \
    "$(find "$C/triton" -name '*.cubin' 2>/dev/null | wc -l)" "$(find "$C/triton" -type f 2>/dev/null | wc -l)" \
    "$(find "$C/inductor" -type f 2>/dev/null | wc -l)" "$(find "$C/vllm" -type f 2>/dev/null | wc -l)" \
    "$(du -sm "$C" 2>/dev/null | cut -f1)" "$C" > "$out/cold_witness.json"
  if [ "$role" != parent ]; then
    vsnap "$STAGE/venvs/${PR}_$role" "$R/.vsnap_post_$tag"
    diff "$pre" "$R/.vsnap_post_$tag" > "$out/venv_diff.txt"; echo "{\"venv_changed_lines\":$(grep -c '^[<>]' "$out/venv_diff.txt")}" > "$out/venv_check.json"
    rm -f "$pre" "$R/.vsnap_post_$tag"
  fi
  if [ $tag = donor ]; then rm -rf "$STAGE/donor"; mv "$C" "$STAGE/donor"; elif [ $KEEP = 0 ]; then rm -rf "$C"; fi
  log "end $tag rc=$rc $(tail -1 "$out/job_meta.json" 2>/dev/null | grep -o '"wall_s":[0-9.]*,"rc":[0-9]*,"aborted":"[^"]*"') $(cat "$out/cold_witness.json" | cut -c1-80)"
  # the teacher-forced job sometimes aborts in vLLM's teardown (rc 134, "terminate called without an active exception")
  # after tf.json is written; check.py fails a candidate whose runs are missing from tf.json, so accept it
  if [ $rc != 0 ] && [ $tag = tf ] && [ -s "$out/tf.json" ]; then log "tf rc=$rc after tf.json was written (teardown abort): accepted"; rc=0; fi
  [ $rc = 0 ] || { log "$tag failed rc=$rc: stop"; exit 10; }
}
tfjob() {  # teacher-forced check (trusted parent venv) over every run of this job; warm cache of its own
  local dirs; dirs=$(IFS=:; echo "${RUNS[*]}")
  job tf parent "$ISO_ROOT/loads/tf.env" "TF_DIRS=$dirs"
}
case $MODE in
  cal)
    job wu_1_parent parent $EC; job wu_2_fix fix $EC
    for i in 1 2 3 4 5; do ii=$(printf %02d $i)
      if [ $((i % 2)) = 1 ]; then job ab${ii}_1_parent parent $EC; job ab${ii}_2_fix fix $EC; else job ab${ii}_1_fix fix $EC; job ab${ii}_2_parent parent $EC; fi
      job aa${ii}_1_parent parent $EC; job aa${ii}_2_parent parent $EC
    done
    tfjob ;;
  score)
    job wu_1_parent parent $EC
    job ab01_1_cand $CAND $EC; job ab01_2_parent parent $EC; job ab02_1_parent parent $EC; job ab02_2_cand $CAND $EC
    job ab03_1_cand $CAND $EC; job ab03_2_parent parent $EC
    tfjob ;;
  quick)
    job ab01_1_cand $CAND $EC; job ab01_2_parent parent $EC
    tfjob ;;
  transfer)
    for t in $TRANSFERS; do
      if [ $t = warm ]; then rm -rf "$STAGE/cache/warm/${LABEL}_"*
        job tr_warm_w0_parent parent $EC; job tr_warm_1_parent parent $EC; job tr_warm_w0_cand $CAND $EC; job tr_warm_2_cand $CAND $EC
        rm -rf "$STAGE/cache/warm/${LABEL}_"*
      else job tr_${t}_1_parent parent $ISO_ROOT/loads/tr_$t.env; job tr_${t}_2_cand $CAND $ISO_ROOT/loads/tr_$t.env; fi
    done
    tfjob ;;
  donor) job donor parent $EC ;;
  smoke) job ab01_1_parent parent $ISO_ROOT/loads/smoke.env; tfjob ;;
  *) echo "mode: cal | score | quick | transfer | donor | smoke"; exit 1;;
esac
log "DONE $MODE $LABEL"
"$SYSPY" -I "$ISO_ROOT/harness/check.py" "$ISO_ROOT" "$LABEL" | tee -a "$LOG"
