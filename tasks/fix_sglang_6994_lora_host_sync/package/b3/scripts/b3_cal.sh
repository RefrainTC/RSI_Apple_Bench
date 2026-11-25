#!/bin/bash
# b3_cal.sh <gpu> <pr> <label> <cal|score> [--transfer] : batch-3 calibration / scoring job for a screened PR (human fix
# as the candidate). Env files: b3/env/<pr>_cal.env (official load, ROTATE=1), <pr>_cor.env (CORRECT=1 pass),
# <pr>_tr_<name>.env (transfer loads, names in TRANSFERS of the cal env). Runs go through b3_job.sh (busy-GPU refusal,
# foreign-process abort, VRAM watchdog), with the same ports / caches / STOP_batch3 checks as b3_pairs.sh.
#   cal   : 3 correctness passes (parent, parent, fix), 2 untimed warm-ups (parent, fix), 5 AB pairs (AB/BA alternating) + 5 A/A
#   score : 2 untimed warm-ups, 1 correctness pass of the fix, 3 AB pairs (ABBA); --transfer adds one AB pair per transfer load
# Output: b3/cal/s<pr>/<label>/<tag>/ ; analysis: scripts/cal_analyze.py.
set -uo pipefail
GPU=$1; PR=$2; LABEL=$3; MODE=$4; TRF=${5:-}
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; P2=$ISO_ROOT; B3=$P2/b3; HT=$(bash $B3/scripts/hosttag.sh)
RUN=$B3/cal/s$PR/$LABEL; LOG=$RUN/cal.log; mkdir -p $RUN
EC=$B3/env/${PR}_cal.env; EK=$B3/env/${PR}_cor.env; . $EC
for r in parent fix; do [ -f $STAGE/b3/.staged/venv_${PR}_$r ] || { echo "not staged: ${PR}_$r" >> $LOG; exit 2; }; done
export P2_STORAGE=tmp-nvme HF_HOME_OVERRIDE=$STAGE/b3/hf P2_SCRATCH_ROOT=$STAGE/b3/scratch
echo "$(date -u '+%F %T') start $MODE $LABEL host=$HT gpu=$GPU $TRF" >> $LOG
job() {  # tag role envfile
  local tag=$1 role=$2 ef=$3 out=$RUN/$1 rc
  [ -f "$out/job_meta.json" ] && grep -q '"rc":0,"aborted":""' "$out/job_meta.json" && return 0
  [ -e $P2/STOP_batch3 ] && { echo "$(date -u '+%F %T') STOP_batch3 before $tag" >> $LOG; exit 8; }
  [ -d "$out" ] && mv "$out" "${out}.failed.$(date +%H%M%S)"
  local C=$STAGE/b3/cache/s$PR/$role/$HT-g$GPU HO=0; [ "$HT" = n1 ] && HO=1000; local PB=${PORT:-$(( 31000 + 200 * GPU + HO ))}
  EXTRA_ENV="VLLM_PORT=$PB MASTER_PORT=$((PB + 50)) SGLANG_PORT_BASE=$((PB + 100)) VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton" JOB_TIMEOUT=1200 WATCH_FRAC=97 \
    bash $B3/scripts/b3_job.sh b3c_${PR}_${LABEL}_$tag $GPU $STAGE/b3/venvs/${PR}_$role $out -- \
    bash $P2/screen/with_env.sh $ef $B3/bench/run_b3.sh $out timed > $RUN/$tag.joblog 2>&1; rc=$?
  echo "$(date -u '+%F %T') end $tag rc=$rc $(tail -1 $out/job_meta.json 2>/dev/null | cut -c1-120)" >> $LOG
  [ $rc = 0 ] || { echo "$(date -u '+%F %T') $tag failed rc=$rc: stop" >> $LOG; exit 10; }
}
if [ "$MODE" = cal ]; then
  job cor_1_parent parent $EK; job cor_2_parent parent $EK; job cor_3_fix fix $EK
  job wu_1_parent parent $EC; job wu_2_fix fix $EC
  for i in 1 2 3 4 5; do ii=$(printf %02d $i)
    if [ $((i % 2)) = 1 ]; then job ab${ii}_1_parent parent $EC; job ab${ii}_2_fix fix $EC; else job ab${ii}_1_fix fix $EC; job ab${ii}_2_parent parent $EC; fi
    job aa${ii}_1_parent parent $EC; job aa${ii}_2_parent parent $EC
  done
else
  job wu_1_parent parent $EC; job wu_2_fix fix $EC; job cor_3_fix fix $EK
  job ab01_1_fix fix $EC; job ab01_2_parent parent $EC; job ab02_1_parent parent $EC; job ab02_2_fix fix $EC; job ab03_1_fix fix $EC; job ab03_2_parent parent $EC
  if [ "$TRF" = --transfer ]; then for t in $TRANSFERS; do job tr_${t}_1_parent parent $B3/env/${PR}_tr_$t.env; job tr_${t}_2_fix fix $B3/env/${PR}_tr_$t.env; done; fi
fi
echo "$(date -u '+%F %T') DONE $MODE $LABEL" >> $LOG
