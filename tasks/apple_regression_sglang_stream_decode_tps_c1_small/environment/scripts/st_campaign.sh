#!/bin/bash
# st_campaign.sh <gpu> <job> [<job> ...] : ST verifier passes one after another on one GPU, each result copied to $RSI_OUT.
#   job = <ws name>:<arm fix|noop|f1|f2>:<result name>     e.g. ws_q_fix_g3:fix:or1_g3
#   env: RSI_ST_MODEL (HF repo id of the substitute model), RSI_EXTRA_SERVE_ARGS (both arms).
#   The workspace is (re)staged from the pinned tree right before its first use in this campaign.
# rc 3 (GPU busy) / 4 (foreign process) -> move the result aside, wait 10 min, retry (max 4 tries).
S="$(cd "$(dirname "$0")" && pwd)"; R=${RSI_OUT:-${RSI_ROOT:?set RSI_ROOT}/out}/results
RES=${RSI_ROOT:?set RSI_ROOT}/st/res; L=${RSI_ROOT:?set RSI_ROOT}/st/logs; export RSI_ROOT RSI_OUT RSI_TASK_DIR RSI_CUDA_HOME RSI_CPU_MAP RSI_ARCH
gpu=$1; shift
mkdir -p $R $L
LOG=$L/st_campaign_gpu$gpu.log
: "${RSI_ST_MODEL:?}"; export RSI_ST_MODEL RSI_EXTRA_SERVE_ARGS RSI_HARNESS_DIR
declare -A staged
for job in "$@"; do
  IFS=: read ws arm res <<< "$job"
  if [ -f $L/STOP_st_gpu$gpu ]; then echo "$(date '+%F %T') STOP file present, ending before $res" >> $LOG; break; fi
  if [ -z "${staged[$ws]:-}" ]; then
    RSI_GPU=$gpu bash $S/stage_ws.sh $ws $arm > $L/stage_${ws}.log 2>&1 \
      || { echo "$(date '+%F %T') STAGE FAILED $ws:$arm" >> $LOG; continue; }
    staged[$ws]=1
  fi
  for try in 1 2 3 4; do
    echo "$(date "+%F %T") start $res ws=$ws arm=$arm gpu=$gpu try=$try model=$RSI_ST_MODEL extra=\"$RSI_EXTRA_SERVE_ARGS\" harness=${RSI_HARNESS_DIR:-default}" >> $LOG
    RSI_GPU=$gpu bash $S/run_phase.sh $ws $res 2 > $L/${res}.log 2>&1; rc=$?
    sp=$(grep -o '"speedup[a-z_]*": [0-9.]*' $RES/$res/verifier/reward.json 2>/dev/null | head -1)
    rw=$(grep -o '"reward": [0-9.]*' $RES/$res/verifier/reward.json 2>/dev/null | head -1)
    echo "$(date '+%F %T') end $res rc=$rc $sp $rw" >> $LOG
    if [ $rc -eq 3 ] || [ $rc -eq 4 ]; then mv $RES/$res $RES/_aborted_${res}_$(date +%H%M%S) 2>/dev/null; sleep 600; continue; fi
    break
  done
  cp -a $RES/$res $R/ 2>/dev/null; cp -a $L/${res}.log $L/mem_st_${res}.* $R/$res/ 2>/dev/null
done
echo "$(date '+%F %T') CAMPAIGN DONE gpu=$gpu" >> $LOG
