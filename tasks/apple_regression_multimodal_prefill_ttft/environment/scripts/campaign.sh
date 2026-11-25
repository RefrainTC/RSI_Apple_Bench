#!/bin/bash
# campaign.sh <gpu> <job> [<job> ...] : run verifier passes one after another on one GPU, then copy each result to $RSI_OUT.
#   job = <store>:<ws name>:<arm>:<result name>:<seed>   e.g. tmp:ws_fix:fix:or_s20261120:20261120
#   The workspace is (re)staged from the pinned tree right before its first use in this campaign.
# rc 3 (GPU busy) / 4 (foreign process) -> move the result aside, wait 10 min, retry (max 4 tries).
# Serve args (both arms): --attention-backend triton --mem-fraction-static 0.50 (documented values).
S="$(cd "$(dirname "$0")" && pwd)"; R=${RSI_OUT:-${RSI_ROOT:?set RSI_ROOT}/out}/results
RES=${RSI_ROOT:?set RSI_ROOT}/mm/res; L=${RSI_ROOT:?set RSI_ROOT}/mm/logs; export RSI_ROOT RSI_OUT RSI_CPU_MAP RSI_ARCH
gpu=$1; shift
mkdir -p $R $L
LOG=$L/campaign_gpu$gpu.log
export MM_EXTRA_SERVE_ARGS="--attention-backend triton --mem-fraction-static ${MM_MEM_FRACTION:-0.50}"  # 0.60 on an RTX 5090
declare -A staged
for job in "$@"; do
  IFS=: read store ws arm res seed <<< "$job"
  if [ -f $L/STOP_gpu$gpu ]; then echo "$(date '+%F %T') STOP file present, ending before $res" >> $LOG; break; fi
  if [ -z "${staged[$store:$ws]:-}" ]; then
    RSI_GPU=$gpu RSI_STORE=$store bash $S/stage_ws.sh $ws $arm > $L/stage_${store}_${ws}.log 2>&1 \
      || { echo "$(date '+%F %T') STAGE FAILED $store:$ws:$arm" >> $LOG; continue; }
    staged[$store:$ws]=1
  fi
  for try in 1 2 3 4; do
    echo "$(date '+%F %T') start $res store=$store ws=$ws arm=$arm seed=$seed gpu=$gpu try=$try extra=\"$MM_EXTRA_SERVE_ARGS\"" >> $LOG
    RSI_GPU=$gpu RSI_STORE=$store bash $S/run_phase.sh $ws $res $seed 2 > $L/${res}.log 2>&1; rc=$?
    sp=$(grep -o '"speedup": [0-9.]*' $RES/$res/verifier/reward.json 2>/dev/null | head -1)
    rw=$(grep -o '"reward": [0-9.]*' $RES/$res/verifier/reward.json 2>/dev/null | head -1)
    echo "$(date '+%F %T') end $res rc=$rc $sp $rw" >> $LOG
    if [ $rc -eq 3 ] || [ $rc -eq 4 ]; then mv $RES/$res $RES/_aborted_${res}_$(date +%H%M%S) 2>/dev/null; sleep 600; continue; fi
    break
  done
  cp -a $RES/$res $R/ 2>/dev/null; cp -a $L/${res}.log $L/mem_${res}.* $R/$res/ 2>/dev/null
done
echo "$(date '+%F %T') CAMPAIGN DONE gpu=$gpu" >> $LOG
