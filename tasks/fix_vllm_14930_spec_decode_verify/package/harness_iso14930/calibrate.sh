#!/bin/bash
# calibrate.sh <taskdir> [n_ab=5] [n_aa=5] : oracle (human fix as overlay) vs parent, A/A, overlay-equivalence,
# correctness reference, scored-output reference, witness tolerance, effective-settings fingerprint, integrity manifests.
# Writes $TD/calib/{oracle_stage/, correct_ref.json, out_ref*.json, runs/, summary.json, calib.json}. Uses the task's GPU.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TD=$(readlink -f "$1"); NAB=${2:-5}; NAA=${3:-5}; H=$ISO_ROOT/harness_iso14930
source "$TD/task.env"
export OUT_MIN_EXACT=${OUT_MIN_EXACT:-0} OUT_TIE_NATS SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE 2>/dev/null   # calibration: no output threshold (it is derived here)
: "${CORRECT_MIN_EXACT:?task.env must set CORRECT_MIN_EXACT}" "${CORRECT_TIE_NATS:?task.env must set CORRECT_TIE_NATS}"
# compile caches (staged tasks): a fresh root for this job; warm-ups fill it, every run reads its own copy; deleted at exit
if [ -n "${P2_CACHE_ROOT:-}" ]; then
  export P2_CACHE_MODE=job P2_CACHE_ROOT=$P2T/cache/$TASK/job-calib-$(date +%s)
  trap 'rm -rf "$P2_CACHE_ROOT"' EXIT
fi
C=$TD/calib; mkdir -p $C/runs; LOG=$C/calibrate.log
echo "$(date '+%F %T') CALIBRATE START storage=${P2_STORAGE:-nas} cpus=$(bash $ISO_ROOT/scripts/p2_cpus.sh $GPU) fresh_venv=${P2_FRESH_VENV:-0} audit=${AUDIT:-0}" >> $LOG
# scorer + venv manifests before any run (the reference part is added at the end)
bash $H/integrity.sh make "$TD" >> $LOG 2>&1
SITEP=$("$VENV/bin/python3" -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
SITEF=$("$FIX_VENV/bin/python3" -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
python3 -I $H/mk_oracle_stage.py "$SITEP" "$SITEF" "$PKG" "$C/oracle_stage" >> $LOG 2>&1
echo "$(date '+%F %T') oracle stage: $(cat $C/oracle_stage/oracle_stage.json | tr -d '\n' | cut -c1-300)" >> $LOG
# correctness: parent twice (determinism), fix once, against parent #1
for t in ref1:parent ref2:parent fixc:fix; do
  n=${t%%:*}; r=${t##*:}
  [ -f "$C/runs/correct_$n/correct.json" ] || bash $H/run_variant.sh "$TD" $r - "p2c_${TASK}_correct_$n" "$C/runs/correct_$n" correct >> $LOG 2>&1
done
# harness sanity on the first run (audit records, venv guard, metadata): a harness bug must not spoil a calibration
python3 -I $H/sanity_run.py "$TD" "$C/runs/correct_ref1" >> $LOG 2>&1 || { echo "$(date '+%F %T') SANITY FAIL on correct_ref1: calibration stopped" >> $LOG; exit 1; }
cp "$C/runs/correct_ref1/correct.json" "$C/correct_ref.json" 2>/dev/null
for n in ref2 fixc; do
  echo "correct $n vs ref1: $(python3 -I $TD/compare.py $C/correct_ref.json $C/runs/correct_$n/correct.json $CORRECT_MIN_EXACT $CORRECT_TIE_NATS)" >> $LOG
done
# scored-output reference (offline tasks with outputs.json): one parent run on the official load + the same with logprobs
if [ "${OUTREF:-0}" = 1 ]; then
  for t in outref:timed outreflp:refout; do n=${t%%:*}; m=${t##*:}
    [ -f "$C/runs/$n/outputs.json" ] || bash $H/run_variant.sh "$TD" parent - "p2c_${TASK}_$n" "$C/runs/$n" $m ${SCORE_WORKLOAD:-} >> $LOG 2>&1
  done
  cp "$C/runs/outref/outputs.json" "$C/out_ref.json"; cp "$C/runs/outreflp/outputs.json" "$C/out_ref_lp.json"
  echo "$(date '+%F %T') out_ref: $(python3 -I -c "
import json; a=json.load(open('$C/out_ref.json')); b=json.load(open('$C/out_ref_lp.json'))
same=[sum(x['ids']==y['ids'] for x,y in zip(a['passes'][k],b['passes'][k])) for k in (0,1)]
n=len(a['passes'][0]); r0=a['passes'][0]
print('requests', n, 'ids identical without/with logprobs per pass', same, 'distinct outputs in pass 0', len({tuple(x['ids']) for x in r0}))")" >> $LOG
fi
# timing pairs: fix vs parent (ABBA), parent vs parent (A/A), fixvenv vs parentvenv (overlay equivalence, 1 pair)
python3 -I $H/pairs.py "$TD" "$C/runs" cal --plan "ab:fix:parent:$NAB,aa:parent:parent:$NAA,xv:fixvenv:parentvenv:1" --warmup parent,fix \
  --warmup-workload "${WARMUP_WORKLOAD:-}" \
  --workload "${SCORE_WORKLOAD:-}" >> $LOG 2>&1
python3 -I $H/summarize.py "$TD" "$C/runs" --label cal --calib /dev/null > $C/summary.json
python3 -I $H/calib_summary.py "$C/summary.json" "$C/calib.json"
# manifests again, now including the references (scorer and venv hashes must be unchanged)
bash $H/integrity.sh make "$TD" >> $LOG 2>&1
echo "$(date '+%F %T') CALIBRATE DONE" >> $LOG
