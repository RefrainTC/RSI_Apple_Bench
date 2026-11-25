#!/bin/bash
# smoke.sh [gpu=1] : harness smoke on one GPU before calibration (dev-warm caches, not a measurement):
# parent timed x2, fix timed x1, parent / fix correct. Then outcheck, compare, settings, audit, sanity.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
G=${1:-1}; S=$TD/smoke; mkdir -p $S
source $TD/task.env
SITEP=$($VENV/bin/python3 -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
SITEF=$($FIX_VENV/bin/python3 -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
[ -d $TD/calib/oracle_stage ] || python3 -I $H/mk_oracle_stage.py "$SITEP" "$SITEF" vllm $TD/calib/oracle_stage
for t in p1:parent:timed f1:fix:timed p2:parent:timed cp:parent:correct cf:fix:correct; do
  IFS=: read n r m <<< "$t"
  [ -f $S/$n/job_meta.json ] && grep -q '"rc":0,"aborted":""' $S/$n/job_meta.json && continue
  stop_check "smoke $n"; gpu_free $G || exit 3
  log "smoke $n ($r $m) start $(hostctx)"
  GPU=$G bash $H/run_variant.sh $TD $r - p2x_iso13365_smoke_$n $S/$n $m > $S/$n.log 2>&1; rc=$?
  log "smoke $n rc=$rc $(tail -1 $S/$n/job_meta.json 2>/dev/null | cut -c1-400)"
done
cd $TD
python3 -I - <<'PY'
import json, os, sys, importlib.util
td = os.getcwd(); S = os.path.join(td, "smoke")
def mod(n):
    sp = importlib.util.spec_from_file_location(n, os.path.join(td, n + ".py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
M, OC, SE = mod("metric"), mod("outcheck"), mod("settings")
for n in ("p1", "f1", "p2"):
    m = M.metric(os.path.join(S, n)) or {}
    print(n, "primary", m.get("primary"), "timed", [round(x, 4) for x in m.get("timed_s") or []], "prime", [round(x, 3) for x in m.get("prime_s") or []], "warm", [round(x, 3) for x in m.get("warm_s") or []], "fe_steps", m.get("fe_steps"), "core_pids", m.get("core_pids"), "runner", m.get("runner_pid"))
for a, b in (("f1", "p1"), ("p2", "p1")):
    r = OC.check(os.path.join(S, a), os.path.join(S, b), "/nonexistent")
    print("outcheck", a, "vs", b, json.dumps({k: r[k] for k in ("pass", "exact", "total", "bad", "core_equal", "fe_steps_equal", "accepted_timed", "proposed_timed", "core_diff", "bad_sample")})[:900])
    sa, sb = SE.settings(os.path.join(S, a)), SE.settings(os.path.join(S, b))
    print("settings diff", {k: (sa.get(k), sb.get(k)) for k in set(sa) | set(sb) if sa.get(k) != sb.get(k)}, "required_ok", sa["_required_ok"], sb["_required_ok"])
print("settings p1", json.dumps(SE.settings(os.path.join(S, "p1"))))
o = json.load(open(os.path.join(S, "p1", "outputs.json")))
for i, c in enumerate(o["core"]): print("core", i, json.dumps(c)[:300])
PY
python3 -I $TD/compare.py $S/cp/correct.json $S/cf/correct.json 1 0 | cut -c1-1500
python3 -I -c "
import json; c=json.load(open('$S/cp/correct.json')); print('live', c['live']); print('unit', c['unit']['api'], c['unit'].get('n'), c['unit'].get('s'), 'none:', sum(r is None for r in c['unit']['results']))
for g,v in c['groups'].items(): print(g, v['core'], v['fe'], v['s'])
f=json.load(open('$S/cf/correct.json')); print('fix unit s', f['unit'].get('s'))"
for n in p1 f1 p2 cp cf; do echo "sanity $n: $(python3 -I $H/sanity_run.py $TD $S/$n | cut -c1-600)"; done
grep -h -c . $S/*/audit.*.jsonl 2>/dev/null | head; log "smoke done"
