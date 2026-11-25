#!/bin/bash
# smoke.sh : iso12380 smoke on this pod's GPU 0 (not a calibration): parent timed, fix timed, parent refout, parent and
# fix correctness runs; then harness sanity, correctness compare, output + step-trace check, settings diff, witness.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
. $ISO_ROOT/ops_iso12380/common.sh
stop_check; gpu0_free || exit 3
source $TD/task.env; export P2_AUDIT_IGNORE
exec 9>"$P2/locks/iso12380_${HT}_gpu$GPU.lock"; flock -w 600 9 || { echo "lock timeout"; exit 3; }
SM=$P2/smoke/iso12380_$HT; mkdir -p $SM
[ -f $TD/pristine.sha256 ] || { echo "workspace not built yet"; exit 2; }
SITEP=$("$VENV/bin/python3" -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
SITEF=$("$FIX_VENV/bin/python3" -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
[ -f "$TD/calib/oracle_stage/oracle_stage.json" ] || python3 -I $H/mk_oracle_stage.py "$SITEP" "$SITEF" "$PKG" "$TD/calib/oracle_stage" > $SM/oracle_stage.log 2>&1
export P2_CACHE_MODE=job P2_CACHE_ROOT=$P2T/cache/iso12380/smoke-$HT-$(date +%s); trap 'rm -rf "$P2_CACHE_ROOT"' EXIT
for spec in p1:parent:timed f1:fix:timed lp:parent:refout cp:parent:correct cf:fix:correct; do
  stop_check
  IFS=: read n role mode <<< "$spec"
  [ -f "$SM/$n/job_meta.json" ] && grep -q '"rc":0,"aborted":""' "$SM/$n/job_meta.json" && continue
  echo "$(date '+%F %T') smoke $n $role $mode" >> $SM/smoke.log
  bash $H/run_variant.sh "$TD" $role - "p2k_iso12380_${HT}_$n" "$SM/$n" $mode > "$SM/$n.log" 2>&1
  echo "$(date '+%F %T') smoke $n rc=$? $(tail -1 $SM/$n/job_meta.json 2>/dev/null | cut -c1-300)" >> $SM/smoke.log
done
cd $SM && python3 -I - "$TD" "$SM" <<'PY' > $SM/smoke_report.json 2> $SM/smoke_report.err
import importlib.util, json, os, subprocess, sys
td, sm = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(os.environ["ISO_ROOT"], "harness_iso12380"))
import witness as W, audit_check as AC
def mod(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(td, n + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
M, OC, SE = mod("metric"), mod("outcheck"), mod("settings")
rep = {}
for n in ("p1", "f1", "lp"):
    d = os.path.join(sm, n)
    try:
        m = M.metric(d); meta = json.load(open(os.path.join(d, "job_meta.json")))
        rep[n] = {"metric": m, "rc": meta.get("rc"), "wall_s": meta.get("wall_s"), "peak_mem_mib": meta.get("peak_mem_mib"),
                  "witness": W.witness(d, m["witness_reported_s"]) if m else None,
                  "audit": AC.events(d, True, os.environ.get("P2_AUDIT_IGNORE", "")), "settings": SE.settings(d)}
    except Exception as e:
        rep[n] = {"error": repr(e)}
try:
    lp = json.load(open(os.path.join(sm, "lp", "outputs.json"))); ref = json.load(open(os.path.join(sm, "p1", "outputs.json")))
    f1 = json.load(open(os.path.join(sm, "f1", "outputs.json")))
    rep["lp_vs_p1_ids_identical"] = all(x["ids"] == y["ids"] for p, q in zip(lp["passes"], ref["passes"]) for x, y in zip(p, q))
    s = OC.all_stats(f1, ref, lp, float("inf")); rep["f1_vs_p1_outputs"] = {k: v for k, v in s.items() if k != "bad_sample"}
    rep["f1_vs_p1_steps"] = OC.step_check(os.path.join(sm, "f1"), os.path.join(sm, "p1"))
    s1, s2 = rep["p1"]["settings"], rep["f1"]["settings"]
    rep["settings_diff"] = {k: (s1.get(k), s2.get(k)) for k in set(s1) | set(s2) if s1.get(k) != s2.get(k)}
except Exception as e:
    rep["outputs_error"] = repr(e)
for n in ("cp", "cf"):
    try:
        meta = json.load(open(os.path.join(sm, n, "job_meta.json"))); rep[n] = {"rc": meta.get("rc"), "wall_s": meta.get("wall_s"), "audit": AC.events(os.path.join(sm, n), True, os.environ.get("P2_AUDIT_IGNORE", ""))}
    except Exception as e:
        rep[n] = {"error": repr(e)}
r = subprocess.run(["python3", "-I", os.path.join(td, "compare.py"), os.path.join(sm, "cp", "correct.json"), os.path.join(sm, "cf", "correct.json"), "0", "100"], capture_output=True, text=True)
rep["correct_fix_vs_parent_untied"] = r.stdout[-3000:]
r = subprocess.run(["python3", "-I", os.path.join(os.environ["ISO_ROOT"], "harness_iso12380", "sanity_run.py"), td, os.path.join(sm, "p1")], capture_output=True, text=True)
rep["sanity_p1"] = r.stdout[-2000:]
print(json.dumps(rep, indent=1, default=str))
PY
echo "$(date '+%F %T') SMOKE DONE" >> $SM/smoke.log
