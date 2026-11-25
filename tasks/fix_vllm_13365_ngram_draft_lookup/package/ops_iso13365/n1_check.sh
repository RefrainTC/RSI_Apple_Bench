#!/bin/bash
# n1_check.sh : n1 GPU1 environment check before scoring jobs there (dev-warm caches, not a measurement):
# one parent timed run + one fix timed run; reports times, outcheck vs each other, settings, peak VRAM.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
G=1; S=$TD/n1check; mkdir -p $S
source $TD/task.env
[ "$P2T" = $P2T ] || { log "n1_check: not on n1 (P2T=$P2T)"; exit 2; }
for t in p1:parent f1:fix; do
  IFS=: read n r <<< "$t"
  [ -f $S/$n/job_meta.json ] && grep -q '"rc":0,"aborted":""' $S/$n/job_meta.json && continue
  stop_check "n1check $n"; gpu_free $G || exit 3
  log "n1check $n start $(hostctx)"
  GPU=$G bash $H/run_variant.sh $TD $r - p2x_iso13365_n1c_$n $S/$n timed > $S/$n.log 2>&1; rc=$?
  log "n1check $n rc=$rc $(tail -1 $S/$n/job_meta.json 2>/dev/null | cut -c1-300)"
done
cd $TD && python3 -I - <<'PY'
import json, os, importlib.util
td = os.getcwd(); S = os.path.join(td, "n1check")
def mod(n):
    sp = importlib.util.spec_from_file_location(n, os.path.join(td, n + ".py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
M, OC = mod("metric"), mod("outcheck")
for n in ("p1", "f1"):
    m = M.metric(os.path.join(S, n)) or {}; j = json.load(open(os.path.join(S, n, "job_meta.json")))
    print(n, "primary", m.get("primary"), "timed", [round(x, 4) for x in m.get("timed_s") or []], "cpus", j.get("cpus"), "peak", j.get("peak_mem_mib"), "wall", j.get("wall_s"))
r = OC.check(os.path.join(S, "f1"), os.path.join(S, "p1"), td)
print("outcheck f1 vs p1", json.dumps({k: r.get(k) for k in ("ref", "pass", "exact", "total", "core_equal", "fe_steps_equal", "parent_matches_ref")}))
PY
log "n1check done"
