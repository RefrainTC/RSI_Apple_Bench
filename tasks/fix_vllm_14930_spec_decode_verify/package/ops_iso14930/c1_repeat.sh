#!/bin/bash
# c1_repeat.sh <gpu> : canary C1 (skip verification) vs parent on the SCREEN-like load without decoys (KIND=repeat), 1 AB
# pair + outcheck against the paired parent: does an output-only check catch it when drafts are (almost) always right?
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; O=$P2/ops_iso14930/study/c1_repeat
source $P2/ops_iso14930/guard.sh; source $TD/task.env; export P2_AUDIT_IGNORE P2_AUDIT_NEED_END SETTINGS_IGNORE
exec 9>"$P2/locks/gpu$G.lock"; flock -w 7200 9 || exit 3
stop_check || exit 0; gpu_free $G || exit 3
mkdir -p $O; echo "$(date -Is) start"
GPU=$G python3 -I $H/pairs.py $TD $O/runs rp --plan "ab:cand:parent:1" --warmup parent,cand --warmup-workload SEED_BASE=1000 --stage $TD/results/C1/snap/stage --workload "KIND=repeat"
python3 -I - "$TD" "$O/runs" <<'PY'
import json, os, sys, statistics as st
sys.path.insert(0, sys.argv[1]); import outcheck as OC
r = sys.argv[2]; c = os.path.join(r, "rp_ab01_1_cand"); p = os.path.join(r, "rp_ab01_2_parent")
C, B = json.load(open(os.path.join(c, "outputs.json"))), json.load(open(os.path.join(p, "outputs.json")))
st_ = [OC.rep_stats(C["reps"][i], B["reps"][i], 256, 3, True) for i in range(len(B["reps"]))]
oc = {"identical_frac": min(s["exact_frac"] for s in st_), "min_same_inc_frac": min(s["same_inc_frac"] for s in st_),
      "tps_rel": [s["tps_rel"] for s in st_], "why": [s["bad"] for s in st_], "ref": "paired_parent (direct rep_stats)",
      "parent_mean_tps": [s["ref_mean_tps"] for s in st_]}
m = {k: [x["wall_s"] for x in json.load(open(os.path.join(d, "passes.json")))["reps"]] for k, d in (("cand", c), ("parent", p))}
print(json.dumps({"exact_frac_min": oc.get("identical_frac"), "same_inc_min": oc.get("min_same_inc_frac"), "tps_rel": oc.get("tps_rel"), "why": oc.get("why"), "ref": oc.get("ref"),
                  "speedup": st.median(m["parent"]) / st.median(m["cand"]), "cand": m["cand"], "parent": m["parent"]}))
PY
echo "$(date -Is) C1REPEAT DONE"
