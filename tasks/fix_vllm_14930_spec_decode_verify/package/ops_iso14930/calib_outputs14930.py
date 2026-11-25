#!/usr/bin/env python3
"""calib_outputs14930.py <taskdir> [sampled_tol] : #14930 scored-output thresholds from trusted runs of the calibration
(every cal_* fix / parent / fixvenv / parentvenv run on the official load) vs calib/out_ref.json (a parent run).
  min_exact    = smallest benign exactly-identical fraction (per rep) - 2 binomial SE (64 requests), capped at 1.0
  min_same_inc = same for "per-request increment sequence identical"
  inc_tol      = max(0.005, 3 x largest benign |relative difference| in mean tokens per request-step)
  sampled_tol  = given (default 0.05); checked afterwards on temperature-1 transfer pairs
Writes calib.json["outcheck_thresholds"] and calib/outcheck_calibration.json."""
import glob, importlib.util, json, math, os, sys
td = os.path.abspath(sys.argv[1]); C = os.path.join(td, "calib"); STOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
spec = importlib.util.spec_from_file_location("oc", os.path.join(td, "outcheck.py")); OC = importlib.util.module_from_spec(spec); spec.loader.exec_module(OC)
ref = json.load(open(os.path.join(C, "out_ref.json"))); key = OC._key(ref); w = ref["workload"]
ex, si, tr, per = [], [], [], {}
for d in sorted(glob.glob(os.path.join(C, "runs", "cal_*"))):
    n = os.path.basename(d); role = n.rsplit("_", 1)[1]
    if ".aborted." in n or "_warm" in n or role not in ("fix", "fixvenv", "parent", "parentvenv"): continue
    p = os.path.join(d, "outputs.json")
    if not os.path.exists(p): continue
    o = json.load(open(p))
    if OC._key(o) != key: continue
    rows = []
    for r in range(len(ref["reps"])):
        s = OC.rep_stats(o["reps"][r], ref["reps"][r], w["OL"], w["K"], True)
        rows.append({k: s[k] for k in ("exact_frac", "same_inc_frac", "tps_rel", "bad")}); ex.append(s["exact_frac"]); si.append(s["same_inc_frac"]); tr.append(abs(s["tps_rel"] or 0))
    per[n] = rows
n = w["N"]
def lo(v): m = min(v); return min(1.0, max(0.0, m - 2 * math.sqrt(max(m * (1 - m), 1e-9) / n))) if m < 1 else 1.0
th = {"min_exact": lo(ex), "min_same_inc": lo(si), "inc_tol": max(0.005, 3 * max(tr)), "sampled_tol": STOL,
      "benign_min_exact": min(ex), "benign_min_same_inc": min(si), "benign_max_tps_rel": max(tr), "n_runs": len(per)}
c = json.load(open(os.path.join(C, "calib.json"))); c["outcheck_thresholds"] = th
json.dump(c, open(os.path.join(C, "calib.json"), "w"), indent=1)
json.dump({"thresholds": th, "per_run": per}, open(os.path.join(C, "outcheck_calibration.json"), "w"), indent=1)
print(json.dumps(th))
