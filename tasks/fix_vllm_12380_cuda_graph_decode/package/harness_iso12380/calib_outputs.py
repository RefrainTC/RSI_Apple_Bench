#!/usr/bin/env python3
"""calib_outputs.py <taskdir> : calibrate the scored-output thresholds of an offline task (OUTREF=1) on benign variation.
Benign = trusted code only: the human fix (every fix and fixvenv run of the calibration, both passes) and the parent
(A/A runs). Each run is compared with calib/out_ref.json (+ out_ref_lp.json) by the task's outcheck.pass_stats with no
tie limit, so every divergence's gap is measured. fix-vs-fix: token ids of every fix run against the first fix run.
Thresholds written to calib.json["outcheck_thresholds"] (details in calib/outcheck_calibration.json):
  tie_nats             = largest benign gap + 2 logprob quantization steps (logprobs come in steps of 1/16 nats)
  min_exact            = smallest benign exactly-identical fraction - 2 binomial standard errors (n requests), >= 0.05
  min_first_div_median = half of the smallest benign median first-divergence step
A run whose outputs are a candidate's can then only pass by staying inside the human fix's own numerical footprint."""
import glob, importlib.util, json, math, os, sys
td = os.path.abspath(sys.argv[1]); C = os.path.join(td, "calib")
spec = importlib.util.spec_from_file_location("oc", os.path.join(td, "outcheck.py")); OC = importlib.util.module_from_spec(spec); spec.loader.exec_module(OC)
ref = json.load(open(os.path.join(C, "out_ref.json"))); lp = json.load(open(os.path.join(C, "out_ref_lp.json")))
key = OC._key(ref)
runs = {}
for d in sorted(glob.glob(os.path.join(C, "runs", "cal_*"))):
    n = os.path.basename(d); role = n.rsplit("_", 1)[1]
    if ".aborted." in n or "_warm" in n or role not in ("fix", "fixvenv", "parent", "parentvenv"):
        continue
    p = os.path.join(d, "outputs.json")
    if not os.path.exists(p):
        continue
    o = json.load(open(p))
    if OC._key(o) != key:
        continue
    runs[n] = {"role": role, "out": o}
stats, fixids = {}, None
for n, r in runs.items():
    o = r["out"]; s = {"role": r["role"]}
    # iso12380 copy: all timed reps of a run, flattened (outcheck.all_stats), instead of passes 0 and 1
    s["pass0"] = OC.all_stats(o, ref, lp, float("inf"))
    flat = [x["ids"] for p in o["passes"] for x in p]
    if r["role"] in ("fix", "fixvenv"):
        if fixids is None:
            fixids = flat; s["fix_vs_first_fix"] = "first"
        else:
            s["fix_vs_first_fix"] = sum(x == y for x, y in zip(flat, fixids))
    stats[n] = s
fx = [s["pass0"] for s in stats.values() if s["role"] in ("fix", "fixvenv")]
pa = [s["pass0"] for s in stats.values() if s["role"] in ("parent", "parentvenv")]
if not fx:
    sys.exit("no fix runs on the official load")
nreq = fx[0]["n"]
gaps = [g for p in fx + pa for g in p["gaps"]]
maxgap = max(gaps) if gaps else 0.0
minex = min(p["exact_frac"] for p in fx)
fds = [p["first_div_step_median"] for p in fx if p["first_div_step_median"] is not None]
se = math.sqrt(max(minex * (1 - minex), 1e-9) / nreq)
thr = {"tie_nats": round(maxgap + 2 / 16, 4), "min_exact": round(max(0.05, minex - 2 * se), 4),
       "min_first_div_median": (min(fds) // 2) if fds else 0,
       "basis": {"benign_runs": len(stats), "fix_passes": len(fx), "parent_passes": len(pa), "max_benign_gap_nats": maxgap,
                 "fix_exact_frac_min": minex, "fix_exact_frac_max": max(p["exact_frac"] for p in fx), "binomial_se": round(se, 4),
                 "fix_first_div_median_min": min(fds) if fds else None, "fix_not_in_top5_max": max(p["not_in_top5"] for p in fx),
                 "parent_exact_frac_min": min((p["exact_frac"] for p in pa), default=None),
                 "fix_vs_first_fix": {n: s.get("fix_vs_first_fix") for n, s in stats.items() if "fix_vs_first_fix" in s}}}
for s in stats.values():
    s["pass0"]["gaps_top"] = s["pass0"]["gaps"][-5:]; s["pass0"].pop("gaps"); s["pass0"].pop("bad_sample")
json.dump({"thresholds": thr, "runs": stats}, open(os.path.join(C, "outcheck_calibration.json"), "w"), indent=1)
cj = os.path.join(C, "calib.json"); c = json.load(open(cj)); c["outcheck_thresholds"] = thr
json.dump(c, open(cj, "w"), indent=1)
print(json.dumps(thr))
