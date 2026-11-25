#!/usr/bin/env python3
"""jobvar14930.py <taskdir> <label> ... [--write] : between-job spread of the human fix's score for #14930 (adapted from
scripts/jobvar.py: GPU3 in the CCD map, metric = median-of-reps wall). Per job: GPU, speedup, pair values, median cand /
parent metric, verdict, flags. Then REF = median of job speedups, SD, band = max(0.5 %, 2 SD / REF), parent drift."""
import json, os, statistics as st, sys
td = sys.argv[1]; labels = [a for a in sys.argv[2:] if not a.startswith("--")]; rows = []
CCD = {"0-7,32-39": "0", "8-15,40-47": "1", "16-23,48-55": "2", "24-31,56-63": "3"}
for l in labels:
    R = os.path.join(td, "results", l)
    try: s = json.load(open(os.path.join(R, "summary.json"))); runs = json.load(open(os.path.join(R, "runs_summary.json")))["runs"]
    except Exception as e: print(f"skip {l}: {e}"); continue
    if not s.get("speedup"): print(f"skip {l}: no speedup ({s.get('verdict')})"); continue
    sc = {k: v for k, v in runs.items() if v.get("label") == "sc" and v.get("kind") == "ab"}
    m = {r: [v["metric"]["primary"] for v in sc.values() if v.get("role") == r and v.get("metric")] for r in ("cand", "parent")}
    reps = {r: [x for v in sc.values() if v.get("role") == r and v.get("metric") for x in v["metric"]["reps_s"]] for r in ("cand", "parent")}
    rows.append({"label": l, "gpu": CCD.get((s.get("cpus") or ["?"])[0], "?"), "speedup": round(s["speedup"], 5), "pairs": (s.get("pairs") or {}).get("values"),
                 "verdict": s.get("verdict"), "flags": s.get("flags"), "cand_s": round(st.median(m["cand"]), 4), "parent_s": round(st.median(m["parent"]), 4),
                 "cand_reps": reps["cand"], "parent_reps": reps["parent"], "transfer": {k: v.get("median_speedup") for k, v in (s.get("transfer") or {}).items()}})
for r in rows: print(json.dumps({k: v for k, v in r.items() if not k.endswith("_reps")}))
sp = [r["speedup"] for r in rows]; par = [r["parent_s"] for r in rows]; cnd = [r["cand_s"] for r in rows]
if len(sp) >= 2:
    ref = st.median(sp); sd = st.stdev(sp)
    out = {"ref_speedup": round(ref, 5), "verdict_band": round(max(0.005, 2 * sd / ref), 5), "n_jobs": len(sp), "sd": round(sd, 5), "rel_sd": round(sd / ref, 5),
           "min": min(sp), "max": max(sp), "gpus": sorted({r["gpu"] for r in rows}), "parent_rel_sd": round(st.stdev(par) / st.mean(par), 5), "cand_rel_sd": round(st.stdev(cnd) / st.mean(cnd), 5),
           "parent_range": [min(par), max(par)], "cand_range": [min(cnd), max(cnd)], "calib_ref": json.load(open(os.path.join(td, "calib", "calib.json")))["ref_speedup"]}
    print(json.dumps(out))
    if "--write" in sys.argv: json.dump(dict(out, jobs=rows), open(os.path.join(td, "calib", "ref_jobs.json"), "w"), indent=1); print("written calib/ref_jobs.json")
