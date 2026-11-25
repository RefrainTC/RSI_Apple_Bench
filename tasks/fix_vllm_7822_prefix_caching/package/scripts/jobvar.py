#!/usr/bin/env python3
"""jobvar.py <taskdir> <label> [<label> ...] [--write] : between-job spread of the human fix's score (coordinator
2026-11-23). Each label is one independent score.sh job of the human fix. Prints per job: GPU, speedup, pair values,
median cand / parent TPOT and per-pair TTFT ratio (cand / parent). Then REF = median of the job speedups, SD between
jobs, band = max(0.5%, 2 x SD / REF), and the parent-TPOT drift across jobs. --write saves calib/ref_jobs.json, which
score_phase2.sh then uses for REF and the verdict band."""
import json, os, statistics as st, sys
td = sys.argv[1]; labels = [a for a in sys.argv[2:] if not a.startswith("--")]; rows = []
for l in labels:
    R = os.path.join(td, "results", l)
    try:
        s = json.load(open(os.path.join(R, "summary.json"))); runs = json.load(open(os.path.join(R, "runs_summary.json")))["runs"]
    except Exception as e:
        print(f"skip {l}: {e}"); continue
    if not s.get("speedup"):
        print(f"skip {l}: no speedup ({s.get('verdict')})"); continue
    gpu = {"0-7,32-39": "0", "8-15,40-47": "1", "16-23,48-55": "2"}.get((s.get("cpus") or ["?"])[0], "?")   # CCD pinning per GPU
    sc = {k: v for k, v in runs.items() if v.get("label") == "sc" and v.get("kind") == "ab"}
    tp = {r: [v["metric"]["primary"] for v in sc.values() if v.get("role") == r and v.get("metric")] for r in ("cand", "parent")}
    tt = {}
    for v in sc.values():
        if v.get("metric"): tt.setdefault(v["ii"], {})[v["role"]] = v["metric"].get("median_ttft_ms")
    ttr = [round(x["cand"] / x["parent"], 4) for _, x in sorted(tt.items()) if x.get("cand") and x.get("parent")]
    rows.append({"label": l, "gpu": gpu, "speedup": s["speedup"], "pairs": (s.get("pairs") or {}).get("values"), "verdict": s.get("verdict"),
                 "cand_tpot": round(st.median(tp["cand"]), 4) if tp["cand"] else None, "parent_tpot": round(st.median(tp["parent"]), 4) if tp["parent"] else None,
                 "ttft_ratios": ttr, "cpus": s.get("cpus")})
for r in rows: print(json.dumps(r))
sp = [r["speedup"] for r in rows]; par = [r["parent_tpot"] for r in rows if r["parent_tpot"]]; cnd = [r["cand_tpot"] for r in rows if r["cand_tpot"]]
if len(sp) >= 2:
    ref = st.median(sp); sd = st.stdev(sp); band = max(0.005, 2 * sd / ref)
    out = {"ref_speedup": round(ref, 5), "verdict_band": round(band, 5), "n_jobs": len(sp), "sd": round(sd, 5), "rel_sd": round(sd / ref, 5),
           "min": min(sp), "max": max(sp), "parent_tpot_rel_sd": round(st.stdev(par) / st.mean(par), 5) if len(par) > 1 else None,
           "parent_tpot_range": [min(par), max(par)] if par else None, "cand_tpot_rel_sd": round(st.stdev(cnd) / st.mean(cnd), 5) if len(cnd) > 1 else None,
           "calib_ref": json.load(open(os.path.join(td, "calib", "calib.json")))["ref_speedup"], "jobs": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "jobs"}))
    if "--write" in sys.argv:
        if len(sp) < 5: sys.exit("fewer than 5 jobs, not written")
        json.dump(out, open(os.path.join(td, "calib", "ref_jobs.json"), "w"), indent=1); print("written calib/ref_jobs.json")
