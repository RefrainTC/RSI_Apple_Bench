#!/usr/bin/env python3
"""jobs_summary.py <taskdir> <label> [<label> ...] : between-job REF of the human fix (iso13365 version of jobvar.py:
host-aware). Per job: host, CPUs, verdict, speedup (median of its pairs), pair values, parent / fix medians, GPU minutes
(sum of job_meta wall_s of every run of the job, incl. correctness run and warm-ups), elapsed minutes, peak VRAM.
Then median REF, range, SD (sample), relative SD, band = max(0.5 %, 2 SD / REF)."""
import glob, json, os, re, statistics as st, sys
td = sys.argv[1]; rows = []
for l in sys.argv[2:]:
    R = os.path.join(td, "results", l)
    try:
        s = json.load(open(os.path.join(R, "summary.json"))); rs = json.load(open(os.path.join(R, "runs_summary.json")))
    except Exception as e:
        print(f"skip {l}: {e}"); continue
    log = open(os.path.join(R, "score.log")).read() if os.path.exists(os.path.join(R, "score.log")) else ""
    m = re.search(r"host=(\S+)", log); host = "n1" if (m and "-n1-" in m.group(1)) else ("main" if m else "?")
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(R, "**", "job_meta.json"), recursive=True)]
    gpu_min = sum(float(x.get("wall_s") or 0) for x in metas) / 60
    t = [float(x["start"]) for x in metas] + [float(x["end"]) for x in metas]
    sc = [v for v in rs["runs"].values() if v.get("label") == "sc"]
    med = lambda role: round(st.median([v["metric"]["primary"] for v in sc if v["role"] == role and v.get("metric")]), 4) if sc else None
    rows.append({"label": l, "host": host, "cpus": s.get("cpus"), "verdict": s.get("verdict"), "speedup": s.get("speedup"),
                 "pairs": (s.get("pairs") or {}).get("values"), "parent_med_s": med("parent"), "cand_med_s": med("cand"),
                 "gpu_min": round(gpu_min, 2), "elapsed_min": round((max(t) - min(t)) / 60, 2) if t else None,
                 "runs": len(metas), "peak_mib": max(int(x.get("peak_mem_mib") or 0) for x in metas) if metas else None,
                 "env_suspect": s.get("environment_suspect_pairs"), "correct": (s.get("correctness") or {}).get("pass"),
                 "flags": s.get("flags")})
for r in rows:
    print(json.dumps(r))
sp = [r["speedup"] for r in rows if r["speedup"]]
if len(sp) >= 2:
    ref = st.median(sp); sd = st.stdev(sp)
    print(json.dumps({"n_jobs": len(sp), "hosts": sorted({r["host"] for r in rows}), "ref_median": round(ref, 5), "min": round(min(sp), 5),
                      "max": round(max(sp), 5), "sd": round(sd, 5), "rel_sd": round(sd / ref, 5), "band": round(max(0.005, 2 * sd / ref), 5),
                      "parent_med_range": [min(r["parent_med_s"] for r in rows), max(r["parent_med_s"] for r in rows)],
                      "gpu_min_median": st.median([r["gpu_min"] for r in rows]), "peak_mib_max": max(r["peak_mib"] or 0 for r in rows)}))
