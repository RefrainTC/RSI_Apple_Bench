#!/usr/bin/env python3
"""outliers.py <runs dir> ... : per-rep outlier census over timed runs of the official load (passes.json with 7 reps).
Slow rep = wall > 1.05 x the run's median rep. Cause: 'tail' if its engine steps exceed the run's median steps by > 10
(one request left the copy pattern and ran on alone), else 'timing' (same steps, slower wall). Also: what the screen's
metric (sum of reps) would have given vs the median, per role."""
import glob, json, os, statistics as st, sys
rows = []
for root in sys.argv[1:]:
    for p in glob.glob(os.path.join(root, "**", "passes.json"), recursive=True):
        d = os.path.dirname(p)
        if "_warm" in d or ".aborted." in d: continue
        try: j = json.load(open(p)); m = json.load(open(os.path.join(d, "job_meta.json")))
        except Exception: continue
        w = j.get("workload") or {}
        if w.get("REPS") != 7 or w.get("TEMP") != 0 or w.get("K") != 3 or w.get("N") != 64 or w.get("PRIME") != 1 or w.get("SEED_BASE") != 0: continue
        role = os.path.basename(d).rsplit("_", 1)[-1]
        ws = [r["wall_s"] for r in j["reps"]]; ss = [r["steps"] for r in j["reps"]]; mw, ms = st.median(ws), st.median(ss)
        slow = [(i, round(x / mw, 3), "tail" if s > ms + 10 else "timing") for i, (x, s) in enumerate(zip(ws, ss)) if x > 1.05 * mw]
        rows.append({"run": d, "role": role, "gpu": m.get("gpu"), "median": mw, "sum": sum(ws), "slow": slow})
by = {}
for r in rows: by.setdefault(r["role"], []).append(r)
out = {}
for role, rs in by.items():
    nrep = 7 * len(rs); sl = [s for r in rs for s in r["slow"]]
    out[role] = {"runs": len(rs), "reps": nrep, "slow_reps": len(sl), "tail": sum(1 for s in sl if s[2] == "tail"), "timing": sum(1 for s in sl if s[2] == "timing"),
                 "runs_with_slow": sum(1 for r in rs if r["slow"]), "max_slow_ratio": max([s[1] for s in sl] or [None]),
                 "runs_with_ge4_slow": sum(1 for r in rs if len(r["slow"]) >= 4)}
print(json.dumps(out, indent=1))
for r in rows:
    if r["slow"]: print(r["role"], r["gpu"], os.path.relpath(r["run"], os.environ["ISO_ROOT"]), r["slow"])
