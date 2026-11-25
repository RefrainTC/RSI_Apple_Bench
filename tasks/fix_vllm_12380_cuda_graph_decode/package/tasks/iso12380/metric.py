"""metric.py : #12380 metric = MEDIAN wall time of the 10 timed generate() reps, lower is better. Also: mean rep (upstream
benchmark_latency's "Avg latency"), per-rep times, step-trace counters, and the witness's reported time (sum of reps,
which run back to back).
Changed from the mean to the median on 2026-11-24 after the first calibration, before any scoring counted: the parent is
CPU-launch-bound, and 2 of 15 parent runs had a single rep slowed by host interference (3.65 s and 2.72 s against a
typical 2.19 s), which put the mean-based A/A at 19 % while the fix (graph replay) never moved (1.445-1.448 s)."""
import json, os, statistics as st
def metric(out):
    p = os.path.join(out, "passes.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); R = d.get("reps") or []
    if not R:
        return None
    w = [r["wall_s"] for r in R]
    return {"primary": st.median(w), "mean_rep_s": sum(w) / len(w), "min_rep_s": min(w), "max_rep_s": max(w),
            "per_rep_s": [round(x, 4) for x in w], "witness_reported_s": sum(w),
            "decode_steps": sum(r["decode_steps"] for r in R), "graph_decode_steps": sum(r["graph_decode_steps"] for r in R),
            "steps": sum(r["steps"] for r in R), "hooks": d.get("hooks"), "process_wall_s": d.get("process_wall_s"),
            "load": [d.get(k) for k in ("bs", "in", "ol", "reps", "seed_base")]}
